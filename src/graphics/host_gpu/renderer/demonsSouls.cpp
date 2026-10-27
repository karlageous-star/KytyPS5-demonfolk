#include "graphics/host_gpu/renderer/demonsSouls.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "loader/systemContent.h"
#include "frame-capture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// KYTY_COMPUTE_CHAIN=0: no chains (10-14, the 8 GB laptop's GPU hangs at 2-2, a TDR in ~half of the 10-minute tours
// with chains: none in 8 tours without them, at ~25% less fps there). Fencing only the culling, list-building or atomic
// programs, a chain length cap of 8, or barriers for every V#, image and BDA read overlap did not stop them, nor did
// fencing the programs that continue chains most (10-18, an RTX 3080 Laptop at the 1-1 arena: with fe938d3a2d22b5c2
// fenced the hangs were rarer, and an RTX 5090 lost 12% of its frame rate).
extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_linear_copy_spans_mode {1};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_compute_chain_mode {[] {
	const char* value = std::getenv("KYTY_COMPUTE_CHAIN");
	return value != nullptr && value[0] == '0' ? 0u : 1u;
}()};
}

namespace Libs::Graphics::DemonsSouls {
bool IsSupportedGame() {
	// Called by the dispatch path after game metadata has been loaded.
	static const bool supported = [] {
		if (const char* value = std::getenv("KYTY_DS_PROFILE"); value != nullptr && value[0] == '0') {
			return false;
		}
		std::string title;
		return Loader::SystemContentParamSfoGetString("TITLE_ID", &title) && IsSupportedTitle(title);
	}();
	return supported;
}

bool ComputeChains() {
	static const bool reported = [] {
		if (kyty_local_compute_chain_mode.load(std::memory_order_relaxed) == 0)
			std::printf("Compute chains: off (KYTY_COMPUTE_CHAIN=0)\n");
		return true;
	}();
	(void)reported;
	return IsSupportedGame() && kyty_local_compute_chain_mode.load(std::memory_order_relaxed) != 0;
}

// The engine's cs_memset32 shader (coredata/enginesupport/shaders/_ps5/cs_memset32) as the builds seen have it:
// 01.007.000's, and 01.005.000's with the period in s12 instead of s8 (the same translated program).
constexpr std::array<uint64_t, 3> LinearCopyShaders {0xeb7456322124ecc7ULL, 0x81bb2b1b9c751ecaULL, 0x2a12b35175b11703ULL};

// The programs whose dispatches never join a chain unless KYTY_CHAIN_FENCE lists others (an empty value fences none).
// fe938d3a2d22b5c2 is the program that continues chains most (10-14: ~50K dispatches in 10 s). 10-18, an RTX 3080
// Laptop (NVIDIA 566.07) on the world tour's 1-1 arena save: with chains the GPU hung (TDR within the first 7 points) in
// 3 of 4 full routes and 1 of 3 short ones; with fe93 fenced, 3 of 3 full and 3 of 3 short routes ran through, at
// 26.2 fps instead of 26.8 (14-17 without chains). Which dispatch it overlaps with there is not found yet.
constexpr std::array<uint64_t, 1> DefaultChainFence {0xfe938d3a2d22b5c2ULL};

bool ComputeChains(uint64_t shader_hash) {
	if (!ComputeChains()) return false;
	static const std::vector<uint64_t> only = [] {
		std::vector<uint64_t> list;
		for (const char* at = std::getenv("KYTY_CHAIN_ONLY"); at != nullptr && *at != '\0';) {
			char*      end  = nullptr;
			const auto hash = std::strtoull(at, &end, 16);
			if (end == at) break;
			list.push_back(hash);
			if (*end != ',' && *end != ';') break;
			at = end + 1;
		}
		if (!list.empty()) std::printf("Compute chains: only %zu programs (KYTY_CHAIN_ONLY)\n", list.size());
		return list;
	}();
	// KYTY_CHAIN_FENCE=<hash>[;<hash>...] and KYTY_CHAIN_FENCE_RANGE=<lo>-<hi> (hex): those programs' dispatches get
	// barriers, every other one chains (DefaultChainFence without KYTY_CHAIN_FENCE; bisections for a chain hang).
	struct Fence {
		std::vector<uint64_t> list;
		uint64_t              lo = 1, hi = 0; // (empty range)
	};
	static const Fence fence = [] {
		Fence       parsed;
		const char* value = std::getenv("KYTY_CHAIN_FENCE");
		if (value == nullptr) parsed.list.assign(DefaultChainFence.begin(), DefaultChainFence.end());
		for (const char* at = value; at != nullptr && *at != '\0';) {
			char*      end  = nullptr;
			const auto hash = std::strtoull(at, &end, 16);
			if (end == at) break;
			parsed.list.push_back(hash);
			if (*end != ',' && *end != ';') break;
			at = end + 1;
		}
		if (const char* range = std::getenv("KYTY_CHAIN_FENCE_RANGE"); range != nullptr && *range != '\0') {
			char* end = nullptr;
			parsed.lo = std::strtoull(range, &end, 16);
			parsed.hi = end != nullptr && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : parsed.lo;
		}
		if (value == nullptr && parsed.lo > parsed.hi)
			std::printf("Compute chains: %zu program(s) never chained by default (KYTY_CHAIN_FENCE= for none)\n",
			            parsed.list.size());
		else if (!parsed.list.empty() || parsed.lo <= parsed.hi)
			std::printf("Compute chains: %zu programs fenced, range %016llx-%016llx (KYTY_CHAIN_FENCE)\n", parsed.list.size(),
			            static_cast<unsigned long long>(parsed.lo), static_cast<unsigned long long>(parsed.hi));
		return parsed;
	}();
	if ((shader_hash >= fence.lo && shader_hash <= fence.hi) ||
	    std::find(fence.list.begin(), fence.list.end(), shader_hash) != fence.list.end())
		return false;
	return only.empty() || std::find(only.begin(), only.end(), shader_hash) != only.end();
}

// Verified shader semantics: dst[i] = src[i % period], i < count. Only the
// linear subset is replaced. Other formats, workgroup shapes, overlaps and
// GPU-owned control words stay on the shader path; CopyBuffer owns coherence.
bool TryLinearCopy(const ShaderComputeInputInfo& input, BufferCache& cache, uint32_t x, uint32_t y,
                   uint32_t z, uint32_t mode) {
	const auto& program = *input.stage.program;
	static const bool usa = [] { const char* value = std::getenv("KYTY_DS_LINEAR_COPY"); return value == nullptr || value[0] != '0'; }();
	const bool masked = usa && program.shader_hash == 0xdc18751cb6791ef9ULL;
	const bool modulo = std::find(LinearCopyShaders.begin(), LinearCopyShaders.end(), program.shader_hash) != LinearCopyShaders.end() &&
	                    (usa || program.shader_hash != 0x2a12b35175b11703ULL);
	if (!IsSupportedGame() || !(modulo || masked) || program.user_data_base != 0)
		return false;
	const LinearCopyDispatch dispatch {.user_data          = input.stage.resources.user_data,
	                                   .threads            = {input.threads_num[0], input.threads_num[1],
	                                                          input.threads_num[2]},
	                                   .group_id           = {input.group_id[0], input.group_id[1], input.group_id[2]},
	                                   .thread_ids         = static_cast<uint32_t>(input.thread_ids_num),
	                                   .workgroup_register = static_cast<uint32_t>(input.workgroup_register),
	                                   .tg_size            = input.tg_size_en,
	                                   .thread_dimensions  = input.dispatch_thread_dimensions,
	                                   .masked             = masked};
	return TryLinearCopy(dispatch, cache, x, y, z, mode);
}

bool LinearCopyMasked(uint64_t shader_hash) { return shader_hash == 0xdc18751cb6791ef9ULL; }

bool TryLinearCopy(const LinearCopyDispatch& dispatch, BufferCache& cache, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t mode) {
	const auto copy = LinearCopyOf(dispatch, x, y, z, mode);
	if (!copy) return false;
	cache.CopyBuffer(copy->dst, copy->src, copy->bytes, false, false, true);
	cache.ScheduleCopyFeedback(copy->dst, copy->bytes);
	if (FrameCapture::Active()) {
		FrameCapture::g_call.buffers.push_back({copy->src, copy->bytes, FrameCapture::StageCs, 0});
		FrameCapture::g_call.buffers.push_back({copy->dst, copy->bytes, FrameCapture::StageCs, 1});
		FrameCapture::g_call.buffers.push_back({copy->parameters, 8, FrameCapture::StageCs, 0});
	}
	LiveCounters::Add(LiveCounters::LinearCopies);
	return true;
}

// (Diagnostic, KYTY_LINEAR_COPY_TRACE: why a dispatch of the copy shader took the shader path.)
static void LinearCopyRefused(const char* reason, uint64_t dst = 0, uint64_t count = 0, uint64_t period = 0) {
	static const bool           trace = std::getenv("KYTY_LINEAR_COPY_TRACE") != nullptr;
	static std::atomic<uint32_t> printed {0};
	if (trace && printed.fetch_add(1) < 300)
		std::printf("LINEAR-COPY-REFUSED %s dst 0x%llx count %llu period %llu\n", reason,
		            static_cast<unsigned long long>(dst), static_cast<unsigned long long>(count),
		            static_cast<unsigned long long>(period));
}

std::optional<LinearCopy> LinearCopyOf(const LinearCopyDispatch& dispatch, uint32_t x, uint32_t y, uint32_t z, uint32_t mode) {
	const auto& data = dispatch.user_data;
	if (data.size() != 12 || dispatch.thread_dimensions || mode != 0x41 || dispatch.threads[0] != 64 ||
	    dispatch.threads[1] != 1 || dispatch.threads[2] != 1 || !dispatch.group_id[0] || dispatch.group_id[1] ||
	    dispatch.group_id[2] || dispatch.thread_ids != 1 || dispatch.workgroup_register != 12 || dispatch.tg_size ||
	    y != 1 || z != 1 || x == 0)
		return LinearCopyRefused("dispatch shape"), std::nullopt;
	ShaderBufferResource source, destination, parameters;
	std::memcpy(source.fields, data.data(), 16);
	std::memcpy(destination.fields, data.data() + 4, 16);
	std::memcpy(parameters.fields, data.data() + 8, 16);
	for (const auto* d: {&source, &destination}) {
		// Exact control bits observed for idxen format_x UInt32. Reject every
		// unanalysed format/swizzle/OOB/cache/addressing variation.
		if (d->fields[3] != 0x00014004 || (d->fields[1] & 0xffff0000u) != 0x00040000u ||
		    d->Base48() == 0 || (d->Base48() & 3) != 0)
			return LinearCopyRefused("buffer descriptor"), std::nullopt;
	}
	if (parameters.fields[3] != 0x0004dfac || (parameters.fields[1] & 0xffff0000u) != 0x00100000u ||
	    parameters.NumRecords() != 1 || (parameters.Base48() & 3) != 0)
		return LinearCopyRefused("parameter descriptor"), std::nullopt;
	std::array<uint32_t, 2> controls {};
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(parameters.Base48(), controls.data(),
	                                                           sizeof(controls)))
		return LinearCopyRefused("parameters not CPU-clean", destination.Base48()), std::nullopt;
	const uint64_t count = controls[0], period = controls[1], bytes = count * 4;
	const uint64_t index_bits = count == 0 ? 0 : (uint64_t {1} << std::bit_width(count - 1)) - 1;
	const bool linear = dispatch.masked ? (period & index_bits) == index_bits : period >= count;
	if (count == 0 || !linear || count > source.NumRecords() ||
	    count > destination.NumRecords() || x != (count + 63) / 64 || bytes > 16 * 1024 * 1024)
		return LinearCopyRefused("count/period", destination.Base48(), count, period), std::nullopt;
	const auto src = source.Base48(), dst = destination.Base48();
	if ((src < dst + bytes && dst < src + bytes) ||
	    (parameters.Base48() < dst + bytes && dst < parameters.Base48() + 16))
		return LinearCopyRefused("overlap", dst, count, period), std::nullopt;
	// (Over consecutive mappings: the game's heaps are mapped a few hundred KiB at a time, and a 4 MiB copy of its
	// constants spanned ten, which took the shader path every frame: its destination GPU-written, every table
	// evaluation reading descriptors there waited for the GPU, ~2 drains a frame after the first area.)
	const auto unique = kyty_local_linear_copy_spans_mode.load(std::memory_order_relaxed) != 0
	                        ? &LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings
	                        : &LibKernel::Memory::IsUniqueGuestBackingRange;
	if (!unique(src, bytes) || !unique(dst, bytes) || !unique(parameters.Base48(), 16)) {
		static const bool trace = std::getenv("KYTY_LINEAR_COPY_TRACE") != nullptr;
		if (trace)
			std::printf("LINEAR-COPY-MAPPINGS src 0x%llx end 0x%llx dst 0x%llx end 0x%llx bytes 0x%llx\n",
			            static_cast<unsigned long long>(src), static_cast<unsigned long long>(LibKernel::Memory::MappingEnd(src)),
			            static_cast<unsigned long long>(dst), static_cast<unsigned long long>(LibKernel::Memory::MappingEnd(dst)),
			            static_cast<unsigned long long>(bytes));
		return LinearCopyRefused(!LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings(src, bytes)   ? "aliased backing (source)"
		                         : !LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings(dst, bytes) ? "aliased backing (destination)"
		                                                                                     : "aliased backing (parameters)",
		                         dst, count, period),
		       std::nullopt;
	}
	return LinearCopy {src, dst, bytes, parameters.Base48()};
}
} // namespace Libs::Graphics::DemonsSouls
