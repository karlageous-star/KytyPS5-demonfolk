#include "graphics/host_gpu/renderer/demonsSouls.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "frame-capture.h"
#include "loader/systemContent.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
// KYTY_LINEAR_COPY_SPANS (default 1; 0 for a same-process A/B): the linear copy accepts a range over consecutive
// mappings (chenxiao 9b50c39a), not only a range inside one mapping.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_linear_copy_spans_mode {1};
}

namespace Libs::Graphics::DemonsSouls {
bool IsSupportedGame() {
	// Called by the dispatch path after game metadata has been loaded.
	static const bool supported = [] {
		// KYTY_DS_PROFILE=0: none of the game's code patches, compute chain or linear copy (A/B tests).
		if (const char* value = std::getenv("KYTY_DS_PROFILE"); value != nullptr && value[0] == '0') {
			return false;
		}
		std::string title;
		return Loader::SystemContentParamSfoGetString("TITLE_ID", &title) && IsSupportedTitle(title);
	}();
	return supported;
}

// Verified shader semantics: dst[i] = src[i % period], i < count. Only the
// linear subset is replaced. Other formats, workgroup shapes, overlaps and
// GPU-owned control words stay on the shader path; CopyBuffer owns coherence.
bool TryLinearCopy(const ShaderComputeInputInfo& input, BufferCache& cache, uint32_t x, uint32_t y,
                   uint32_t z, uint32_t mode) {
	const auto& program = *input.stage.program;
	// The copy kernels by release (their code differs only in the shader id the first instruction loads
	// into m0): dst[i] = src[i % period] (PPSA01341 1.007 eb74563..., PPSA01342 1.004 2a12b35...), and
	// PPSA01342's dst[i] = src[i & mask] (dc18751...), the same user data, i < count; parameters count
	// and period or mask. Only the linear subset is replaced (period >= count; mask keeping every i).
	// KYTY_DS_LINEAR_COPY=0: only PPSA01341's kernel (A/B of the PPSA01342 ones).
	static const bool usa = [] {
		const char* value = std::getenv("KYTY_DS_LINEAR_COPY");
		return value == nullptr || value[0] != '0';
	}();
	const bool modulo = program.shader_hash == 0xeb7456322124ecc7ULL || (usa && program.shader_hash == 0x2a12b35175b11703ULL);
	const bool masked = usa && program.shader_hash == 0xdc18751cb6791ef9ULL;
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

bool LinearCopyMasked(uint64_t shader_hash) {
	return shader_hash == 0xdc18751cb6791ef9ULL;
}

bool TryLinearCopy(const LinearCopyDispatch& dispatch, BufferCache& cache, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t mode) {
	const auto& data   = dispatch.user_data;
	const bool  modulo = !dispatch.masked;
	if (data.size() != 12 || dispatch.thread_dimensions || mode != 0x41 || dispatch.threads[0] != 64 ||
	    dispatch.threads[1] != 1 || dispatch.threads[2] != 1 || !dispatch.group_id[0] || dispatch.group_id[1] ||
	    dispatch.group_id[2] || dispatch.thread_ids != 1 || dispatch.workgroup_register != 12 || dispatch.tg_size ||
	    y != 1 || z != 1 || x == 0)
		return false;
	ShaderBufferResource source, destination, parameters;
	std::memcpy(source.fields, data.data(), 16);
	std::memcpy(destination.fields, data.data() + 4, 16);
	std::memcpy(parameters.fields, data.data() + 8, 16);
	for (const auto* d: {&source, &destination}) {
		// Exact control bits observed for idxen format_x UInt32. Reject every
		// unanalysed format/swizzle/OOB/cache/addressing variation.
		if (d->fields[3] != 0x00014004 || (d->fields[1] & 0xffff0000u) != 0x00040000u ||
		    d->Base48() == 0 || (d->Base48() & 3) != 0)
			return false;
	}
	if (parameters.fields[3] != 0x0004dfac || (parameters.fields[1] & 0xffff0000u) != 0x00100000u ||
	    parameters.NumRecords() != 1 || (parameters.Base48() & 3) != 0)
		return false;
	std::array<uint32_t, 2> controls {};
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(parameters.Base48(), controls.data(),
	                                                           sizeof(controls)))
		return false;
	const uint64_t count = controls[0], period = controls[1], bytes = count * 4;
	// The mask keeps every index below count: all bits of count - 1 and below are set in it.
	const uint64_t index_bits = count == 0 ? 0 : (uint64_t {1} << std::bit_width(count - 1)) - 1;
	const bool     linear     = modulo ? period >= count : (controls[1] & index_bits) == index_bits;
	if (count == 0 || !linear || count > source.NumRecords() ||
	    count > destination.NumRecords() || x != (count + 63) / 64 || bytes > 16 * 1024 * 1024)
		return false;
	const auto src = source.Base48(), dst = destination.Base48();
	if ((src < dst + bytes && dst < src + bytes) ||
	    (parameters.Base48() < dst + bytes && dst < parameters.Base48() + 16))
		return false;
	// Over consecutive mappings (chenxiao 9b50c39a): the game's heaps are mapped a few hundred KiB at a time, and a 4 MiB
	// copy of its constants spanned ten, which took the shader path every frame: its destination GPU-written, every
	// table evaluation reading descriptors there waited for the GPU (~2 drains a frame after the first area). The copy
	// reads and writes through guest addresses (BufferCache::CopyBuffer), so each mapping's part being unique is enough.
	const auto unique = kyty_local_linear_copy_spans_mode.load(std::memory_order_relaxed) != 0
	                        ? &LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings
	                        : &LibKernel::Memory::IsUniqueGuestBackingRange;
	if (!unique(src, bytes) || !unique(dst, bytes) || !unique(parameters.Base48(), 16))
		return false;
	cache.CopyBuffer(dst, src, bytes, false, false);
	cache.ScheduleCopyFeedback(dst, bytes);
	if (FrameCapture::Active()) {
		// Use the validated, executed extents, not the descriptor's whole allocation.
		// Consumed copies return before the generic compute binding capture.
		FrameCapture::g_call.buffers.push_back({src, bytes, FrameCapture::StageCs, 0});
		FrameCapture::g_call.buffers.push_back({dst, bytes, FrameCapture::StageCs, 1});
		FrameCapture::g_call.buffers.push_back({parameters.Base48(), sizeof(controls), FrameCapture::StageCs, 0});
	}
	LiveCounters::Add(LiveCounters::LinearCopies);
	return true;
}
} // namespace Libs::Graphics::DemonsSouls
