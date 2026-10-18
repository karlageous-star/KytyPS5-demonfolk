#include "common/assert.h"
#include "live-trace-gpu.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/BdaReadPlan.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "frame-capture.h"
#include "frame-gen.h"
#include "native-preparation-state.h"
#include "live-counters.h"

#include <unordered_map>
#include <xxhash.h>
#include "xpr-capture.h"
#include "live-census.h"
#include "live-counters.h"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>

extern "C" {
// KYTY_CHAIN_BDA_READS=1: a dispatch that reaches memory through pointers only to read it chains on its read ranges.
volatile std::atomic_uint32_t kyty_local_chain_bda_reads_mode {0};
// KYTY_DISPATCH_BARRIER_MERGE (live; default 0 until measured): 1 leaves out the write hazard barrier of a dispatch
// that just drained its chain for a range conflict (the drain, compute work before all later commands, and the chain's
// starting barrier, everything before it before compute, already order it), and makes the hazard barrier and the
// indirect argument barrier of a dispatch that starts a chain one barrier (both destination stages). The owner's heavy
// 1-1 scene (OWNER3, 10-19): ~1,000 hazard barriers, ~800 chain drains and ~550 argument barriers a frame.
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_dispatch_barrier_merge_mode {0};
}
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static uint64_t BufferDescriptorSize(const ShaderBufferResource& descriptor) {
	const uint64_t records = descriptor.NumRecords();
	const uint64_t stride  = descriptor.Stride();
	return stride == 0 ? records : records * stride;
}

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = BufferDescriptorSize(source);
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

static bool ResolveComputePatternFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                      uint32_t group_y, uint32_t group_z, uint32_t mode,
                                      ShaderBufferResource& resolved_descriptor,
                                      uint32_t& resolved_clear, uint64_t& resolved_size);

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer& buffer, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z,
                                                uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	// Only a written buffer can be consumed as a clear: without one this is always false (and
	// it ran a texture-cache lookup per buffer on every dispatch).
	if (std::none_of(program.info.buffers.begin(), program.info.buffers.end(),
	                 [](const auto& resource) { return resource.written; })) {
		return false;
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	// Consuming takes ClearMeta of a written buffer, which holds only for a metadata surface
	// (IsMeta): without one, the read checks and the fill evaluation below cannot change the answer.
	bool written_meta = false;
	for (uint32_t i = 0; i < program.info.buffers.size() && !written_meta; i++)
		written_meta = program.info.buffers[i].written &&
		               cache.IsMeta(DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]).Base48());
	if (!written_meta) {
		return false;
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		ShaderBufferResource fill_descriptor;
		uint32_t             fill_value   = 0;
		uint64_t             fill_size    = 0;
		const bool           uniform_fill =
		    ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, fill_descriptor,
		                             fill_value, fill_size) ||
		    ResolveComputePatternFill(input, group_x, group_y, group_z, mode, fill_descriptor,
		                              fill_value, fill_size);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				const bool known =
				    uniform_fill && descriptor.Base48() == fill_descriptor.Base48();
				if (known ? cache.ClearMeta(descriptor.Base48(), fill_value)
				          : cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = BufferDescriptorSize(descriptor);
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

static bool ResolveComputePatternFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                      uint32_t group_y, uint32_t group_z, uint32_t mode,
                                      ShaderBufferResource& resolved_descriptor,
                                      uint32_t& resolved_clear, uint64_t& resolved_size) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	const auto& user_data = resources.user_data;
	if (program.info.buffers.size() != 1 || resources.buffers.size() != 1 ||
	    !program.info.images.empty() || !program.info.samplers.empty() ||
	    program.info.uses_dma || input.dispatch_thread_dimensions || mode != 0x41u ||
	    user_data.size() != 10 || program.user_data_base != 0) {
		return false;
	}
	const auto& resource   = program.info.buffers.front();
	const auto& raw        = resources.buffers.front();
	const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
	if (!resource.written || resource.read || resource.atomic || resource.scalar ||
	    resource.max_byte_extent != 4 ||
	    (resource.formatted && descriptor.Format() != Prospero::BufferFormat::k32UInt) ||
	    (descriptor.Stride() != 4 && descriptor.Stride() != 0) || descriptor.SwizzleEnabled() ||
	    descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    resource.packed_stride != descriptor.PackedStride() || raw.dword_count != 4 ||
	    descriptor.Base48() == 0) {
		return false;
	}
	for (uint32_t i = 0; i < raw.dword_count; i++) {
		if (raw.dwords[i] != user_data[i]) {
			return false;
		}
	}
	const uint32_t clear  = user_data[4];
	const uint32_t period = user_data[9];
	const uint32_t slots  = period == 0u ? 4u : std::min(period, 4u);
	for (uint32_t slot = 1; slot < slots; slot++) {
		if (user_data[4 + slot] != clear) {
			return false;
		}
	}
	if (input.threads_num[0] != 64 || input.threads_num[1] != 1 || input.threads_num[2] != 1 ||
	    group_x == 0 || group_y != 1 || group_z != 1 || !input.group_id[0] || input.group_id[1] ||
	    input.group_id[2] || input.thread_ids_num != 1 || input.wave_size != 64 ||
	    input.tg_size_en) {
		return false;
	}
	const uint64_t count = user_data[8];
	const auto     size  = BufferDescriptorSize(descriptor);
	if (count == 0 || size == 0 || size > UINT32_MAX || count * sizeof(uint32_t) != size ||
	    group_x != (count + input.threads_num[0] - 1) / input.threads_num[0]) {
		return false;
	}
	resolved_descriptor = descriptor;
	resolved_clear      = clear;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size) &&
	    !ResolveComputePatternFill(input, group_x, group_y, group_z, mode, descriptor,
	                               packed_clear, size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		// Track deferred DCC state while the original dispatch writes the metadata allocation.
		cache.TrackDccFill(descriptor.Base48(), size, packed_clear);
		static std::atomic<uint32_t> logged_metadata_clears {0};
		if (logged_metadata_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: metadata fill shader=0x%016" PRIx64
			     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
			     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
		}
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode,
                                    uint64_t indirect_args) {
	FrameCapture::Scope frame_capture("Dispatch", true);
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations(true);
	m_context.GetCommandScheduler().FlushIfDrained();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();
	if (FrameCapture::Active()) {
		FrameCapture::g_call.cs_address = sh_ctx.GetCs().cs_regs.data_addr;
		FrameCapture::g_call.indirect   = indirect_args;
		FrameCapture::g_call.groups[0]  = thread_group_x;
		FrameCapture::g_call.groups[1]  = thread_group_y;
		FrameCapture::g_call.groups[2]  = thread_group_z;
	}
	LiveCensus::Scope census(LiveCensus::Dispatch, sh_ctx.GetCs().cs_regs.data_addr, indirect_args != 0);
	LiveTrace::MarkAfter gpu_mark {[&] { return m_context.GetCommandScheduler().Current().RawHandle(); },
	                               LiveTrace::MarkDispatch, sh_ctx.GetCs().cs_regs.data_addr};
	if (LiveCounters::g_last_dispatch_shader != 0) {
		LiveCounters::Add(LiveCounters::DispatchAfterDispatch);
		if (LiveCounters::g_last_dispatch_shader == sh_ctx.GetCs().cs_regs.data_addr)
			LiveCounters::Add(LiveCounters::DispatchSameShader);
	}
	LiveCounters::g_last_dispatch_shader = sh_ctx.GetCs().cs_regs.data_addr;

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (!ShaderAddressValid(sh_ctx.GetCs().cs_regs.data_addr)) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	// A compute shader the normal path dispatched before (TableDispatchSeen) whose program allows table mode.
	if (kyty_local_table_dispatch_mode.load(std::memory_order_relaxed) != 0 &&
	    (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0 &&
	    (indirect_args != 0 || (thread_group_x != 0 && thread_group_y != 0 && thread_group_z != 0)) &&
	    TableDispatch(buffer, thread_group_x, thread_group_y, thread_group_z, indirect_args))
		return;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	// The linear copy shader, while its code stands: checked from the registers without its program (700+ small
	// copies a frame in Boletaria, each preparing the same program).
	if (indirect_args == 0 && cs_regs.cs_regs.data_addr == m_linear_copy_shader &&
	    ShaderMapGeneration() == m_linear_copy_generation && (++m_linear_copy_uses & 63u) != 0 &&
	    !FrameCapture::Active()) {
		const auto&                           regs = cs_regs.cs_regs;
		const DemonsSouls::LinearCopyDispatch dispatch {
		    .user_data          = std::span<const uint32_t> {cs_regs.cs_user_sgpr.value, regs.user_sgpr},
		    .threads            = {regs.num_thread_x, regs.num_thread_y, regs.num_thread_z},
		    .group_id           = {regs.tgid_x_en, regs.tgid_y_en, regs.tgid_z_en},
		    .thread_ids         = regs.tidig_comp_cnt + 1u,
		    .workgroup_register = regs.user_sgpr,
		    .tg_size            = regs.tg_size_en,
		    .thread_dimensions  = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0,
		    .masked             = m_linear_copy_masked};
		if (DemonsSouls::TryLinearCopy(dispatch, m_context.GetBufferCache(), thread_group_x, thread_group_y,
		                               thread_group_z, mode)) {
			TableDispatchSeen(sh_ctx.GetCs(), true);
			ResetBindings();
			return;
		}
	}

	NativePreparationScratch<ShaderComputeInputInfo> input_storage;
	auto& input_info = input_storage.Get();
	ResetNativeStageInput(input_info);
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	const auto census_shader = sh_ctx.GetCs().cs_regs.data_addr;
	// KYTY_SKIP_EMPTY_INDIRECT=1 (opt-in, off by default; Jetsku f04c9587): an indirect dispatch whose guest
	// arguments no GPU write is pending for says zero groups in some dimension, so it does nothing and is not
	// recorded (no binding, barrier or dispatch). Thread-dimension dispatches ignore the arguments: not skipped.
	static const bool skip_empty_indirect = [] {
		const char* value = std::getenv("KYTY_SKIP_EMPTY_INDIRECT");
		return value != nullptr && std::atoi(value) == 1;
	}();
	if (skip_empty_indirect && indirect_args != 0 && !use_thread_dimensions &&
	    !m_context.GetBufferCache().HasGpuDirtyBytes(indirect_args, 3u * sizeof(uint32_t))) {
		std::array<uint32_t, 3> groups {};
		if (Libs::LibKernel::Memory::TryReadBacking(indirect_args, groups.data(), sizeof(groups)) &&
		    (groups[0] == 0 || groups[1] == 0 || groups[2] == 0)) {
			static std::atomic<uint32_t> skipped_log {0};
			if (skipped_log.fetch_add(1, std::memory_order_relaxed) < 8) {
				LOGF("GraphicsRenderDispatchDirect: KYTY_SKIP_EMPTY_INDIRECT skips an indirect dispatch of "
				     "%ux%ux%u groups\n",
				     groups[0], groups[1], groups[2]);
			}
			return;
		}
	}
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 0);
	std::string unevaluated;
	const auto  shader_generation = ShaderMapGeneration(); // (before the program reads the code)
	const auto  compute_program   = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, sh_regs, input_info, indirect_args != 0 ? &unevaluated : nullptr);
	if (!compute_program && !unevaluated.empty()) {
		// An indirect dispatch whose resource tables do not evaluate (pointers into memory the guest has not
		// mapped, while it loads; a stale descriptor no specialization can take, such as an FMASK surface in a
		// sampled slot, seen in the Nexus): it is skipped. With no thread groups in its arguments the GPU never
		// reads the tables; with some it cannot run as the guest built it, and ending the game helps no one.
		std::array<uint32_t, 3> groups {};
		m_context.GetBufferCache().ReadMemory(indirect_args, sizeof(groups));
		std::memcpy(groups.data(), reinterpret_cast<const void*>(indirect_args), sizeof(groups));
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("Dispatch: an indirect dispatch of %ux%ux%u thread groups skipped, its tables did not evaluate: %s",
			            groups[0], groups[1], groups[2], unevaluated.c_str());
		ResetBindings();
		return;
	}
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 1);
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const uint32_t frame_num = static_cast<uint32_t>(m_context.FrameNumber());
	const bool     large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const auto& program   = *input_info.stage.program;
	const auto& resources = input_info.stage.resources;
	if (LiveCounters::g_dispatch_keys_on.load(std::memory_order_relaxed)) {
		// Local diagnostic: is a dispatch (shader, user data, groups) repeated from last frame, and
		// with the same materialized resources? (What a per-object native dispatch record could reuse.)
		static std::unordered_map<uint64_t, uint64_t> previous, current;
		static uint32_t                               keys_frame = 0;
		if (frame_num != keys_frame) {
			previous.swap(current);
			current.clear();
			keys_frame = frame_num;
		}
		const uint32_t groups[3] {thread_group_x, thread_group_y, thread_group_z};
		uint64_t key = XXH3_64bits_withSeed(resources.user_data.data(), resources.user_data.size() * 4u, program.shader_hash);
		key          = XXH3_64bits_withSeed(groups, sizeof(groups), key ^ indirect_args);
		uint64_t value = XXH3_64bits(resources.flattened_srt.data(), resources.flattened_srt.size() * 4u);
		for (const auto* list: {&resources.buffers, &resources.images, &resources.samplers})
			value = XXH3_64bits_withSeed(list->data(), list->size() * sizeof(ShaderRecompiler::IR::DescriptorValue), value);
		current[key] = value;
		const auto found = previous.find(key);
		LiveCounters::Add(found == previous.end() ? LiveCounters::DispatchKeyNew
		                  : found->second == value ? LiveCounters::DispatchKeySame
		                                           : LiveCounters::DispatchKeyChanged);
	}
	if (FrameCapture::Active()) {
		FrameCapture::g_call.cs = program.shader_hash;
		for (size_t i = 0; i < 3; ++i) FrameCapture::g_call.local[i] = input_info.threads_num[i];
	}
	if (indirect_args == 0 && DemonsSouls::TryLinearCopy(input_info, m_context.GetBufferCache(),
	                                                     thread_group_x, thread_group_y,
	                                                     thread_group_z, mode)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "linear_copy";
		m_linear_copy_shader     = sh_ctx.GetCs().cs_regs.data_addr;
		m_linear_copy_generation = shader_generation;
		m_linear_copy_masked     = DemonsSouls::LinearCopyMasked(program.shader_hash);
		TableDispatchSeen(sh_ctx.GetCs(), true);
		ResetBindings();
		return;
	}
	if (indirect_args == 0 && TryConsumeComputeMetaClear(input_info, buffer, thread_group_x,
	                                                     thread_group_y, thread_group_z, mode)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "meta_clear";
		TableDispatchSeen(sh_ctx.GetCs(), true);
		ResetBindings();
		return;
	}
	if (indirect_args == 0 && TryConsumeComputeImageClear(input_info, buffer, thread_group_x,
	                                                      thread_group_y, thread_group_z, mode)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "image_clear";
		TableDispatchSeen(sh_ctx.GetCs(), true);
		ResetBindings();
		return;
	}
	const bool                   has_sampler = !program.info.samplers.empty();
	static std::atomic<uint32_t> dispatch_log_count {0};
	// The limit is checked before counting: past it, no dispatch pays a locked increment.
	if ((large_workgroup || has_sampler) && dispatch_log_count.load(std::memory_order_relaxed) < 512 &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		     thread_group_z, mode, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(resources.samplers[i]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions && indirect_args != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("GraphicsRenderDispatchDirect: indirect dispatch with thread dimensions at"
			     " 0x%016" PRIx64 " uses host-read counts\n",
			     indirect_args);
		}
		indirect_args = 0;
	}
	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	if (indirect_args == 0 && (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0)) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	phase.emplace(LiveCensus::DispatchPhase, census_shader, 2);
	TableDispatchSeen(sh_ctx.GetCs(), use_thread_dimensions);
	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().CreateComputePipeline(input_info, compute_program);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 3);
	NativePreparationScratch<PreparedBindings> binding_storage(
	    kyty_local_binding_scratch_mode.load(std::memory_order_relaxed) != 0);
	auto& bindings = binding_storage.Get();
	PrepareBindingsInto(input_info.stage, bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 4);
	FindBuffers(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 5);
	PrepareBdaBindings(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 6);
	// Materialize the persistent argument owner before final descriptor handles:
	// a tiny CPU upload must not leave an indirect command using a stream slice
	// that can be retired when preparation rotates the command buffer.
	BufferId indirect_id {};
	if (indirect_args != 0) {
		indirect_id = m_context.GetBufferCache().EnsureBufferContents(indirect_args, 3u * sizeof(uint32_t));
	}
	auto&    buffer_cache = m_context.GetBufferCache();
	uint64_t unregistered = buffer_cache.UnregisterEpoch();
	RebindBuffers(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 7);
	RebindImages(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 8);
	// (A buffer resolved above joined into a new one since - by a later buffer's lookup or an image's refresh - no longer
	// owns its bytes, and the dispatch's writes there would be lost: as PrepareGraphicsBindingsInto.)
	for (uint32_t attempt = 0; attempt < 4 && buffer_cache.UnregisterEpoch() != unregistered; ++attempt) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("Bindings: a buffer was unregistered while a dispatch's bindings were made (CS 0x%016llx), resolved "
			            "again\n",
			            static_cast<unsigned long long>(program.shader_hash));
		unregistered = buffer_cache.UnregisterEpoch();
		RebindBuffers(bindings);
	}
	if (FrameGen::Enabled() && program.shader_hash == FrameGen::TaaShader &&
	    bindings.images.size() > 3) {
		// TAA inputs (docs/RE-DEMONS-SOULS.md): slot 1 depth, slot 3 motion vectors.
		FrameGen::OnDispatch(program.shader_hash, bindings.images[1].image_id,
		                     bindings.images[1].desc.info.data.address, bindings.images[3].image_id,
		                     bindings.images[3].desc.info.data.address);
	}
	if (FrameCapture::Active()) {
		auto& call     = FrameCapture::g_call;
		call.prepared  = true;
		call.indirect  = indirect_args;
		call.groups[0] = thread_group_x;
		call.groups[1] = thread_group_y;
		call.groups[2] = thread_group_z;
		for (uint32_t i = 0; i < bindings.images.size(); ++i) {
			auto image = FrameCapture::FromInfo(bindings.images[i].desc.info, FrameCapture::Texture,
			                                    FrameCapture::StageCs, i);
			if (i < program.info.images.size()) {
				image.storage = program.info.images[i].resource_class ==
				                ShaderRecompiler::IR::ImageResourceClass::Storage;
				image.written = program.info.images[i].written ? 1 : 0;
			}
			call.images.push_back(image);
		}
		for (uint32_t i = 0; i < bindings.buffer_sources.size(); ++i) {
			FrameCapture::Buffer source;
			source.address = bindings.buffer_sources[i].address;
			source.size    = bindings.buffer_sources[i].size;
			source.written = i < program.info.buffers.size() && program.info.buffers[i].written;
			call.buffers.push_back(source);
		}
		if (FrameCapture::WantsData(program.shader_hash)) {
			FrameCapture::AddWords("cs.user", 0, resources.user_data);
			FrameCapture::AddWords("cs.srt", 0, resources.flattened_srt);
			for (uint32_t i = 0; i < bindings.buffer_sources.size(); ++i) {
				FrameCapture::AddGuest("cs.buf" + std::to_string(i), bindings.buffer_sources[i].address,
				                       bindings.buffer_sources[i].size);
			}
		}
	}
	if (XprCapture::Enabled() && XprCapture::IsCullProgram(program.shader_hash)) {
		for (size_t i = 0; i < program.info.buffers.size(); ++i) {
			if (program.info.buffers[i].written) {
				XprCapture::LearnCullOutput(bindings.buffer_sources[i].address,
				                            bindings.buffer_sources[i].size);
			}
		}
	}

	vk::Buffer indirect_buffer = nullptr;
	uint64_t   indirect_offset = 0;
	if (indirect_args != 0) {
		auto& cache = m_context.GetBufferCache();
		// The owner EnsureBufferContents resolved, unless it was deleted or no longer covers the arguments.
		// Live registered buffers do not overlap, so a live cover is the one FindBuffer would return.
		auto* owner = cache.GetBufferIfLive(indirect_id);
		if (owner == nullptr || !owner->IsInBounds(indirect_args, 3u * sizeof(uint32_t))) {
			owner = &cache.GetBuffer(cache.FindBuffer(indirect_args, 3u * sizeof(uint32_t)));
		}
		indirect_buffer = owner->Handle();
		indirect_offset = owner->Offset(indirect_args);
	}

	// KYTY_DS_COMPUTE_CHAIN: 1 (default) the Demon's Souls profile leaves out the barrier between two
	// compute dispatches only when neither writes what the other reads or writes (bound buffer and
	// image ranges; dispatches that reach memory through pointers always wait); 2 leaves it out
	// always, as verified on PPSA01341 1.007 with NVIDIA (on the USA 1.004 build with an AMD card,
	// dependent dispatches overlapped: teal tint); 0 keeps every barrier.
	static const int chain_mode = [] {
		const char* value = std::getenv("KYTY_DS_COMPUTE_CHAIN");
		return value == nullptr ? 1 : std::atoi(value);
	}();
	const bool chain_enabled = chain_mode != 0;
	// The fenced programs (fe938d3a2d22b5c2 by default, KYTY_CHAIN_FENCE) never join a chain: their dispatches get the
	// barriers around them, as the other games' do.
	const bool        chain            = chain_enabled && DemonsSouls::IsSupportedGame() &&
	                                     DemonsSouls::ComputeChains(program.shader_hash);
	auto              vk_buffer        = chain ? buffer.ChainHandle() : buffer.Handle();
	PreparedBindings* descriptor_stage = &bindings;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u}, chain);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 9);
	thread_local std::vector<CommandBuffer::ChainAccess> chain_accesses;
	chain_accesses.clear();
	bool chain_unknown = program.info.uses_dma;
	if (chain && chain_mode == 1) {
		for (uint32_t i = 0; i < bindings.buffer_sources.size(); ++i) {
			const auto& source = bindings.buffer_sources[i];
			if (source.size == 0) continue;
			const bool written = i < program.info.buffers.size() && program.info.buffers[i].written;
			chain_accesses.push_back({source.address, source.address + source.size, written});
		}
		for (uint32_t i = 0; i < bindings.images.size() && i < program.info.images.size(); ++i) {
			const auto& data = bindings.images[i].desc.info.data;
			if (data.size == 0) continue;
			chain_accesses.push_back({data.address, data.address + data.size, program.info.images[i].written});
		}
		if (indirect_args != 0) chain_accesses.push_back({indirect_args, indirect_args + 12u, false});
		// KYTY_CHAIN_BDA_READS=1: a program that reaches memory through pointers only to read it (a complete BDA read
		// plan: BuildBdaReadPlan refuses any address write, atomic or unbounded load; planning-only loads emit no
		// code) chains on the ranges its plan evaluates to, the ranges PrepareBdaBindings made current, instead of
		// waiting for every earlier dispatch: a barrier stays when an earlier dispatch of the chain wrote them or
		// this one writes what an earlier one read through them.
		if (chain_unknown && kyty_local_chain_bda_reads_mode.load(std::memory_order_relaxed) != 0) {
			thread_local std::vector<GuestRange> dma_reads;
			dma_reads.clear();
			if (ShaderRecompiler::IR::EvaluateBdaReadPlan(program.bda_read_plan, input_info.stage.resources, dma_reads) &&
			    !dma_reads.empty()) {
				for (const auto& range: dma_reads)
					chain_accesses.push_back({range.address, range.address + range.size, false});
				chain_unknown = false;
			}
		}
	}
	bool continues_chain = chain && buffer.ComputeChainPending();
	bool drained_chain   = false; // (for a range conflict: the hazard barrier is redundant after it)
	static const bool chain_stats = std::getenv("KYTY_CHAIN_STATS") != nullptr;
	static uint64_t   stat_dispatches = 0, stat_elided = 0, stat_hazard = 0, stat_unknown = 0;
	if (continues_chain && chain_mode == 1 && buffer.ChainConflicts(chain_accesses, chain_unknown)) {
		(void)buffer.Handle(); // records the pending compute barrier
		continues_chain = false;
		drained_chain   = true;
		(chain_unknown ? stat_unknown : stat_hazard)++;
	} else if (continues_chain) {
		stat_elided++;
	}
	if (chain_stats && ++stat_dispatches % 20000 == 0) {
		std::printf("Compute chain: %llu dispatches, barrier left out %llu, kept for a range conflict %llu, "
		            "for pointer access %llu\n",
		            static_cast<unsigned long long>(stat_dispatches), static_cast<unsigned long long>(stat_elided),
		            static_cast<unsigned long long>(stat_hazard), static_cast<unsigned long long>(stat_unknown));
	}
	bool has_storage_writes = HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	const bool merge_barriers = kyty_local_dispatch_barrier_merge_mode.load(std::memory_order_relaxed) != 0;
	bool       args_ordered   = false; // the merged barrier below took the indirect argument read too
	if (has_storage_writes && !continues_chain && !(merge_barriers && drained_chain)) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		if (merge_barriers && indirect_args != 0) {
			auto barrier = MakeShaderWriteHazardDependency();
			barrier.dstAccessMask |= vk::AccessFlagBits::eIndirectCommandRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                          vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
			args_ordered = true;
		} else {
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (indirect_args != 0) {
		vk::BufferMemoryBarrier args_barrier {};
		args_barrier.sType         = vk::StructureType::eBufferMemoryBarrier;
		args_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                             vk::AccessFlagBits::eTransferWrite |
		                             vk::AccessFlagBits::eMemoryWrite;
		args_barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead;
		args_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.buffer              = indirect_buffer;
		args_barrier.offset              = indirect_offset;
		args_barrier.size                = 3u * sizeof(uint32_t);
		if (!continues_chain && !args_ordered) {
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                          vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 0, nullptr, 1, &args_barrier, 0,
			                          nullptr);
		}
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
	} else {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}

	// The removed host fence also ordered read-only dispatches before later writers.
	if (chain) {
		buffer.ContinueComputeChain();
		if (chain_mode == 1) buffer.RecordChainAccess(chain_accesses, chain_unknown);
	} else {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	ResetBindings();
	m_context.GetCommandScheduler().CompleteDispatch();
	phase.reset();
}

} // namespace Libs::Graphics
