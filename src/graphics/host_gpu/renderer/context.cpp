#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "live-counters.h"
#include "live-trace.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

#include "gpu_tiler_shaders/copy_batch_spv.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <mutex>

// windows.h (through common/threads.h) defines MemoryBarrier as a function macro: vk::MemoryBarrier is the struct.
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

extern "C" {
// KYTY_TRANSFER_CHAIN (default 0; live): independent buffer copies recorded back to back share one barrier ahead and one
// after (CommandBuffer::TransferHandle).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_transfer_chain_mode {0};
// KYTY_COPY_BATCH (default 0; live): a run of independent small guest linear copies made by one compute dispatch
// (CommandBuffer::BatchCopy).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_copy_batch_mode {0};
}

namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	FlushCopyBatch();
	DrainTransferChain();
	if (m_compute_access_pending) {
		ShaderAccessBarrier(m_buffer, vk::PipelineStageFlagBits::eComputeShader);
		m_compute_access_pending = false;
		m_chain_access.clear();
		m_chain_unknown = false;
	}
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::ChainHandle() const {
	EXIT_IF(IsInvalid() || m_rendering);
	FlushCopyBatch();
	DrainTransferChain();
	return m_buffer;
}

static vk::MemoryBarrier GlobalBarrier(vk::AccessFlags source, vk::AccessFlags destination) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = source;
	barrier.dstAccessMask = destination;
	return barrier;
}

// The chain's barrier after: the copies' transfers before every later command, their writes visible to it.
void CommandBuffer::DrainTransferChain() const {
	if (!m_transfer_pending) return;
	const auto after = GlobalBarrier(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	                                 vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
	                         vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
	m_transfer_pending = false;
	m_transfer_access.clear();
}

vk::CommandBuffer CommandBuffer::TransferHandle(std::span<const TransferAccess> accesses) const {
	EXIT_IF(IsInvalid() || m_rendering);
	FlushCopyBatch();
	if (m_compute_access_pending) (void)Handle(); // the pending compute dependency first (no transfer chain then)
	if (!m_transfer_pending) {
		// The barrier ahead: everything recorded before, at any stage, before the transfers; its writes visible to them.
		const auto before = GlobalBarrier(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                                  vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
		m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
		                         vk::DependencyFlagBits::eByRegion, 1, &before, 0, nullptr, 0, nullptr);
	} else {
		bool conflict = m_transfer_access.size() + accesses.size() > 256;
		for (const auto& pending: m_transfer_access) {
			for (const auto& access: accesses) {
				conflict = conflict || (pending.buffer == access.buffer && (pending.written || access.written) &&
				                        access.begin < pending.end && pending.begin < access.end);
			}
		}
		if (conflict) {
			// A copy touching what an earlier one of the chain writes (or writing what one reads) waits for them.
			const auto between = GlobalBarrier(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
			                                   vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
			m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
			                         vk::DependencyFlagBits::eByRegion, 1, &between, 0, nullptr, 0, nullptr);
			m_transfer_access.clear();
		}
	}
	m_transfer_access.insert(m_transfer_access.end(), accesses.begin(), accesses.end());
	m_transfer_pending = true;
	return m_buffer;
}

namespace {
// copy_batch.comp's pipeline: no descriptors, the table's device address and the copy count as push constants.
struct CopyBatchPipeline {
	vk::PipelineLayout layout   = nullptr;
	vk::Pipeline       pipeline = nullptr;
};

const CopyBatchPipeline& GetCopyBatchPipeline(vk::Device device) {
	static CopyBatchPipeline created;
	static std::once_flag    once;
	std::call_once(once, [&] {
		vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 16};
		vk::PipelineLayoutCreateInfo layout {};
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &created.layout),
		                     "create copy batch pipeline layout");
		vk::ShaderModuleCreateInfo shader {};
		shader.codeSize = sizeof(COPY_BATCH_SPV);
		shader.pCode    = COPY_BATCH_SPV;
		vk::ShaderModule module;
		RequireVulkanSuccess(device.createShaderModule(&shader, nullptr, &module), "create copy batch shader");
		vk::ComputePipelineCreateInfo pipeline {};
		pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline.stage.module = module;
		pipeline.stage.pName  = "main";
		pipeline.layout       = created.layout;
		RequireVulkanSuccess(device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &created.pipeline),
		                     "create copy batch pipeline");
		device.destroyShaderModule(module, nullptr);
	});
	return created;
}

constexpr uint64_t CopyBatchMaxBytes  = 64 * 1024; // a workgroup's loop: 256 dwords a thread at most
constexpr size_t   CopyBatchMaxCopies = 1024;
} // namespace

bool CommandBuffer::BatchCopy(const Buffer& source, uint64_t source_offset, const Buffer& destination,
                              uint64_t destination_offset, uint64_t size) const {
	if (kyty_local_copy_batch_mode.load(std::memory_order_relaxed) == 0 || IsInvalid() || size == 0 ||
	    size > CopyBatchMaxBytes || ((source_offset | destination_offset | size) & 3u) != 0) {
		return false;
	}
	if (!source.HasDeviceAddress() || !destination.HasDeviceAddress()) return false;
	const auto source_address      = source.BufferDeviceAddress();
	const auto destination_address = destination.BufferDeviceAddress();
	EndRendering(); // (a rendering's Handle() left no batch)
	const TransferAccess accesses[] = {
	    {static_cast<VkBuffer>(source.Handle()), source_offset, source_offset + size, false},
	    {static_cast<VkBuffer>(destination.Handle()), destination_offset, destination_offset + size, true},
	};
	if (m_copy_batch.empty()) {
		// Whatever else is pending (a compute chain's barrier, a transfer chain's) first, then the batch alone.
		(void)Handle();
	} else {
		bool conflict = m_copy_batch.size() >= CopyBatchMaxCopies;
		for (const auto& pending: m_copy_batch_access) {
			for (const auto& access: accesses) {
				conflict = conflict || (pending.buffer == access.buffer && (pending.written || access.written) &&
				                        access.begin < pending.end && pending.begin < access.end);
			}
		}
		// A copy touching what an earlier one writes (or writing what one reads) runs after them.
		if (conflict) FlushCopyBatch();
	}
	m_copy_batch.push_back({source_address + source_offset, destination_address + destination_offset,
	                        static_cast<uint32_t>(size / 4), 0});
	m_copy_batch_buffers.push_back({accesses[0].buffer, accesses[1].buffer, source_offset, destination_offset, size});
	m_copy_batch_access.insert(m_copy_batch_access.end(), std::begin(accesses), std::end(accesses));
	LiveCounters::Add(LiveCounters::BatchedCopies);
	return true;
}

// The batch: everything before it, at any stage, before its copies, and its copies before everything after.
void CommandBuffer::FlushCopyBatch() const {
	if (m_copy_batch.empty()) return;
	const auto count = static_cast<uint32_t>(m_copy_batch.size());
	const auto bytes = count * sizeof(BatchedCopy);
	auto&      table = m_context.GetBufferCache().GetTableUploadBuffer();
	// No wait for room: a wait could submit this very buffer. Without room the copies are transfers.
	const auto [data, offset] = table.HasDeviceAddress() ? table.Map(bytes, 16, false)
	                                                      : std::pair<uint8_t*, uint64_t> {nullptr, 0};
	if (data != nullptr) {
		std::memcpy(data, m_copy_batch.data(), bytes);
		table.Commit();
		const auto& pipeline = GetCopyBatchPipeline(m_graphics.device);
		const auto  before   = GlobalBarrier(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                                     vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
		m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
		                         {}, 1, &before, 0, nullptr, 0, nullptr);
		m_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		struct {
			uint64_t table;
			uint32_t count;
			uint32_t unused;
		} constants {table.BufferDeviceAddress() + offset, count, 0};
		m_buffer.pushConstants(pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), &constants);
		m_buffer.dispatch(count, 1, 1);
		const auto after = GlobalBarrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
		                                 vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
		m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands,
		                         {}, 1, &after, 0, nullptr, 0, nullptr);
		LiveCounters::Add(LiveCounters::CopyBatches);
	} else {
		const auto before = GlobalBarrier(vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                                  vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
		m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
		                         &before, 0, nullptr, 0, nullptr);
		for (const auto& copy: m_copy_batch_buffers) {
			const vk::BufferCopy region {copy.source_offset, copy.destination_offset, copy.size};
			m_buffer.copyBuffer(copy.source, copy.destination, 1, &region);
		}
		const auto after = GlobalBarrier(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
		                                 vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
		m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 1,
		                         &after, 0, nullptr, 0, nullptr);
	}
	m_copy_batch.clear();
	m_copy_batch_buffers.clear();
	m_copy_batch_access.clear();
}

void CommandBuffer::ContinueComputeChain() const {
	EXIT_IF(IsInvalid() || m_rendering);
	m_compute_access_pending = true;
}

bool CommandBuffer::ChainConflicts(std::span<const ChainAccess> accesses, bool unknown) const {
	if (!m_compute_access_pending) return false;
	if (unknown || m_chain_unknown) return true;
	for (const auto& pending: m_chain_access) {
		for (const auto& access: accesses) {
			if ((pending.written || access.written) && access.begin < pending.end && pending.begin < access.end) {
				return true;
			}
		}
	}
	return false;
}

void CommandBuffer::RecordChainAccess(std::span<const ChainAccess> accesses, bool unknown) const {
	m_chain_unknown = m_chain_unknown || unknown;
	// A long chain is ended rather than scanned: past 256 ranges the next dispatch waits.
	if (m_chain_access.size() + accesses.size() > 256) {
		m_chain_unknown = true;
		return;
	}
	m_chain_access.insert(m_chain_access.end(), accesses.begin(), accesses.end());
}

vk::CommandBuffer CommandBuffer::HandleForFullBarrier() const {
	EXIT_IF(IsInvalid());
	FlushCopyBatch(); // its copies before the caller's barrier
	// The caller records a full AllCommands memory dependency immediately.
	m_transfer_pending = false;
	m_transfer_access.clear();
	m_compute_access_pending = false;
	m_chain_access.clear();
	m_chain_unknown = false;
	return m_buffer;
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
static void ReplayBegin(std::span<const LocalVulkanRecording::Segment> segments,
                        const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto command = *static_cast<const VkCommandBuffer*>(segments[0].data);
	VkCommandBufferBeginInfo info {};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (dispatch.vkBeginCommandBuffer(command, &info) != VK_SUCCESS) {
		EXIT("deferred vkBeginCommandBuffer failed\n");
	}
	const auto slot = *static_cast<const uint32_t*>(segments[1].data);
	if (slot != UINT32_MAX && LiveTrace::g_timestamp_pool != nullptr) {
		const auto pool = static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool);
		dispatch.vkCmdResetQueryPool(command, pool, slot * 2u, 2u);
		dispatch.vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, slot * 2u);
	}
}
#endif

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	InvalidateGraphicsState();
	// Not Handle(): nothing may be recorded before the buffer begins.
	const vk::CommandBuffer buffer = m_buffer;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// The recording worker owns this pool's buffers while it replays them; begin there too.
	{
		const VkCommandBuffer raw = buffer;
		m_timestamp_slot          = LiveTrace::g_on.load(std::memory_order_relaxed) && LiveTrace::g_timestamp_pool
		                                ? LiveTrace::g_timestamp_next.fetch_add(1) % LiveTrace::TimestampSlots
		                                : UINT32_MAX;
		const LocalVulkanRecording::Segment segments[] {{&raw, sizeof(raw)}, {&m_timestamp_slot, sizeof(m_timestamp_slot)}};
		if (LocalVulkanRecording::EnqueueDeferred(ReplayBegin, segments, false)) {
			return;
		}
	}
#endif

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::InvalidateGraphicsState() const noexcept {
	++m_graphics_generation;
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

// The same rendering: the size, the layers, the depth/stencil attachment and the color attachments up to the count
// (those past it are no part of a rendering). (All eight compared, at every table draw: ~1% of the render thread at
// Latria standing.)
static bool SameRendering(const RenderState& a, const RenderState& b) {
	if (a.num_color_attachments != b.num_color_attachments || a.width != b.width || a.height != b.height ||
	    a.num_layers != b.num_layers || !(a.depth_stencil_attachment == b.depth_stencil_attachment))
		return false;
	for (uint32_t i = 0; i < a.num_color_attachments; i++)
		if (!(a.color_attachments[i] == b.color_attachments[i])) return false;
	return true;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering && SameRendering(m_render_state, state)) {
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	LiveCounters::Add(LiveCounters::RenderPasses);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
