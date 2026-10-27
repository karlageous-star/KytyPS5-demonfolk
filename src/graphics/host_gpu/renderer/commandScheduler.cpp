#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionManager.h"
#include "anti-lag.h"
#include "async-upload.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#include "device-fault.h"
#endif
#include "live-counters.h"
#include "live-trace-gpu.h"
#include "local-platform.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <optional>

extern "C" {
// Dispatches recorded per submission (KYTY_DISPATCH_BATCH).
volatile std::atomic_uint32_t kyty_local_dispatch_batch {32};
// KYTY_BATCH_WHILE_BUSY=1 (chenxiao 0c29dbf9): a dispatch batch goes on while the GPU still runs work handed to it.
volatile std::atomic_uint32_t kyty_local_batch_while_busy_mode {0};
// KYTY_FLUSH_ON_DRAIN (live; default 2): 1 submits the recorded work at a dispatch entry, or a draw entry outside a
// render pass, once the GPU has finished every earlier submission; 2 also ends an open render pass for it; 0 never.
// N7 (new game, tutorial tunnel, same session, 5 interleaved 12 s rounds): 0 -> 2 gave 21.98 -> 23.43 fps (+0.9 to
// +1.9 in every round), p99 53.7 -> 49.7 ms, 16.8 drain flushes a frame, the same picture. Mode 1 barely fires
// (0.1 a frame): the GPU runs dry while the render thread records draws inside one render pass.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_flush_on_drain_mode {2};
// KYTY_FLUSH_DEFER (default 2; 0: never): a flush point nothing waits on (a label without an interrupt, the end of a
// queue's slice) leaves the open buffer open while the GPU still has at least this many submissions to run
// (DeferrableFlush). Demon's Souls new game, 1080p, live A/B 0 -> 2: +0.86 FPS in each of 4 rounds (24.46 -> 25.32, E8,
// medium preset) and +0.36 (E5, logging on).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_flush_defer_mode {2};
// Draws recorded per submission (KYTY_DRAW_BATCH, 0: no limit).
volatile std::atomic_uint32_t kyty_local_draw_batch {256};
// Buffer uploads of pages dirty since before the open command buffer began go into its upload prologue.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_upload_prologue {1};
// 0: refresh the master timeline before every drain attempt.
// 1: prove the pending-operation queue empty before the timeline query and the
//    operation lock. Draws and dispatches call this with nothing to retire.
volatile std::atomic<uint32_t> kyty_local_pending_drain_mode {0};
#ifdef KYTY_LOCAL_VULKAN_RECORDING
extern volatile std::atomic<uint32_t> kyty_local_vulkan_recording_mode;
#endif
}

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

bool OrdinaryAsyncSubmitRequested() {
	const auto* value = std::getenv("KYTY_ASYNC_SUBMIT");
	if (value == nullptr || value[0] != '1' || value[1] != '\0') return false;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// The recording stream owns its submit ordering. Do not nest two workers.
	if (kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0) return false;
#endif
	return true;
}

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	if (result == vk::Result::eErrorDeviceLost) DeviceFault::Report();
#endif
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this), m_prologue(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }),
      m_tick_monitor([this](std::stop_token stop) {
	      LocalPlatform::SetThreadName("Kyty.TickMon");
	      {
		      vk::QueryPoolCreateInfo info {};
		      info.queryType  = vk::QueryType::eTimestamp;
		      info.queryCount = LiveTrace::TimestampSlots * 2u;
		      vk::QueryPool pool {};
		      if (m_graphics.device.createQueryPool(&info, nullptr, &pool) == vk::Result::eSuccess) {
			      LiveTrace::g_timestamp_period = m_graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
			      LiveTrace::g_timestamp_pool   = static_cast<VkQueryPool>(pool);
		      }
		      // Per-draw marks (`tracem`): reset on the host, read after tracing stops.
		      info.queryCount = LiveTrace::MarkSlots;
		      vk::QueryPool marks {};
		      if (m_graphics.device.createQueryPool(&info, nullptr, &marks) == vk::Result::eSuccess) {
			      m_graphics.device.resetQueryPool(marks, 0, LiveTrace::MarkSlots);
			      LiveTrace::g_mark_pool = static_cast<VkQueryPool>(marks);
			      static vk::Device device = m_graphics.device;
			      LiveTrace::g_read_marks = [](uint32_t count) {
				      const auto            pool = static_cast<VkQueryPool>(LiveTrace::g_mark_pool);
				      std::vector<uint64_t> values(count);
				      if (count != 0 &&
				          device.getQueryPoolResults(pool, 0, count, count * sizeof(uint64_t), values.data(),
				                                     sizeof(uint64_t),
				                                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) ==
				              vk::Result::eSuccess) {
					      for (uint32_t i = 0; i < count; ++i)
						      LiveTrace::Append(LiveTrace::GpuMarkValue, i,
						                        static_cast<uint64_t>(static_cast<double>(values[i]) *
						                                              LiveTrace::g_timestamp_period));
				      }
				      device.resetQueryPool(pool, 0, LiveTrace::MarkSlots);
			      };
		      }
	      }
	      uint64_t next = 0;
	      while (!stop.stop_requested()) {
		      if (!LiveTrace::g_on.load(std::memory_order_relaxed)) {
			      next = 0;
			      std::this_thread::sleep_for(std::chrono::milliseconds(5));
			      continue;
		      }
		      if (next == 0) next = m_master.KnownGpuTick() + 1;
		      vk::SemaphoreWaitInfo wait {};
		      const auto            semaphore = m_master.Handle();
		      wait.semaphoreCount           = 1;
		      wait.pSemaphores              = &semaphore;
		      wait.pValues                  = &next;
		      if (m_graphics.device.waitSemaphores(&wait, 2000000) == vk::Result::eSuccess) {
			      LiveTrace::Event(LiveTrace::TickDone, next);
			      auto& mapped = LiveTrace::g_tick_slot[next % LiveTrace::TimestampSlots];
			      if (const auto slot = mapped.exchange(0); slot != 0 && LiveTrace::g_timestamp_pool != nullptr) {
				      uint64_t times[2] {};
				      if (m_graphics.device.getQueryPoolResults(
				              static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool), (slot - 1u) * 2u, 2u,
				              sizeof(times), times, sizeof(uint64_t), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
					      LiveTrace::Event(LiveTrace::GpuSpan, next);
					      LiveTrace::Event(LiveTrace::GpuSpanNs,
					                       static_cast<uint64_t>(static_cast<double>(times[0]) * LiveTrace::g_timestamp_period),
					                       static_cast<uint64_t>(static_cast<double>(times[1]) * LiveTrace::g_timestamp_period));
				      }
			      }
			      ++next;
		      }
	      }
      }) {}

void CommandScheduler::CompleteDispatch() {
	// Expose command-processor/GPU overlap without a host fence. Bound the
	// recording batch; resource retirement still uses the submission timeline.
	// The boundary costs one all-commands dependency, which lets no part of the
	// next batch begin before this one drains, so its size trades that drain
	// against how early the queue receives work. Dispatches inside a batch keep
	// the same chain barriers at any size.
	// (A deferred flush goes here once the GPU is down to its last submission: DeferrableFlush.)
	if (FlushedDeferred(4)) return;
	const auto batch = std::clamp(kyty_local_dispatch_batch.load(std::memory_order_relaxed), 1u, 65536u);
	if (++m_recorded_dispatches < batch) {
		return;
	}
	// While the GPU still runs work handed to it (m_driver_tick: the last tick the driver took) the batch goes on,
	// asked at the batch's end and every 4 dispatches after it, up to 8 batches.
	if (kyty_local_batch_while_busy_mode.load(std::memory_order_relaxed) != 0 && m_recorded_dispatches < 8 * batch &&
	    ((m_recorded_dispatches != batch && (m_recorded_dispatches & 3u) != 0) ||
	     GpuBusy())) {
		return;
	}
	CheckActive();
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

// Draw batches expose translated work to the GPU while the guest command buffer is still being recorded.
void CommandScheduler::CompleteDraws(uint32_t draws) {
	if (FlushedDeferred(draws)) return;
	const auto batch = kyty_local_draw_batch.load(std::memory_order_relaxed);
	if (batch == 0 || (m_recorded_draws += draws) < batch) return;
	const bool end = m_recorded_draws - draws < batch;
	if (m_recorded_draws < 8 * batch && ((!end && (m_recorded_draws & 15u) >= draws && draws < 16) || GpuBusy())) {
		return;
	}
	CheckActive();
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

bool CommandScheduler::GpuBusy() {
	const auto handed = m_driver_tick.load(std::memory_order_acquire);
	if (m_master.IsFree(handed)) return false;
	m_master.Refresh();
	return !m_master.IsFree(handed);
}

// N3 (new game, Boletaria): 12.7 of 16.4 ms a frame of GPU idle fell inside stretches where the render thread
// recorded one guest submission for up to 7.9 ms without a flush point: the GPU waited for work already recorded.
// At most one timeline query each 0.2 ms; the boundary is CompleteDispatch's (an all-commands dependency).
void CommandScheduler::FlushIfDrained() {
	// A deferred flush (DeferrableFlush) is asked here whatever KYTY_FLUSH_ON_DRAIN says: every 16 entries.
	if (m_flush_deferred && FlushedDeferred(1)) return;
	const auto mode = kyty_local_flush_on_drain_mode.load(std::memory_order_relaxed);
	if (mode == 0 || !Active() || m_command.IsInvalid() || m_drain_entries++ == 0) {
		return;
	}
	const auto now = Common::Timer::QueryPerformanceCounter();
	if (now - m_last_drain_check < Common::Timer::QueryPerformanceFrequency() / 5000) {
		return;
	}
	m_last_drain_check = now;
	const auto tick    = CurrentTick();
	if (tick <= 1 || (m_command.IsRendering() && mode < 2) || !IsFree(tick - 1)) {
		return;
	}
	LiveCounters::Add(LiveCounters::DrainFlushes);
	m_command.EndRendering();
	VulkanMemoryBarrier dependency {};
	dependency.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	dependency.dstAccessMask = dependency.srcAccessMask;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                   vk::PipelineStageFlagBits::eAllCommands, {}, 1, &dependency,
	                                   0, nullptr, 0, nullptr);
	Flush();
}

// A flush point nothing waits on (a label without an interrupt, the end of a queue's slice: CommandProcessor::
// BufferFlushDeferrable) while the GPU has the configured backlog of submissions still to run: the buffer
// stays open, and goes with the next flush, or once the GPU is down to its last submission (FlushedDeferred: asked by
// FlushIfDrained and CompleteDispatch, so it never runs dry waiting for this one), or before the GPU thread waits for
// work (FlushDeferred). Each submission costs the GPU a drain between back-to-back command buffers and the recording
// thread a submission. Deferring whenever the GPU merely still ran something (one submission) halved the submissions
// too but starved the GPU after short ones (chenxiao 31119bdf).
// (Ended as a submission ends it: the render pass and a compute chain's pending barrier, so the commands after it are
// recorded as they would be into the next buffer.)
void CommandScheduler::DeferrableFlush() {
	const auto backlog = kyty_local_flush_defer_mode.load(std::memory_order_relaxed);
	if (backlog != 0 && !m_command.IsInvalid() && BacklogAtLeast(backlog)) {
		m_command.EndRendering();
		(void)m_command.Handle();
		m_flush_deferred = true;
		return;
	}
	Flush();
}

void CommandScheduler::FlushDeferred() {
	if (m_flush_deferred) Flush();
}

// While a flush is deferred, every 16 units of `weight` (draws or dispatch entries) whether the GPU is down to its last
// submission: then the open buffer is submitted (true).
bool CommandScheduler::FlushedDeferred(uint32_t weight) {
	if (!m_flush_deferred) return false;
	if ((m_deferred_checks += weight) < 16) return false;
	m_deferred_checks = 0;
	if (BacklogAtLeast(2)) return false;
	Flush();
	return true;
}

// Avoid a timeline query unless the cached values could meet the requested backlog. The cached GPU tick never leads
// actual completion, so this early-out is conservative.
bool CommandScheduler::BacklogAtLeast(uint64_t count) {
	const auto backlog = [this] {
		const auto submitted = m_master.CurrentTick() - 1;
		const auto done      = m_master.KnownGpuTick();
		return submitted > done ? submitted - done : 0;
	};
	if (backlog() < count) return false;
	m_master.Refresh();
	return backlog() >= count;
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	// The final timeline wait above proves all queued command buffers completed.
	StopSubmitThread();
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations(bool at_draw_entry) {
	const bool use_hint = kyty_local_pending_drain_mode.load(std::memory_order_relaxed) != 0;
	if (use_hint && m_pending_operation_count.load(std::memory_order_acquire) == 0) return;
	// Merge the timeline-query throttle with the empty-queue and driver-front hints. Explicit drains
	// still refresh every time unless the caller opted into the draw-entry throttle.
	static const bool throttle = [] {
		const auto* value = std::getenv("KYTY_PENDING_REFRESH_THROTTLE");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	const auto now = Common::Timer::QueryPerformanceCounter();
	const bool refresh_due = !throttle || !at_draw_entry || m_last_pending_refresh == 0 ||
	                         now - m_last_pending_refresh >= Common::Timer::QueryPerformanceFrequency() / 5000;
	if (use_hint) {
		// Entries whose submit is still open or queued cannot have completed. Avoid querying the timeline
		// until the FIFO submit worker has handed the oldest entry to the driver.
		const auto front = m_pending_front_tick.load(std::memory_order_acquire);
		if (front > m_driver_tick.load(std::memory_order_acquire)) return;
		if (!m_master.IsFree(front) && refresh_due) {
			m_master.Refresh();
			m_last_pending_refresh = now;
		}
	} else if (refresh_due) {
		m_master.Refresh();
		m_last_pending_refresh = now;
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
			m_pending_front_tick.store(m_pending_operations.empty() ? UINT64_MAX : m_pending_operations.front().tick,
			                           std::memory_order_relaxed);
			m_pending_operation_count.fetch_sub(1, std::memory_order_relaxed);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		const auto tick = CurrentTick();
		m_pending_operations.push({std::move(operation), tick});
		if (m_pending_operations.size() == 1) m_pending_front_tick.store(tick, std::memory_order_relaxed);
		m_pending_operation_count.fetch_add(1, std::memory_order_release);
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		LiveTrace::Event(LiveTrace::GpuDone, operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

bool CommandScheduler::Active() const noexcept {
	return m_command.m_registers != nullptr;
}

uint64_t CommandScheduler::CurrentTick() const noexcept {
	return m_master.CurrentTick();
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid() || !m_prologue.IsInvalid());
	++m_command_serial;
	m_command_clock    = g_cpu_dirty_clock.load(std::memory_order_acquire);
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	LiveTrace::g_mark_command = static_cast<VkCommandBuffer>(m_command.m_buffer);
	return m_command;
}

vk::CommandBuffer CommandScheduler::UploadPrologue(uint64_t last_dirty, uint64_t written_serial) {
	if (kyty_local_upload_prologue.load(std::memory_order_relaxed) == 0 || m_command.IsInvalid() ||
	    last_dirty > m_command_clock || written_serial == m_command_serial) {
		return nullptr;
	}
	if (m_prologue.IsInvalid()) {
		m_prologue.m_buffer = m_command_pool.Commit();
		m_prologue.Begin();
		VulkanMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		m_prologue.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                                    vk::PipelineStageFlagBits::eTransfer, {}, 1, &before,
		                                    0, nullptr, 0, nullptr);
	}
	return m_prologue.Handle();
}

CommandScheduler::SubmitEntry CommandScheduler::ClosePrologue() {
	if (m_prologue_hook) m_prologue_hook(m_prologue.Handle());
	VulkanMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	m_prologue.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                    vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after,
	                                    0, nullptr, 0, nullptr);
	const SubmitEntry entry {m_prologue.m_buffer, UINT32_MAX};
	m_prologue.m_buffer = nullptr;
	return entry;
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
namespace {
// vkEndCommandBuffer + vkQueueSubmit executed by the recording worker, in order with
// the commands it has replayed. The tick is assigned on the producer as before.
struct DeferredSubmit {
	VkCommandBuffer      commands[CommandScheduler::MaxSubmitEntries];
	uint32_t             count;
	VkQueue              queue;
	Common::Mutex*       queue_mutex;
	uint32_t             waits, signals;
	VkSemaphore          wait_semaphores[SubmitInfo::MaxSemaphores];
	uint64_t             wait_ticks[SubmitInfo::MaxSemaphores];
	VkPipelineStageFlags wait_stages[SubmitInfo::MaxSemaphores];
	VkSemaphore          signal_semaphores[SubmitInfo::MaxSemaphores];
	uint64_t             signal_ticks[SubmitInfo::MaxSemaphores];
	uint64_t             tick;
	uint64_t             upload_sequence; // KYTY_ASYNC_UPLOAD copies these buffers read
	uint32_t             timestamp_slot;  // live trace (the last buffer the stream ends)
	std::atomic<uint64_t>* driver_tick;   // CommandScheduler::m_driver_tick
};
void ReplaySubmit(std::span<const LocalVulkanRecording::Segment> segments,
                  const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto& submit = *static_cast<const DeferredSubmit*>(segments[0].data);
	AsyncUpload::Wait(submit.upload_sequence);
	LiveTrace::Event(LiveTrace::GpuSubmit, submit.tick, 1);
	for (uint32_t i = 0; i < submit.count; ++i) {
		if (submit.timestamp_slot != UINT32_MAX && i + 1 == submit.count && LiveTrace::g_timestamp_pool != nullptr) {
			dispatch.vkCmdWriteTimestamp(submit.commands[i], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			                             static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool),
			                             submit.timestamp_slot * 2u + 1u);
			LiveTrace::g_tick_slot[submit.tick % LiveTrace::TimestampSlots].store(submit.timestamp_slot + 1u,
			                                                                        std::memory_order_release);
		}
		if (dispatch.vkEndCommandBuffer(submit.commands[i]) != VK_SUCCESS) {
			EXIT("deferred vkEndCommandBuffer failed, tick=%" PRIu64 "\n", submit.tick);
		}
	}
	VkTimelineSemaphoreSubmitInfo timeline {};
	timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	timeline.waitSemaphoreValueCount   = submit.waits;
	timeline.pWaitSemaphoreValues      = submit.wait_ticks;
	timeline.signalSemaphoreValueCount = submit.signals;
	timeline.pSignalSemaphoreValues    = submit.signal_ticks;
	VkSubmitInfo info {};
	info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	info.pNext                = &timeline;
	info.waitSemaphoreCount   = submit.waits;
	info.pWaitSemaphores      = submit.wait_semaphores;
	info.pWaitDstStageMask    = submit.wait_stages;
	info.commandBufferCount   = submit.count;
	info.pCommandBuffers      = submit.commands;
	info.signalSemaphoreCount = submit.signals;
	info.pSignalSemaphores    = submit.signal_semaphores;
	for (uint32_t i = 0; i < submit.signals; ++i) {
		if (submit.signal_ticks[i] != submit.tick) continue;
		DeviceFault::NoteSubmit(static_cast<uint64_t>(submit.signal_semaphores[i]), submit.tick, // NOLINT
		                        submit.commands, submit.count, submit.wait_semaphores, submit.wait_ticks,
		                        submit.waits);
		break;
	}
	VkResult result;
	{
		Common::LockGuard lock(*submit.queue_mutex);
		result = Libs::Graphics::AntiLag::QueueSubmit(submit.queue, info, dispatch.vkQueueSubmit,
		                                               dispatch.vkQueueSubmit2);
	}
	if (result != VK_SUCCESS) {
		if (result == VK_ERROR_DEVICE_LOST) DeviceFault::Report();
		EXIT("deferred vkQueueSubmit failed: %d, tick=%" PRIu64 "\n", static_cast<int>(result), submit.tick);
	}
	submit.driver_tick->store(submit.tick, std::memory_order_release);
	LocalVulkanRecording::NoteDeferredSubmitDone();
}
} // namespace
#endif

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	m_command.EndRendering();
	// Handle() records any pending compute-chain barrier before the buffer is closed.
	(void)m_command.Handle();
	std::array<SubmitEntry, 2> entries {};
	size_t n = 0;
	if (!m_prologue.IsInvalid()) entries[n++] = ClosePrologue();
	entries[n++] = {m_command.m_buffer, m_command.m_timestamp_slot};
	const auto tick = SubmitBuffers(std::span<const SubmitEntry>(entries.data(), n), std::move(submit));
	m_command.m_buffer = nullptr;
	m_recorded_dispatches = 0;
	m_recorded_draws = 0;
	m_flush_deferred      = false;
	m_deferred_checks = 0;
	return tick;
}

uint64_t CommandScheduler::SubmitBuffers(std::span<const SubmitEntry> entries, SubmitInfo submit) {
	PageManager::FlushDeferredProtection();
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);
	EXIT_IF(m_graphics.queue == nullptr || entries.empty() || entries.size() > MaxSubmitEntries);

#ifdef KYTY_LOCAL_VULKAN_RECORDING
	EXIT_IF(m_async_submit && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0);
	if (LocalVulkanRecording::DeferredSubmitEnabled()) {
		DeferredSubmit deferred {};
		deferred.tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), deferred.tick);
		deferred.count = static_cast<uint32_t>(entries.size());
		deferred.timestamp_slot = UINT32_MAX;
		deferred.driver_tick = &m_driver_tick;
		for (size_t i = 0; i < entries.size(); ++i) {
			deferred.commands[i] = entries[i].buffer;
			deferred.timestamp_slot = entries[i].timestamp_slot;
		}
		deferred.upload_sequence = AsyncUpload::SubmitSequence();
		deferred.queue = m_graphics.queue;
		deferred.queue_mutex = &m_graphics.queue_mutex;
		deferred.waits = submit.num_wait_semaphores;
		deferred.signals = submit.num_signal_semaphores;
		for (uint32_t i = 0; i < deferred.waits; ++i) {
			deferred.wait_semaphores[i] = submit.wait_semaphores[i];
			deferred.wait_ticks[i] = submit.wait_ticks[i];
			deferred.wait_stages[i] = static_cast<VkPipelineStageFlags>(submit.wait_stages[i]);
		}
		for (uint32_t i = 0; i < deferred.signals; ++i) {
			deferred.signal_semaphores[i] = submit.signal_semaphores[i];
			deferred.signal_ticks[i] = submit.signal_ticks[i];
		}
		const LocalVulkanRecording::Segment segments[] {{&deferred, sizeof(deferred)}};
		LocalVulkanRecording::NoteDeferredSubmitQueued();
		if (!LocalVulkanRecording::EnqueueDeferred(ReplaySubmit, segments, true)) {
			LocalVulkanRecording::Drain();
			ReplaySubmit(segments, VULKAN_HPP_DEFAULT_DISPATCHER);
		}
		LiveTrace::Event(LiveTrace::GpuSubmit, deferred.tick);
		return deferred.tick;
	}
#endif

	std::array<vk::CommandBuffer, MaxSubmitEntries> buffers {};
	for (size_t i = 0; i < entries.size(); ++i) {
		buffers[i] = entries[i].buffer;
		EXIT_NOT_IMPLEMENTED(buffers[i].end() != vk::Result::eSuccess);
	}

	if (m_async_submit || OrdinaryAsyncSubmitRequested()) {
		if (!m_async_submit) {
			m_async_submit = true;
			m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
			LOGF("Scheduler: ordinary FIFO queue-submit worker enabled\n");
		}
		SubmitJob job {};
		job.buffers = buffers;
		job.buffer_count = static_cast<uint32_t>(entries.size());
		job.submit = std::move(submit);
		job.upload_sequence = AsyncUpload::SubmitSequence();
		job.debug_op = m_command.m_debug_op;
		job.debug_submit = m_command.m_debug_submit_id;
		job.debug_arg0 = m_command.m_debug_arg0;
		job.debug_arg1 = m_command.m_debug_arg1;
		job.debug_arg2 = m_command.m_debug_arg2;
		job.debug_arg3 = m_command.m_debug_arg3;
		job.debug_arg4 = m_command.m_debug_arg4;
		uint64_t tick;
		{
			std::lock_guard lock(m_submit_mutex);
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
			tick = job.tick;
			m_submit_jobs.push_back(std::move(job));
			++m_submits_queued;
		}
		m_submit_available.notify_one();
		return tick;
	}

	AsyncUpload::Drain();
	vk::Result result;
	uint64_t tick;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);
		vk::TimelineSemaphoreSubmitInfo timeline {};
		timeline.waitSemaphoreValueCount = submit.num_wait_semaphores;
		timeline.pWaitSemaphoreValues = submit.wait_ticks.data();
		timeline.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline.pSignalSemaphoreValues = submit.signal_ticks.data();
		vk::SubmitInfo info {};
		info.pNext = &timeline;
		info.waitSemaphoreCount = submit.num_wait_semaphores;
		info.pWaitSemaphores = submit.wait_semaphores.data();
		info.pWaitDstStageMask = submit.wait_stages.data();
		info.commandBufferCount = static_cast<uint32_t>(entries.size());
		info.pCommandBuffers = buffers.data();
		info.signalSemaphoreCount = submit.num_signal_semaphores;
		info.pSignalSemaphores = submit.signal_semaphores.data();
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		static_assert(sizeof(vk::CommandBuffer) == sizeof(VkCommandBuffer));
		DeviceFault::NoteSubmit(static_cast<uint64_t>(static_cast<VkSemaphore>(m_master.Handle())), tick, // NOLINT
		                        reinterpret_cast<const VkCommandBuffer*>(buffers.data()),
		                        static_cast<uint32_t>(entries.size()),
		                        reinterpret_cast<const VkSemaphore*>(submit.wait_semaphores.data()),
		                        submit.wait_ticks.data(), submit.num_wait_semaphores);
#endif
		result = static_cast<vk::Result>(Libs::Graphics::AntiLag::QueueSubmit(
		    static_cast<VkQueue>(m_graphics.queue), *reinterpret_cast<const VkSubmitInfo*>(&info),
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit, VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2));
		m_driver_tick.store(tick, std::memory_order_release);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op, m_command.m_debug_submit_id,
		                  m_command.m_debug_arg0, m_command.m_debug_arg1, m_command.m_debug_arg2,
		                  m_command.m_debug_arg3, m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	return tick;
}

void CommandScheduler::QueueSubmit(const SubmitJob& job) {
	// The producer snapshots the sequence before it advances to later upload jobs.
	AsyncUpload::Wait(job.upload_sequence);
	const auto& submit = job.submit;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = submit.num_wait_semaphores;
	timeline.pWaitSemaphoreValues = submit.wait_ticks.data();
	timeline.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline.pSignalSemaphoreValues = submit.signal_ticks.data();
	vk::SubmitInfo info {};
	info.pNext = &timeline;
	info.waitSemaphoreCount = submit.num_wait_semaphores;
	info.pWaitSemaphores = submit.wait_semaphores.data();
	info.pWaitDstStageMask = submit.wait_stages.data();
	info.commandBufferCount = job.buffer_count;
	info.pCommandBuffers = job.buffers.data();
	info.signalSemaphoreCount = submit.num_signal_semaphores;
	info.pSignalSemaphores = submit.signal_semaphores.data();
	vk::Result result;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		static_assert(sizeof(vk::CommandBuffer) == sizeof(VkCommandBuffer));
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		DeviceFault::NoteSubmit(static_cast<uint64_t>(static_cast<VkSemaphore>(m_master.Handle())), job.tick, // NOLINT
		                        reinterpret_cast<const VkCommandBuffer*>(job.buffers.data()), job.buffer_count,
		                        reinterpret_cast<const VkSemaphore*>(submit.wait_semaphores.data()),
		                        submit.wait_ticks.data(), submit.num_wait_semaphores);
#endif
		result = static_cast<vk::Result>(Libs::Graphics::AntiLag::QueueSubmit(
		    static_cast<VkQueue>(m_graphics.queue), *reinterpret_cast<const VkSubmitInfo*>(&info),
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit, VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2));
		m_driver_tick.store(job.tick, std::memory_order_release);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("async vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit,
		                  job.debug_arg0, job.debug_arg1, job.debug_arg2, job.debug_arg3, job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	LocalPlatform::SetThreadName("Kyty.Submit");
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			m_submit_available.wait(lock, [this, &stop] { return stop.stop_requested() || !m_submit_jobs.empty(); });
			if (m_submit_jobs.empty()) return;
			job = std::move(m_submit_jobs.front());
			m_submit_jobs.pop_front();
		}
		QueueSubmit(job);
		{
			std::lock_guard lock(m_submit_mutex);
			++m_submits_done;
		}
		m_submit_available.notify_all();
	}
}

void CommandScheduler::WaitSubmitted() {
	std::unique_lock lock(m_submit_mutex);
	const auto queued = m_submits_queued;
	m_submit_available.wait(lock, [this, queued] { return m_submits_done >= queued; });
}

void CommandScheduler::StopSubmitThread() {
	if (m_submit_thread.joinable()) {
		m_submit_thread.request_stop();
		m_submit_available.notify_all();
		m_submit_thread.join();
	}
	std::lock_guard lock(m_submit_mutex);
	EXIT_IF(!m_submit_jobs.empty() || m_submits_done != m_submits_queued);
}

void CommandScheduler::BeginNext() {
	CheckActive();
	m_drain_entries = 0;
	BeginCommand();
}

} // namespace Libs::Graphics
