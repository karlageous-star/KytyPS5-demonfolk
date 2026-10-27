#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <array>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <span>

#include <queue>

#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	// Finish a complete dispatch; may submit, but never waits for the GPU.
	void           CompleteDispatch();
	// Finish `draws` complete draws; may submit (KYTY_DRAW_BATCH), but never waits for the GPU.
	void           CompleteDraws(uint32_t draws);
	// KYTY_FLUSH_ON_DRAIN: at a draw or dispatch entry, submits what is recorded once the GPU has finished every
	// earlier submission (never waits).
	void           FlushIfDrained();
	// KYTY_FLUSH_DEFER: a flush nothing waits on (CommandProcessor::BufferFlushDeferrable) stays open while the GPU
	// has the configured backlog to run; FlushDeferred submits it before the GPU thread waits.
	void           DeferrableFlush();
	void           FlushDeferred();
	[[nodiscard]] bool FlushPending() const noexcept { return m_flush_deferred; }
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	// Wait only until queued ordinary submissions reach the driver, without holding queue_mutex.
	void                      WaitSubmitted();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations(bool at_draw_entry = false);
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept;
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	// Upload dirty pages in a prologue that runs before the open command buffer.
	[[nodiscard]] vk::CommandBuffer UploadPrologue(uint64_t last_dirty, uint64_t written_serial);
	void SetPrologueHook(std::function<void(vk::CommandBuffer)> hook) { m_prologue_hook = std::move(hook); }
	[[nodiscard]] uint64_t CommandSerial() const noexcept { return m_command_serial; }
	[[nodiscard]] uint64_t         CurrentTick() const noexcept;
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	static constexpr size_t MaxSubmitEntries = 8;

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	struct SubmitEntry {
		vk::CommandBuffer buffer         = nullptr;
		uint32_t          timestamp_slot = UINT32_MAX;
	};

	struct SubmitJob {
		std::array<vk::CommandBuffer, MaxSubmitEntries> buffers {};
		uint32_t                                        buffer_count = 0;
		SubmitInfo submit;
		uint64_t tick = 0;
		uint64_t upload_sequence = 0;
		uint32_t debug_op = 0;
		uint64_t debug_submit = 0;
		uint32_t debug_arg0 = 0, debug_arg1 = 0, debug_arg2 = 0, debug_arg3 = 0;
		uint64_t debug_arg4 = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);
	uint64_t SubmitBuffers(std::span<const SubmitEntry> entries, SubmitInfo submit);
	[[nodiscard]] SubmitEntry ClosePrologue();
	[[nodiscard]] bool GpuBusy();
	[[nodiscard]] bool BacklogAtLeast(uint64_t count);
	void QueueSubmit(const SubmitJob& job);
	void SubmitThread(std::stop_token stop);
	void StopSubmitThread();
	// A deferred flush done once the GPU is down to its last submission, asked every 16 units of `weight`
	// (DeferrableFlush).
	[[nodiscard]] bool     FlushedDeferred(uint32_t weight);
	std::function<void(vk::CommandBuffer)> m_prologue_hook;

	MasterSemaphore              m_master;
	uint64_t                     m_last_pending_refresh = 0;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	CommandBuffer                m_prologue;
	uint64_t                     m_command_clock  = 0;
	uint64_t                     m_command_serial = 1;
	uint32_t                     m_recorded_dispatches = 0;
		uint32_t                     m_recorded_draws      = 0;
	// KYTY_FLUSH_ON_DRAIN: draw and dispatch entries since the last submission, and the last drain check (QPC).
	uint32_t                     m_drain_entries       = 0;
	uint64_t                     m_last_drain_check    = 0;
	// KYTY_FLUSH_DEFER: DeferrableFlush left the open buffer open, and the units asked since the last check.
	bool                         m_flush_deferred      = false;
	uint32_t                     m_deferred_checks     = 0;
	// The last tick handed to the driver (its vkQueueSubmit returned: on the recording worker for a deferred one).
	std::atomic<uint64_t> m_driver_tick {0};
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	// Count of live entries in m_pending_operations, updated under
	// m_operation_mutex and read lock-free as a "nothing to retire" hint.
	std::atomic<uint32_t> m_pending_operation_count {0};
	// The tick of m_pending_operations' oldest entry (UINT64_MAX: none), published with the count.
	std::atomic<uint64_t> m_pending_front_tick {UINT64_MAX};
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	// Local diagnostic: records every tick's completion while the live trace runs.
	std::jthread                 m_tick_monitor;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	// One producer, one FIFO consumer; command-pool reuse still waits for GPU completion.
	std::mutex m_submit_mutex;
	std::condition_variable m_submit_available;
	std::deque<SubmitJob> m_submit_jobs;
	uint64_t m_submits_queued = 0, m_submits_done = 0;
	bool m_async_submit = false;
	std::jthread m_submit_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
