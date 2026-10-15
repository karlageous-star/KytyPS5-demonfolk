#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>

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

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

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

	struct SubmitJob {
		vk::CommandBuffer buffer = nullptr;
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
	void QueueSubmit(const SubmitJob& job);
	void SubmitThread(std::stop_token stop);
	void StopSubmitThread();

	MasterSemaphore              m_master;
	uint64_t                     m_last_pending_refresh = 0;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	uint32_t                     m_recorded_dispatches = 0;
	// The last tick handed to the driver (its vkQueueSubmit returned: on the recording worker for a deferred one).
	std::atomic<uint64_t> m_driver_tick {0};
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	// Count of live entries in m_pending_operations, updated under
	// m_operation_mutex and read lock-free as a "nothing to retire" hint.
	std::atomic<uint32_t> m_pending_operation_count {0};
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
