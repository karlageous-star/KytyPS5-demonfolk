# Ordinary queue submission port

Reference: `49be9a0517a974149b0a41e4f1b43f007d48a84c` in the ASTRO-performing `f9464f82` line.

`KYTY_ASYNC_SUBMIT=1` enables a dedicated FIFO queue-submit worker in the ordinary renderer path. Default is off. Vulkan recording disables this route; the existing deferred recording worker remains responsible for that path. Switching recording on after the ordinary worker starts is rejected.

The producer ends the command buffer, copies all submit arrays and error breadcrumbs, captures the async-upload sequence, allocates its timeline tick in FIFO order, and queues an owned job. The worker waits for that upload sequence before submitting under the existing graphics queue mutex. No caller stack pointers or mutable command state are retained. Command-pool entries remain tagged with their recording tick and cannot be reused until that tick completes on the GPU.

Shutdown submits the final command buffer and waits through the final allocated tick before joining the worker. The worker drains any remaining jobs on stop. `WaitSubmitted` snapshots the queued-job count and waits only for driver acceptance, outside queue locks. Windows presentation uses this before taking presentation locks, preserving the newer fork's defense against presenting a semaphore whose signal is still queued on a CPU worker.

The scheduler timeline test covers waits before submission, pool growth behind an unsignaled external semaphore, reuse after completion, 1,027 resource versions, concurrent/reentrant deferred callbacks, and shutdown. It now calls `WaitSubmitted` before signaling that external semaphore, distinguishing driver publication from GPU completion. Run with `KYTY_ASYNC_SUBMIT=0` and `1`, then the GPU command-lane and stream-buffer tests in both modes. Game qualification must use matched ASTRO patch/cache/resolution and recording off. No performance or device-loss fix is claimed before measurements.

Revert the commit or leave the option unset to restore synchronous submission. No title defaults are changed.
