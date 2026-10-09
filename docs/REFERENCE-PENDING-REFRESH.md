# Reference scheduler function merge

The draw-entry timeline refresh throttle from reference f9464f82 (original change f16d4d1d,
subsequently narrowed there) is merged with this branch's empty-pending-queue fast path.
`KYTY_PENDING_REFRESH_THROTTLE=1` limits timeline queries to one per 200 microseconds at draw,
draw-run and dispatch entry. The original pending-operation retirement and priority-thread waits
are retained. Finish, shutdown and other explicit drain callers refresh immediately.

The reference's rejected leave-queued priority-operation variant is excluded; it previously broke
the GPU command-lane test. The option defaults off pending scheduler, GPU command-lane and
same-route ASTRO/DS qualification. Clearing the option restores the original timing policy.
No local FPS improvement has been measured yet.
