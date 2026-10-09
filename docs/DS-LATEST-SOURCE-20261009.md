# Demon's Souls: latest source review and bounded ports, 2026-10-09

## Source and scope

Canonical source: `C:\Users\himav\Desktop\KytyPS5 Source`.
Branch: `astro-ds-integration-20261009` (name retained from earlier work).
Latest user scope: Demon's Souls, review the current chenxiao release and latest upstream, include useful compatible improvements, build for manual testing, and finish. No further ASTRO BOT development or testing.

Fetched chenxiao `experiment/perf-40fps-20260926` at `b22d0f023ab24c084e756a7a8c1e17ca4c3be6c1`, release `v20261009`. Fetched upstream `main` at `7b9997baea0d385e9e8fc2bd3abeb02110bc4148`; upstream has not advanced since the earlier review today. Remote branches are references, not a wholesale replacement of this locally modified source. Local fixes, switches and the existing player are preserved.

Release: https://github.com/chenxiao07/KytyPS5/releases/tag/v20261009
Upstream: https://github.com/KytyPS5/KytyPS5/commit/7b9997baea0d385e9e8fc2bd3abeb02110bc4148

The release targets PPSA01341 v1.07 and v1.05. Our established game copy is PPSA01342 v1.04. Its game patches and automatic shader extraction cannot be assumed compatible. Maintainer FPS figures use an i9-14900K and RTX 5090, not this Ryzen 5 5500 / RX 6650 XT. No local FPS improvement or 30 FPS qualification follows from those figures.

## New chenxiao commits since 3ee9be77

- `4704d19a`: a bit table for deferred-protection packet classification and a high-water mark for constant RAM reset. **Ported the constant RAM part.** The local command processor has no `KeepsDeferredProtection` classifier or `CopyStateFrom`; their other hunks are not applicable. `WriteConstRam` is the only writer, construction initializes all words to zero, reset clears through the highest written word and resets the limit. Focused coverage checks sparse writes, the final RAM word, a lower write after a previous high write, and repeated empty resets.
- `19256455`: reuse temporary readback collections and date GPU writes in groups. **Ported only scratch reuse.** Each call takes storage from a per-thread home, leaving nested calls separate storage; only temporary range/copy/envelope/queue-region collections are pooled. Request-owned pages and parts remain independent. Vectors above 4096 elements are released instead of retained. Queue command recording consumes the region span before returning; no queued payload references this scratch. The newer GPU write-log representation is not transplanted into our separately guarded log index.
- `4b7f00c4`: optimistic texture-region and image-start queries before taking the cache lock. **Deferred.** Correctness relies on the newer stable filter storage and shared-read lock protocol. This is not equivalent to removing our current lock.
- `90e561e2`: slot liveness metadata before objects and table target liveness checks. **Deferred.** The release uses a different atomic, chunked slot container; ours uses deque slots and optional objects. Removing serial checks without porting and qualifying the ownership model is not justified.
- `d5d0b5b4`: a 64 KiB read-watcher granule bit filter. **Deferred.** Requires a complete audit of watcher updates, remaps and protection lifetimes, plus concurrency regressions.
- `5c0833a4`: unchanged image transitions avoid a state lock and global state epoch. **Not applicable as a direct port.** Our image transition function lacks the newer speculative state lock/epoch machinery and already returns early for whole-image transitions with the same layout/access. The existing guest-write-barrier override and mip-remapping behavior are preserved.
- `ea973702`: inline read-watcher granule tests in table evaluation. **Deferred with its d5d0b5b4 prerequisite.** The newer reader also participates in speculation; ours differs.
- `e9996ef7`: incrementally update the table buffer-region index on registrations. **Deferred.** The target index and lifetime/stamp representation are absent here; our native region epoch cache is a separate optional experiment. This is not a standalone compatible hunk.
- `b22d0f02`: release documentation and publishing. **Reviewed, not imported over local documentation.**

## Broader release changes

Since v20261007, the release also adds lane-local wave64 compute execution, broader table/native precompile variants, background cache saves, partial texture updates preserving untouched GPU bytes, GPU-flip garbage collection, and latest-frame presentation. Some corresponding work already exists in this fork; others depend on the newer table/speculation architecture. The optional 120 FPS game patch is version-specific and changes movie speed, so it is not applied to our game.

## Upstream ports

The exact float-class-mask subset of `7b9997ba` was integrated earlier today with CPU regressions. The new bounded adaptation of `13e3ca31` loads each packed shader-data word once for up to four memory byte offsets. It preserves our existing offset count, shift/mask decoding, buffer words and runtime-format setup, rather than importing upstream's different native emitter API. The last partial group remains bounded by the local count. Broader upstream IR/CFG/emitter rewrites are outside this wrap-up.

## Test status and known limits at commit time

Clean 7a1343d8 production/test build completed. Float-class, SPIR-V array and exhaustive half-float clear CPU tests passed. Direct recording-mode scheduler tests passed with ordinary async submit both off and on.

The first byte-exact GPU regression failed because its common store helper selected the aligned s48 descriptor, while loads used the misaligned s0 descriptor. Corrected the fixture to use s0 for both operations and added a specialization assertion. This is a test correction; final GPU rerun is required before enabling the option in a playable profile.

The GPU command-lane selector failed at `unmap native completion` both with async submit/throttle enabled and with both disabled. It expects an unrelated unmap to wait for an earlier generic deferred callback; current unmapping uses scoped readback/upload/image-download retirement. This result is unresolved and is not reported as a pass or blamed on one of the new options without a baseline comparison.

Final build/test results and exact package hash are recorded outside the source tree in `C:\Users\himav\Desktop\kyty ps5\Performance Experiments\DS-LATEST-SOURCE-20261009-RESULTS.md`, avoiding a dirty build or inaccurate binary stamp. Final normal profile keeps ordinary async submit, Function-array shrinking, wide clear decoding and target-check memo off. Previous player and reference executables remain available for rollback. Nothing from this integration has been pushed or published.
