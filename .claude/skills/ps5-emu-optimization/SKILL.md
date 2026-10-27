---
name: ps5-emu-optimization
description: Methodology-first playbook for making an HLE PS5 emulator (Kyty-style, Vulkan) run a real game fast, smooth and stable, and for specializing it to one title. Covers how to find and prove a bottleneck, measured wins and dead ends (general), Demon's Souls-specific facts, and how to find the same levers in another PS5 game. Use when asked to raise fps or 1% lows, remove hitches, cut RAM/VRAM, fix long-play slowdowns or crashes, judge an optimization idea, design an A/B, reverse-engineer a game's shaders/code/settings, or carry this work to a new title.
---

# PS5 emulator optimization: method first

Distilled from three weeks (2026-09-25 .. 10-15) and ~350 perf/fix commits taking Demon's Souls (PPSA01341) on KytyPS5
from ~21 fps to 60-105 fps standing (120 fps cap, i9-14900K + RTX 5090), from multi-second shader stalls to none,
and to flat memory over long play. Numbers are that project's; use them as orders of magnitude.
Source: branch `experiment/perf-40fps-20260926` of https://github.com/chenxiao07/KytyPS5 (`KYTY_*` switches,
scripts and file names below live there).

| Task | Read |
|---|---|
| How to measure, A/B, decide, test safely; the tools | `references/methodology.md` |
| What worked / failed in any game (check before proposing an idea) | `references/general.md` |
| Demon's Souls facts: shaders, patch sites, engine settings, passes, quirks | `references/demons-souls.md` |
| Specializing another PS5 title: what generalizes, how each lever was found | `references/other-games.md` |
| Guest code, shaders, patches, crash triage, recurring bug classes | `references/reverse-engineering.md` |

## The method (in order; skipping a step cost days every time)

1. **Set the target like the porting developer.** The console runs this at 60 fps: aim for the frame budget, not
   +10%. Name the structural cause; micro-tuning a structurally bound path ends flat.
2. **Find what bounds the frame before touching code.** Trace one frame flip to flip: render-thread busy vs idle,
   GPU busy, which queue/slice gates the next one, where the game waits. Only time on the critical path converts to
   frame time (a translation thread ~95% busy; GPU ~65% busy at the same spot means: CPU-side translation).
3. **Rank by per-call cost x calls per frame** (a census of scoped timers per phase/shader), then profile the
   critical thread with symbols of the exact build. Census numbers per call beat fps for small changes.
4. **Prove causality and the ceiling.** Inject a known delay at the site (frame-time slope ~1 = on the path, ~0 =
   saving it gains nothing). Make the unsafe "skip it" version and measure: a small ceiling ends the idea.
5. **Quantify the opportunity with a counter before building.** One cheap counter decided ideas in minutes (ring
   copies trimmable by static extents: 0.25% of bytes -> dropped; draws on the slow path: 0.14-0.47 ms -> not the
   lows' cause).
6. **Change only what you can prove removes work or waits, for the whole game.** No per-scene thresholds. Put it
   behind a live switch (a global you can poke) and add a verify mode when it replaces correctness work.
7. **A/B in one process, alternating** (poke the switch, A B A B, 4-6 rounds of 5 s at a fixed spot). Cross-boot
   variance is +-3-5% per world; read single sweeps as noise. Compare render ms / per-call census, not capped fps.
8. **Test short, then broad.** 1 boot, 2-4 rounds, 120 cap for a change; a screenshot for UI; then a multi-world
   soak (worlds x cycles) for stability and memory, and route tours for crashes before keeping risky changes.
9. **Keep or delete.** Not clearly faster -> remove the code (no default-off piles). Record every dead end with its
   numbers and reason, so nobody retries it without a new argument.
10. **Re-profile.** The bottleneck moves (translation-bound -> GPU tail -> game-side waits -> memory latency).

## What the profiler will not tell you

- **Overlapping cache misses.** Late per-draw costs are memory latency spread over many structures (pairs, sets,
  held entries, guest tables in 4 KiB pages). A lookup that shows 2% in the profile can be removed for 0 gain:
  four such changes measured flat (general.md). Confirm with census per call before and after.
- **Symbol misattribution.** Exports-only DLLs name samples after the nearest export: in vcruntime140
  `_NLG_Return2` = memmove's large-copy path, `memchr` = memcmp. Check with a disassembler before chasing.
- **Instrumentation cost.** Scoped timers inflate small phases (~10x); traces add ~10%; never quote instrumented fps.
- **Sampling the wrong build.** Profiles and crash addresses are valid only against that exact exe + map/PDB.

## Rules (the user's, and the ones that saved time)

- **Crashes first, then lows, then average fps.** Stability regressions are never traded for speed.
- **Never make users recompile.** No change to the SPIR-V of shaders already in users' caches unless the cache key is
  versioned; precompile speedups stay tool-side.
- **Logic-only optimization**; hit rate is not savings (count lookup + proof + invalidation + fallback).
- **Protect the player.** Benchmarks run only on a fixed baseline save, swapped in and restored with hash checks
  (restore by hand if a script is killed); one emulator at a time; never rebuild while it runs.
- **Diagnostics cost one relaxed load when off**; keep them out of commits (a separate uncommitted set).
- **Keep the console's timing.** Vblank 60 Hz by default; above it the game clock and movies need their own fixes.
- **Release only when asked; no upstream PRs without being asked.**
- **Short tests, more thinking.** Long sweeps for a 1% change waste hours; reason in code, test briefly.

## Where the time goes (model at 60-100 fps)

1. **The render (translation) thread bounds the frame** in heavy scenes: ~15 ms at Boletaria 1-1 standing, ~5.5K
   table draws at ~1.4 us and ~1.4K table dispatches at ~1.8 us. Heavy walking views have 13-17K draws -> 25-45 ms.
2. **Per-draw cost is memory latency**, not instructions: cold guest tables (4 KiB pages, TLB misses) and emulator
   lookup tables. It grows with distinct data per frame (walking views cost 2x per draw vs standing).
3. **Memory coherence tax**: page protection (every downgrade IPIs all cores running the process), ring copies of
   guest-written constants (~10 MB/frame), uploads, readbacks the game waits on.
4. **Stutter**: first-use translation/pipelines (precompile + recorded warmup), texture streaming bursts, guest
   partial unmaps (a Windows view cannot be split: unmapping it whole and mapping the rest again cost up to 29 ms a
   call on the game's threads; now parked in the view instead, general.md).
5. **Long play**: the game cuts its direct memory finer every level load (mappings 9K -> 50K over 24 loads); every
   per-mapping structure must be O(1) or epoch-cached, or later worlds get slower.

- **Logic-only optimizations.** Every change must name the work or wait it provably removes, for
  the whole game. No tuned thresholds or per-scene knobs; they don't transfer and hide the real cause.
- **Hit rate is not savings.** A cache with 91% hits lost 3.5%; a memo with 10.7K hits/frame was
  slower. Count the full cost of the lookup, proof, invalidation and fallback path.
- **Moving work to another thread rarely helps** when items cost 1-5 us: hand-off costs 0.3-1 us per
  item, plus lock contention and lost cache locality (render split: flat).
- **Diagnostics must cost one relaxed load when off**, and instrumented fps is never a result
  (timed scopes added +32% cost).
- **Default to the console's timing.** Vblank must stay at the console's 60 Hz; a faster virtual
  vblank sped up the game clock (cutscenes 4x, gameplay slow-motion).
- **Suspect hardware for random host crashes** on a machine that also crashes the compiler; pin
  work away from bad cores before debugging.
- **Protect user saves.** Benchmarks run on a fixed baseline save swapped in and out with hash
  verification; never run experiments on the player's own save.
- **One emulator instance at a time**; background scripts kill only the process they started.

## Directions worth trying (ranked by evidence)

Proven large wins, apply first if the emulator lacks them:
1. **Record the Vulkan command stream on a worker thread and defer submits** (translation thread only
   encodes): 21.1 -> 23.7 fps; drains 191 -> 5 per frame.
2. **Native replay records for repeated draws** keyed by shader address + user data, holding
   flattened resource tables in an upload ring and a write-once descriptor set; per-draw cost 0.25 us:
   24.9 -> 30.9 fps. Store a record only on the second sighting (keys new every frame: +6.6%).
3. **Readbacks that don't wait for the whole GPU tail**: copy on a transfer-only queue and wait only
   for the last writer of those bytes / the already-completed timeline value (+1.25 fps; sync
   downloads 1.19 -> 0.29 ms/frame). Most guest readbacks read data finished frames ago.
4. **Cheaper page tracking**: batch/async reprotect, 64 KiB "resource here" bitmaps, per-page
   invalidation epochs, lock-free watched reads, release locks before mprotect in the fault handler
   (+1.1 fps from that alone), partial dirty ranges for huge texture pools.
5. **Barrier dedupe** between dispatches with no work in between (-330 of 920 barriers, +1.4%).
6. **Build**: -O3, ThinLTO, PGO (+8.4% with AOT-compiled resource-table plans), a JIT for linear
   resource-table evaluation. Ship a portable x86-64-v3 build; -march=native leaks GFNI/VNNI.

Stutter:
7. **Build missing pipelines unoptimized first** (VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT, ~150x
   faster on NVIDIA), compile the optimized one in the background and swap: cold stalls 22 s -> 0.
8. **Static precompile from the game files**: enumerate every shader the game ships (bundles + shaders
   embedded in the executable), derive their compile keys, compile offline into a pipeline cache keyed
   by GPU+driver; translate the rest in background threads at startup (prefetch).
9. **Fix translator complexity** (a dominator tree instead of iterating dominator sets: worst shader
   862 -> 77 ms) and make SPIR-V deterministic (hash-order ids made caches never hit).
10. **Never re-create big resources on partial unmap**: keep texture pools alive, upload only mapped
    parts, pool scratch allocations (a 2.5 s freeze came from a new VMA block per detile).

Open, promising (not done in the source project): batching runs of the same compute shader into one
dispatch (~2 ms estimated), async compute for full-screen post passes, host-pointer rings
(VK_EXT_external_memory_host) to avoid mprotect, fast paths for plain 32-bit formatted buffer loads,
persisting background-translated SPIR-V between launches.

Tried and not worth it (details in lessons.md): per-draw preparation reuse, render-thread split,
native replacement of culling/shadow/geometry chains, snapshotting writes for readbacks, shader
objects, DX12, pipeline libraries, strict thread pinning, skipping readback copies.

## Recurring bug classes to check first

- **Stale ranges after guest unmap**: caches, records and uploads that still read an unmapped tail
  (host access violation). Track unmaps with an epoch and re-validate.
- **Lost write protection**: partial unmap/remap restoring the original page access; untracked
  guest writes show up as frozen or detached models ("out-of-body").
- **Lock recursion**: re-acquiring a shared lock (SRW/shared_mutex) behind a waiting writer deadlocks.
- **Stale GPU state across mixed fast/slow paths**: bump the state generation when a fast path
  bypasses the normal binder (black silhouettes, stray triangles). Verify modes miss this; screenshots don't.
- **Queue ownership/slot reuse across queues**: bytes from another window returned ~every 2 s.
- **Windows specifics**: exception dispatch clobbers the guest SysV red zone (patch guest code at load);
  NVIDIA vkQueuePresentKHR holds a device lock (no wait-before-signal pending on the recorder, own present queue).
- **Nondeterminism** (pointer-keyed maps) breaking cache keys.
- **Implicit CP writes ignored**: indirect draws name user SGPRs the CP fills from their arguments
  (START_INST_LOC); the instance ID VGPR counts from 0. Missing or floating objects
  (reverse-engineering.md, "Missing or floating geometry").

When a crash is reported: get host frames (RtlVirtualUnwind or backtrace), the fault address' page
state and the guest registers, symbolize with the build's linker map, then reproduce with the
save/route that triggers it. Guessing from the symptom produced one wrong fix in the source project;
the host frames found the real one immediately.
Directions that paid, ranked, and the dead ends: `references/general.md`.
