# Specializing for another PS5 title

What Demon's Souls showed to be generic (put it in the emulator: every game gains) and what is title-specific (put it
behind an exact title + version + byte check), and how each title-specific lever was found, so the search can be
repeated on a new game.

## Generic: do it in the emulator

- Architecture: recording worker + deferred submits, table mode (translation-time snapshots of resource tables, V#
  cache with epoch proofs, image sets validated per frame), PM4 look-ahead, batch boundaries, upload worker, readbacks
  that wait only for the last writer, barrier rules derived from the guest's own flushes. Every title is bound by
  per-draw/per-dispatch translation the same way; only counts and passes differ.
- Correctness found through one game helps all: START_INST_LOC SGPR + base instance, wave64 lane-locality, BC blocks
  written through uint views, DCC fast clears, multichannel ATRAC9, file ids by path (no hash collisions), kept file
  handles with positional reads, emulator-owned fiber stacks, Windows partial-unmap windows.
- Shader pipeline: inventory from game files + recorded play -> static precompile + recorded warmup + shipped hints;
  never invalidate users' caches.
- Memory policies by machine (RAM 16/32 GB, VRAM 8/12/16+ GB), garbage collection by last use, long-play hygiene
  (structures O(1) or epoch-cached in the number of guest mappings), soaks across levels.

## Title-specific levers and how they were found (repeat these steps)

| Lever | How it was found | What to expect in another title |
|---|---|---|
| Engine settings (convars) | registration pattern in the decrypted executable (lea object / lea name / bool ctor), names + ids by string xrefs, value byte = table base + id; the engine read a command-line args file, injected through a temporary copy; A/B by poking the live byte | most engines have one; survey every boolean with same-process pokes; keep as defaults only image-identical wins (GPU index-buffer culling off: -2800 dispatches, +6.4%); some apply only at level load |
| Engine console | debug-helper strings (`+cp11_...`) and the game's own format strings (`+cp11_loadLevel %d %d`); commands run on a game thread | in-game level loads in 2-4 s make soaks and tours practical; worth hours of searching |
| Engine log | hook the formatted log function in a code cave | named a resource-loading bug (file-id collision) in minutes; do it on day one |
| Render size / aspect | the 16/9 float immediate `0x3fe38e39` in the render setup; the video modes' width/height tables | change the 3D render size, keep the output size (the HUD is laid out for it); patch by unique pattern per version |
| HUD vs 3D vs menus | UI drawn by DrawAuto into a separate 4K RGBA16F canvas; the composite pass reads scene + canvas; UI shaders by hash + user-data layout | mark "scene frames" by one pass (TAA) ran; classify UI per vertex (text batches span areas); menus may be left stretched |
| Frame pacing | time census of sleep/wait/vblank HLE calls by guest call site; the step function's dt clamp convars (gameMin/MaxDeltaTime); play speed measured in m/s from the player position | above 60 Hz the clock may run fast or movies play fast; fix the step and hold non-scene frames to 60 |
| Frame generation inputs | TAA = the compute pass reading history + motion + depth and writing one RGBA16F image; camera rows in the G-buffer VS constants, found by matching words across frames; motion-vector sign checked by turning | expect per-area G-buffer shader variants: a plausibility check on the matrix, or learn layouts |
| Engine work the console pays cheaply | census by shader: thousands of tiny culling dispatches, the engine's memset/memcpy compute shader (count == period), spin-waiting job workers | CPU paths by shader hash when provably equivalent; spin back-off; convars that skip GPU-cheap/CPU-expensive passes |
| Readback sources | sync-readback logging with guest stacks, watch ranges naming the writing shader | most guest reads of GPU data were statistics/culling, not gameplay |
| Engine races exposed by timing | crash dumps with guest registers + write-back evidence ruled out the emulator; repro by teleport tour | guard per traversal site, exact version + bytes |
| Saves and positions | save container (hash + key table), map ids from map file names, spawn coordinates from map files (MSB), player structure from the character-manager singleton, the game's own warp flag | any start position without playing; benchmark saves the player's save never touches |
| Shader inventory | bundles (footer/header formats) + shaders embedded in the executable; materials pair VS/PS; recorded tours for exact specializations | seeds + hints shipped without game code |
| Game versions | 1.05 vs 1.07: almost every shader recompiled, eboot rebuilt by another compiler | never hard-code RVAs: masked signatures, per-version hash tables and caches |

## Bluepoint-engine patterns (likely in other BPE titles)

Job system on fibers (16/64 KiB stacks back to back, a spinlock held across the fiber switch, workers spinning in
WaitForJobs), eastl::vector scene children without locks (timing races), per-frame render contexts and rings
(resource-table pointers new every frame), a resource pool resolving every file path at start, Scream audio with
ambisonics object ports, Bink 2 movies, `cp11_*` convars and a `+command` console, direct memory cut into scattered
64-512 KiB pieces and partially unmapped while streaming.

## First week on a new title

1. Boot and render; triage crashes with host frames (reverse-engineering.md); fix generic bugs generically.
2. Harness: save swap with verification, scripted start, live measure/poke/census/trace.
3. One frame's anatomy at a fixed spot; census by shader and phase; name the dominant passes.
4. Hook the engine log; find the convar table and the console; build in-game level travel.
5. Convar survey by live pokes; frame pacing above 60 Hz; HUD/scene/composite passes.
6. Shader inventory -> precompile seeds; record a tour; check coverage per stage.
7. Soak across levels (memory, mapping counts, fps per cycle) before any release.
8. Keep a facts file like `demons-souls.md` (hashes, addresses, versions, passes, what each one is).
