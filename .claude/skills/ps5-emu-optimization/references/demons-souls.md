# Demon's Souls (PPSA01341) facts

01.007.000 ("1.07") unless noted; eboot mapped at 0x900000000 (decrypted file: va = file offset - 0x4000). Bluepoint
engine (BPE). Hashes are the emulator's shader hashes; 1.05 recompiled nearly every shader (see Versions).

## Shaders and what they are

| Hash / address | Role |
|---|---|
| CS a44e580f4f0cfb2b (1.05 934a6e1197b99148) | TAA at render size (2560x1440, 320x180 groups): slot 0 color RGBA16F, 1 depth R32F, 2 stencil R8UI, 3 motion RG16F, 4 history, 5 mask, 6 output (only write). Its dispatch marks a "scene" frame |
| G-buffer VS f4d2e78cd1769fc7, 08e16923875aa8e9, 0ffcbf08be26e74e (1.05 1d69b5d2..., 5931e753..., a97f6108...) | camera in flattened SRT words: 29 view-proj (row-major, clip = M x world), 45 previous rows x/y/w, 57 jitter (NDC); other areas use other G-buffer shaders/offsets (a fork lists 22) |
| VS 3ec043bd86013caf + PS 1ad8b5870fe4d3ec | final composite to the 3840x2160 flip buffer: scene (render size) + UI canvas 0x282e50000 (3840x2160 RGBA16F); the "4K upscale" is spatial |
| VS 7e880a66bf1b69c6, b2f062027869d4ee (clip xyz in user V#0), bd39a2eba2d20d8b (pixels; constants+680 = 1/3840, 1/2160); PS 4d86d2e2bb1b49cd | HUD/UI DrawAuto into the canvas; the icon PS masks by window xy (const rows 0/1); text is batched across HUD areas; the round icon is the first UI draw |
| CS eb7456322124ecc7 at 0x2088a6600 (1.05 81bb2b1b9c751eca) | engine cs_memset32 = linear copy `dst[i] = src[i % period]`: ~520/frame, 13.7 MB, one 8 MiB + one 4 MiB copy on the culling chain -> CPU path |
| CS 71449641.., b78df07e.., 0794beec.., 958f6e60.. | per-draw triangle (index buffer) culling, ~2800 dispatches/frame -> removed by convar |
| CS ff095e4db6ee5511 (hi-Z), 62e514c79526925a (LOD), 8a7314e3c98ff22e / f133d4a008c3259d (object culling) | culling chain on async queue q49 |
| CS 7e676bf37482d768, decd2b04b5eee050, a3ecbeb1830c210c, 239d5b60.., fc1ba8fc.., de0f4e52.. | end-of-frame lighting tail using the light-union idiom (not lane-local: 2 lanes per invocation) |
| CS d353c2f03e865aad, 6be32a28a3623a8b, db63cac096be744a | light loops: dense indirect probe image (records base+0xC00+i*0x90, T# at base+0x20E0+key*32) |
| CS 55d35da67c9a387a, b59223a986bb112e | runtime BC compressors (write BC3/BC7 blocks through R32G32B32A32_UINT views); PS f240d49b18a9441f = character-creation face |
| CS 0x90271be00 | system image clears (15 MiB every loading frame) |

603 of 622 compute shaders seen at 1-1 are cooked particle shaders, one per emitter, run once a frame.

## Patch sites and runtime addresses (1.07)

- Render setup aspect `c7 43 04 39 8e e3 3f` (`mov dword [rbx+4], 16/9`): 0x90090ae7f (1.05 0x9008ef287); video-mode
  tables of render widths/heights (mode 3 case: output pair then render pair). Patch by unique pattern.
- Time step 0x900b280e0: `dt = max(dt, gameMinDeltaTime < 0 ? fixedDt : gameMinDeltaTime)`; the frame-rate getter
  0x901471b10 only feeds a job time slicer (patching it caps nothing).
- Convar value byte = table base 0x902F13B40 + id (enableAsyncCompute 0xca0, groupCulling 0xc9c,
  r_ForceGpuWaitForPrevFrame 0xce6); XPR convar registrations at eboot va 0x975900-0x976700 (bool ctor 0x450a20).
- Engine command entry 0x90081a990 (console object q[0x902ee4ee8], "+name args"); GameMan travel issues
  `+cp11_loadLevel W L forceReload=true unlockArchstones=false`.
- Player chain: ChrInsMan q[0x9027cf378] -> C = q[+0x258] -> K = q[C+0x38] -> H = q[K+0x20]; HP C+0x710; position
  H+0x20; teleport: K+0x100 position, K+0x110 angles, then byte K+0xfc = 1; no-death OR 0x50 into C+0x580;
  playerNoDamage byte 0x9027cbd50 (1.05 ChrInsMan 0x902898070).
- Job system: WaitForJobs spin 0x83b6c0; fiber Run/ReturnToThread/Switch 0x841f00/0x841f50/0x841fa0 (spinlock
  [obj+0xe08] held across sceFiberSwitch); heap alloc/realloc/free 0x9007f9cf0/0x9007f9f70/0x9007fa380; engine log
  eboot+0x8456f0 (hooking it named a resource-id collision in minutes).
- Guest patches that paid: compute barriers between dispatches omitted +12.7%, memmove fast path +5%, job-worker spin
  -> sleep (CPU 1338% -> 388%).
- Scene-tree null guards (game race exposed by timing): eboot 0x8dafd0, 0xb9d1d0, 0xc5ca90, 0x8db150 family,
  0x8da649, 0x8dad00, 0x8d8c10, character slots 0x9003510e0; int 0x41 after a Scream SamplePlayer assert at
  eboot+0x1e22491 is skipped.

## Engine settings (convars)

Injected as `+name=value` lines through a temporary copy of `$/packagecmdlineargs.txt` (never edit the game files);
some apply only at level load.
- `doIndexBufferCulling=false`: -2800 dispatches/frame, +6.4%, image identical -> default.
- `gameMinDeltaTime=0.004` above 60 Hz: otherwise play speed ~ fps/60 (measured in m/s from the player position).
- `enableAsyncCompute=false`: Latria +4.9%, Shrine -4..-10% -> not a default.
- Diagnostic ceilings only (visual changes, never defaults): `r_projectedDecalEnable=0` (4860 of ~11000 draws are
  decals) or `r_enableCascadeShadows=0` (~1865 draws) each reached the 60 fps cap from the mid-40s at 1-1 (10-06);
  fog off ~+4.6%; `GIEnable=0` breaks the frame. No effect: r_ForceGpuWaitForPrevFrame, cullZeroIndexIndirectCalls,
  useSunShadowIndices, groupCulling, r_fps.
- Test: `cp11_characters_playerNoDamage`, `cp11_unlockAllArchstones`, `cp11_allowInterworldArchstoneWarps=true`
  (set live), `sound_enableAmbisonics=false` (the game's own stereo mix).

## Frame structure

- 1-1: 1024x512 depth segment with ~1245 async dispatches; 7 spot shadows (1024) + 4 sun cascades ~1900 draws;
  prepass ~790; G-buffer (6 targets) ~750; decals ~650-750 (stencil mark + 7 PS sampling depth as R32F and
  read-writing a G-buffer image as storage); lighting/post ~1.5 ms.
- Graphics command buffers carry no SET_CONTEXT_REG (state through *_REG_INDIRECT tables) and start with a
  WAIT_REG_MEM; per frame ~17.7K SET_SH_REG, ~11K indirect register blocks, ~2.8K DRAW_INDEX_INDIRECT, ~2K DISPATCH.
- START_INST_LOC = SPI_SHADER_USER_DATA_GS_2 (s10): v22 = (s10 & 0xffff) + v8, s10 >> 16 = per-object vertex
  offset; 95% of indexed indirect draws start above instance 0.
- Frame end: partial flushes + RELEASE_MEM + WAIT_REG_MEM on itself + frame label; the async culling chain (q49, one
  ~30K-dword command buffer, ~350 dispatches) waits on that label (on the PS5 too) -> no tail/culling overlap; ~10 job
  threads then read culling results (readbacks) before the main submit; occlusion results are read two frames late.
- The game's CPU culling emits only visible draws (0-1.3% zero-count indirect draws). Draws repeat structurally
  (96.6%) but every evaluated word changes when the camera moves: per-frame rings for tables and constants.
- Pacing: the game never waits for vblank on the CPU; the flip queue (one flip per vblank) is the only cap. Movies
  (Bink 2, CPU-decoded, one video frame per flip) need non-scene frames held to 60 above 60 Hz.

## Memory and streaming

- Guest backing 13.5 GiB, ~11.3 GB mapped. Pools: BC textures 0x1009000000 (~1700 layers), a 320 MiB shadow depth
  array 0x102a400000 (1024x1024x80) whose free layers the texture streamer borrows; one 1280x720 surface (0x28a010000)
  alternates D32+HTILE and R32F twice a frame.
- The game builds 120-280 MiB regions from scattered 64-512 KiB physical pieces, releases pool layers by unmapping,
  unmaps a pool of 866 ranges in one munmap (from a job fiber), and cuts memory finer on every load (mappings 9.4K ->
  51K over 24 loads). Partial unmaps of 3-8 MiB views happen in bursts on the submit thread while streaming.
- Level entry streams 1-1.25 GB/frame, then a ~9 s burst; mips arrive by T# BASE_LEVEL; the resource pool resolves
  ~250K file paths at start. Small audio pieces were read synchronously per 16 KiB (keep handles, read through).
- 128 fibers x 16 KiB + 32 x 64 KiB laid back to back: host code running on them overflowed into neighbours
  (emulator-owned fiber stacks fixed a whole crash family).

## Versions 1.05 vs 1.07

1.05 = SDK 9 (fake-signed SELF), 1.07 = SDK 12; same title id and saves; only 10 of ~21K shaders byte-identical
(register allocation) -> per-version shader tables and caches named <title>_<version>; the eboot is the same source
built by another compiler -> patches located by masked signatures (rel32/RIP/frame-size wildcards). 1.05 went 8 -> 60
fps once its idle patch matched.

## Saves and test automation

- USR-DATA: u32 version 7, FNV-1a32 of bytes[16:]; entries keyed by FNV-1a32 of the UPPERCASE name (MAPUID
  0xWWBBCCDD, TRANSFORM, ...); position flag 0x4819bb27 must be 1. Maps: m01 Nexus, m02 Boletaria (1-x), m03 Shrine
  of Storms (4-x), m04 Latria (3-x), m05 Valley (5-x), m06 Stonefang (2-x), m08 tutorial. MSB part positions are
  save coordinates; spawns c0000_*.
- In-game travel: `gamecmd +cp11_loadLevel W L` (2-4 s): 1 1 Boletaria 1-1, 2 1 Stonefang, 3 1 Latria prison,
  4 1 Shrine, 5 1 Valley, 0 1 Nexus. God mode + teleport over MSB enemy coordinates = route tours.
- Reference spots: 1-1 standing at the start (render ~15 ms, the heaviest standing view), Latria initial camera spin.

## Audio, HDR, frame generation, HUD

- Audio: Sony Scream; gameplay sound is two 5th-order ambisonics scenes on object ports; 1330 AT9 files are 16-channel
  third-order ambisonics (one mono block per channel); quiet NPC speech is the game's mix.
- HDR: the game flips PQ buffers when the output reports HDR10 (param attribute bit 0x20000000).
- Frame generation: motion vectors point along the motion (DLSS wants the opposite sign); keep the UI canvas out;
  camera from the G-buffer constants above.
- HUD/menus are laid out for 16:9 (a wider output got a left-aligned HUD); every frame has full-screen alpha quads, so
  "menu" heuristics fail. Ultrawide = wider 3D render + patched aspect + HUD anchored per area; menus stay stretched.
