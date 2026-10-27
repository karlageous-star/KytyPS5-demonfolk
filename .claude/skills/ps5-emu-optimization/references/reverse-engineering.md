# Reverse-engineering the guest game

## Address model

- In Kyty the main module (eboot) is mapped at `0x900000000`; its first PT_LOAD has vaddr 0, so a
  guest address is `0x900000000 + va`. For a decrypted eboot file, `va = file offset - 0x4000` (check
  the program headers of your dump). Guest code is identity-mapped inside the emulator process, so the
  emulator's own debugger/memory reader sees it at the same addresses.
- Host code: link the emulator at a fixed base (Windows: `/DYNAMICBASE:NO`, base `0x140000000`) so
  runtime address = map RVA + base. Classify sampled addresses as guest (`0x800000000..0x1000000000`)
  or host by range.
- Decrypted modules may exist only in memory: read them from the running process (Linux
  `/proc/PID/mem`, Windows ReadProcessMemory + VirtualQueryEx) rather than from disk.

## Disassembly workflow

- Dump bytes from the live process and disassemble with objdump/llvm-objdump using `--adjust-vma` so
  addresses match: e.g. `guest-disasm.py 0x9002dd700 --size 0x100 --mark 0x9002dd73b`.
- Useful modes: `--find <hexbytes>` in a range (optionally data regions), `--dump` 64-bit words,
  `--regions` (executable regions), `--xref <addr>` (scan for rip-relative disp32 that resolve to the
  target; try instruction tails of +4/+5/+8).
- Sample guest threads with pc + callers (the word at rsp is the likely caller: guest code has no
  unwind info) and bucket by 64-byte granule to find hot guest functions.
- Time-HLE census: record the guest return address of every sleep/wait/vblank/equeue/TSC call (the
  call goes through the PLT, so the return address is the real call site). This finds spin loops and
  frame-pacing logic quickly.
- Engine conventions found this way in one title: convars registered with a lea-object/lea-name
  pattern (disable layers such as particles or XPR rendering via a command-line args file the game
  reads), a boot-flow state machine with a `table + state*0x10` handler table and state at `[obj+0xc]`,
  and an FNV-1a-style uppercase-folded name hash. Recover them with small scripts (string xrefs,
  pattern scans) and keep the scripts in the repo.

## Shaders

- Bundles (`*.csdr` in the source title): footer `RDSC` + payload length; header with a count;
  20-byte entries (stage, flags, code bytes, header bytes, relative); 0xff padding; an AGC header
  (starts `'1234'`) after each code block. Some shaders are embedded in the executable's data
  segment as an AGC header array with code 256-byte aligned, anchored by relocations.
- AGC header (96 bytes): self-relative 64-bit pointers, version, header size at 64, shader size at 68,
  type at 90. Port the console's static-input-info logic (resource usage from register lists) to
  compute the same compile keys the emulator computes at run time.
- Materials (`.cmat`, footer `STAM`) name the bundle and the techniques (VS/PS pairs, CS), which gives
  the VS/PS pairing; render-pass states (targets, blend, depth) are learned from a recorded run.
- The resulting seed file drives offline compilation (shard across processes on NVIDIA). Validate by
  comparing the modules a real run compiles against the precompiled set (coverage by stage).

## Missing or floating geometry

An object that is missing or floats while the rest of the frame is right is usually per-draw data
read at the wrong index, not a broken shader. Before bisecting draws, check what the CP writes by
itself for each draw packet. In Demon's Souls (a brazier at Boletaria 1-1 floating without its
stone pillar) the cause was in the original emulator:
- PM4 indirect draws (DRAW_INDIRECT, DRAW_INDEX_INDIRECT, the _MULTI forms) name SH registers the CP
  fills from the draw's arguments: START_INST_LOC (here SPI_SHADER_USER_DATA_GS_2) gets the start
  instance, BASE_VTX_LOC the base vertex. Both fields were ignored (the CPU path logged "partial
  args"), so shaders indexing per-object data with that SGPR read a stale user-data value.
- The instance ID VGPR counts from 0 on the GPU; Vulkan's gl_InstanceIndex includes firstInstance.
  Shaders adding the SGPR to the ID worked by accident; shaders using the SGPR alone broke.
- Fix: instance ID = gl_InstanceIndex - gl_BaseInstance; the named SGPR = gl_BaseInstance (shaders
  are translated per start-instance SGPR); firstInstance = the arguments' start instance, which also
  covers arguments that stay on the GPU. 95% of the game's indexed indirect draws start above 0.
- Proof before the fix: an experiment writing the SGPR on the CPU path (firstInstance 0) restored
  the pillar exactly as on the console; skipping draws, occlusion queries and culling switches had
  only ruled things out.

## Game patches

- Format used: etaHEN-style JSON (`id`, `version`, `process`, `mods[].memory[] {offset, off, on}`).
  Validate title id, app version and process name; find the module base by searching the first
  non-zero `off` pattern in loadable segments; verify every `off` sequence before writing; allocate a
  code cave for writes outside the module; flush the instruction cache; apply when the module loads.
- Several game versions: never hard-code RVAs. Locate sites by masked signatures (wildcards for rel32 targets,
  RIP displacements, frame sizes) and require a unique match; keep shader-hash tables and caches per version.
- Emulator-side guest hooks (instead of patches): verify each audited function by RVA, size and hash
  before redirecting call sites; fall back to the original behaviour if any differs.
- Examples that paid: skipping boot logos/character creation for benchmarking, removing over-conservative compute
  barriers, a faster memmove, sleeping or backing off instead of spinning, the render setup's aspect, null-child
  guards in scene traversals. Always keep them title+version-exact.

## Recurring bug classes (check these first)

| Symptom | Cause | How it was found |
|---|---|---|
| Random guest crashes with garbage `this`/RIP, corrupted lock wait blocks, asserts | host code (exception dispatch ~5 KiB, HLE calls, a 40 KiB temporary) ran on the game's back-to-back 16/64 KiB fiber stacks and overflowed into a neighbour | every crash's RSP was within 3 KiB of a fiber stack top; painted stacks; fix: emulator-owned fiber stacks with guards |
| Silent exit, only a WER dump | a recursive helper (one level per range, 866 ranges) exhausted a fiber stack; a no-access guard cannot dispatch on an exhausted stack | PAGE_GUARD guards + an overflow reporter; make the helper iterative |
| Job-worker crash 1-2 ms after a readback wrote back GPU data | Windows partial unmap = whole view unmapped + parts remapped: windows where trackers' protection or the map entry were missing | crash report listing recent write-backs onto pages the registers point at |
| Null-child crashes in scene traversals | the game's own race, exposed by different timing | write-back ring, mappings and allocator ruled out the emulator; teleport tours reproduced it in 1-3 min |
| Stale ranges after guest unmap | caches/records read an unmapped tail | per-16 KiB-granule unmap epochs (a "last N unmaps" epoch invalidated too much) |
| Frozen or detached models ("out-of-body") | lost write protection after partial unmap/remap: guest writes untracked | protection restored under the lock for watched runs |
| Missing or floating geometry | per-draw data read at the wrong index (START_INST_LOC ignored; a cache key without the start-instance SGPR) | see "Missing or floating geometry" |
| Black world / flicker after a fast path | reads before a pending CP copy; skipped publication epochs; a fast path bypassing the normal binder | slot-hash diff of the same segment fast vs normal; screenshots (verify modes miss these) |
| Device lost | kept descriptor sets holding destroyed buffers (handles recycled as addresses); a barrier recorded before Begin; BDA residency skipped for pointer loads; buffer freed while a download waited only on the last write | validation layer with errors printed and packet caller tags; VK_EXT_device_fault addresses vs live/destroyed BDA buffers; "in use" errors after a loss are symptoms |
| Dirty/wrong textures | GPU BC compressors write blocks through uint views kept as separate images | decode the textures; ground truth from the game's own thumbnails |
| Default "?" models | file ids = path hash (13 collisions in 250K paths) | engine log hook printed the failed resource |
| Exit on garbage descriptors while loading | invalid DCC metadata, unknown formats, T# outside guest memory, SRT address -1 | bind null / read zero instead of exiting |
| Hang in the crash reporter, lock recursion | reporter blocked behind a blocking read; re-acquiring a shared lock behind a waiting writer | external stack dump of all threads of the hung process |
| Emulator silently loses temp-file writes | a Low integrity label inherited from a sandboxed tool | icacls on the exe and folders |
| Guest locals corrupted after a host-handled fault | Windows exception dispatch writes below RSP into the guest's SysV red zone | patch guest code at load |
| Wrong data recurring with a fixed period (~2 s) | slots reused across queues without ownership | suspect reuse whenever the error is periodic |
| Caches never hit across runs | nondeterministic ids or keys (hash/pointer order) | diff the modules of two runs |
| Random host crashes; the compiler crashes too | unstable CPU cores (two P-cores here, disabled in BIOS) | crashes clustered on those cores; pin work away to test |

## Crash triage

1. Crash report must include: faulting thread name, guest registers, code bytes around pc, stack
   words, the host frames (Windows: RtlLookupFunctionEntry/RtlVirtualUnwind from the fault context;
   the walk stops at guest code), the fault page's state/protection (reserved vs committed tells
   "unmapped since" bugs apart), 24 words around each register pointer, the recent GPU write-backs onto those pages,
   and how close RSP is to a fiber/thread stack top.
2. Symbolize host frames with the linker map of that exact build (fixed base: RVA = address - base; offline,
   llvm-symbolizer needs the exe and PDB copied under their original names).
3. Classify with the table above: host access violation on guest memory, stack overflow, guest fault on a red-zone
   access (Windows exception dispatch), a game race, lock/queue deadlock (dump all thread stacks), driver/device loss
   (reduce to the shader/pipeline).
4. Reproduce with the save + route (teleport tour) that triggered it before claiming a fix; then soak.
5. Keep a crash census: grep every run log for the crash marker after each tour; crashes outrank fps.
