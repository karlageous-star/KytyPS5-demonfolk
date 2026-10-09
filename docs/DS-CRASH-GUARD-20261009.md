# Demon's Souls crash mitigation, 2026-10-09

## Evidence

The supplied gameplay log is from emulator `2f947473`, game PPSA01342
01.004.000. A BPE worker faults at `eboot.bin+0xe762d5` while writing
through a null previous-list link (`[rax+0x50]`, RAX=0). The preceding loads
read the endpoint object's temporary-list links at +0x50 and +0x58; both
are zero. R14 is R15+0x20, the object's end endpoint. Offline disassembly
shows a spatial/collision sweep using a temporary active list. This is not
evidence that the whole job should be discarded or that a GPU readback
caused the invalid state. The recent writeback history has no writes onto
the pages selected by the dump.

## Bounded mitigation

`KYTY_DS_DETACHED_UNLINK_GUARD=1` enables a loader patch for this version
only. It is off by default in source. The candidate player enables it for
PPSA01342 only. Installation requires an exact 1,110-byte routine
fingerprint (FNV-1a64 `dcf5496bb1a60ba5`) and the two expected loads at
0xe762cd. Different metadata, patched code, or an allocation failure leave
the original code intact. No on-disk game file is changed.

The trampoline repeats those loads and uses the existing continuation
at 0xe762b7 only if **both links are null and this is the end endpoint**.
A valid tail (next=null, previous!=null), a linked member, and other
malformed states retain the original path. No worker job or draw is
skipped. GPRs beyond the original RAX/RCX outputs, flags, SIMD state,
stack pointer, and the guest red zone are preserved; no host callback is
made. The code allocation lives until process exit. Existing unload and
exit guards prevent freeing a trampoline that another guest thread could
be executing.

The render log reports `Demon's Souls collision guard: detached_unlinks=N`
when the counter changes. An installed guard with a zero counter proves
only that this invalid state was not encountered in that run.

## Limits and rollback

This prevents the specific observed null write. It does **not** establish
or repair the reason the endpoint was not linked. Possible lost collision
pairs or incorrect endpoint ordering still require gameplay checks. Do
not label it a proven crash fix, improved FPS, or stable 30 FPS.

Set `KYTY_DS_DETACHED_UNLINK_GUARD` to `0` in the candidate player's
`launcher-environment.json` and restart to restore original behavior.
The old `KytyPS5 DS Test 2f947473` player remains the executable rollback.
The mitigation is isolated to its source commit and can be reverted.

## Verification

`demons_souls_collision_guard_tests` covers generated trampoline behavior,
unchanged normal/tail removals, exact detached state, malformed cases,
register/flags/SIMD/stack preservation and relative-jump bounds. Offline
validation against the user's original SELF confirms the routine
fingerprint. Build and gameplay results are recorded separately in
`Performance Experiments/C1-job-list-crash` so source documentation does
not assert results before execution.

Further cause investigation should compare the original path and guard
path at the same save: log guard counts, inspect endpoint values/order
and writes to +0x50/+0x58, check combat/collision behavior, and isolate
write-readback/handoff/GPU-buffer-page switches independently. Fiber
floating-point control state is a separate audit finding; it is not
bundled as an unproven fix for this crash.

## Other supplied crash

PPSA28997 fails in `Il2CppUserAssemblies.prx+0x3179bb1` while storing a
byte through a stack local containing a Windows DLL address. Its output
pointer local is above RSP at the fault. The red-zone patcher's one
unrelocatable site does not, by itself, identify the cause. This title is
not covered by the Demon's Souls mitigation. Offline analysis continues;
no fault swallowing or writes into executable Windows pages are allowed.
