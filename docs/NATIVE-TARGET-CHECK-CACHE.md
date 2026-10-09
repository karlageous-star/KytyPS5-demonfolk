# Optional native XPR target-check memo

Adapted from chenxiao `3ee9be77e99bc1788ef050cacca541e844112d87`.
`KYTY_NATIVE_TARGET_CHECK_CACHE=1` enables the memo; default is off.
`=2` compares would-be memo hits against the complete original check, which
remains authoritative and terminates verification on disagreement.

Only draw states captured with every target registered and backed by nonempty
guest data qualify. Epoch zero never qualifies. The only target-slot insert is
TextureCache::InsertImage, which registers nonempty images. DeleteImage
unregisters before retiring a slot, including deferred erasure. Registration
and unregistration increment ResolutionEpoch. Image serial and backing identity
are immutable for a surviving slot. Replacing a draw state constructs a new
state and resets its proof. Failed complete checks clear the memo.

The memo applies only to NativeXprTargetsCurrent. NativeXprEmitTargets still
checks live slot/serial, registration, needs_rebind, SafeToDownload, metadata
clears and current layouts on every emission. GPU ownership/acquisition is not
cached. The existing color/stencil acquisition guard is unchanged; this does
not fix the unresolved ownership crash. Data-less/unregistered targets use the
original check rather than relying on an epoch that does not cover them.

The first delegated draft repeated serial lookups on memo hits and was revised
before build because that retained the cost the upstream change removes.

## Qualification recipe

1. Clean build; ASTRO splash/menu and DS gameplay with mode 2, native/table
   graphics enabled and compute table off. Require zero verifier disagreements.
2. Exercise target creation/deletion, resolution changes and transitions.
   Finite runtime comparison does not prove arbitrary streaming races.
3. Warm same-route mode 0/1 frame-time and visual comparisons. Player profiles
   keep this off until both titles qualify.
4. Roll back by removing the switch or setting 0. No cache-schema change.

No local FPS gain is established by this source change.
