# Function array shrink reducer

This optional sidecar reducer is derived from Jetsku's `KYTY_FUNCTION_ARRAY_SHRINK` change
(`c6d2d8bf3f426ea32f1f5610302075de5bfff527`) and is distributed under the repository's
GPL-2.0 terms. It rewrites only the declared type of a Function-storage array when it
can establish a conservative upper bound for every element pointer produced from that variable.
The caller must validate the original module before attempting a rewrite, validate every changed
output even when normal shader validation is disabled, and fall back to the original module if
validation fails. A successful range calculation is not a substitute for SPIR-V validation or a
semantic proof of the complete module.

The initial port deliberately accepts only unsigned 32-bit constants, loads of
`SubgroupLocalInvocationId` (bounded by the configured maximum, default 128), and checked unsigned
`OpIAdd`, `OpISub`, `OpShiftLeftLogical`, and `OpBitwiseAnd` expressions. Add and shift results that
could wrap are unknown. Subtraction is accepted only when its minimum minuend is at least the
maximum subtrahend. Shift amounts must be below 32; both operands must have the same unsigned
32-bit type as the result. The subgroup maximum is captured once per analysis and remains 128 by
default. Signed values, 64-bit constants, other builtins,
phi/select propagation, and all unrecognized operations produce an unknown range. Unknown means the
array remains at its original size.

For a candidate, every direct variable reference must be the base of a one-index access chain. Each
derived element pointer may only be the pointer operand of a load or store. This rejects whole-array
operations, pointer copies, pointer arithmetic, calls, and unknown pointer uses. Only a candidate's
own declaration result word is ignored: other variable declarations and their initializers are
scanned for both array and element pointer references. The maximum index
must be strictly less than the declared length. Scanning is intentionally conservative: operand
words in unknown instructions are treated as possible uses, so some safe modules may be declined.

The parser bounds-checks every instruction before reading operands and rejects duplicate or
out-of-bound result IDs, malformed analyzed instruction shapes, unterminated functions, and cycles
in the recognized arithmetic graph. Unsupported Phi values remain unknown leaves. Integer byte
sizes use the four-word OpTypeInt shape; floats use the three-word OpTypeFloat shape. It is a narrow
transformer, not a general SPIR-V validator. Function array
initializers, nested indexing, aliases, and pointer escape are not transformed. No optional zero or
poison initialization is performed, and the reducer itself is never default-on.

The tests validate every normal input and every reduced output with SpirvTools for Vulkan 1.3.
Compute fixtures use SubgroupLocalInvocationId and an unknown WorkgroupId input, with debug names
before decorations and the Int64 capability for wide constants. Regressions cover byte statistics,
both store and load bounds, unchanged memory/index operands, subgroup maxima, checked arithmetic,
wrap/underflow, shifts at least 32, signed/64-bit constants, whole-array use, array initialization,
pointer aliases, valid Phi cycles, duplicate IDs, and every incomplete prefix of a valid module.
Illegal pointer initializers, Function OpPtrAccessChain, and direct arithmetic cycles are explicitly
marked parser-safety fixtures and must fail validation before checking conservative fallback.

`KYTY_FUNCTION_ARRAY_SHRINK=1` applies this pass to recompiled guest shader modules in
ProgramCache::CreateModule. The original and reduced modules both undergo Vulkan 1.3 validation
even when ordinary shader validation is disabled. Failed validation uses the original module;
recorded warmup SPIR-V remains unchanged. The conservative subgroup bound is 128. Declared byte
counts in the log describe shader declarations, not measured VRAM savings. The option defaults off.

The standalone `spirv_local_arrays_tests` built and passed on October 9, 2026. Game qualification
is separate: require same-scene visual checks, repeated ASTRO boots beyond the prior device-loss
transition, and DS gameplay before adopting a player profile. First use may compile new driver
pipelines because their module bytes differ.
