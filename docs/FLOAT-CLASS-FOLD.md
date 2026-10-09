# Floating-point class-mask IR fold

`ConstantPropagationPass` folds only exact `FPCmpClass32` masks `0`, `3`, and
`0x3ff`. The SPIR-V emitter in `spirvEmitterAluHelpers.cpp` tests mask bits 0
through 9 and does not inspect higher bits. Therefore `0` matches no class,
`3` selects signaling and quiet NaNs, and `0x3ff` selects every class.

The class emitter identifies NaNs from an all-ones exponent and nonzero
mantissa, then separates signaling from quiet NaNs using the quiet bit. This
includes either sign and every nonzero NaN payload, regardless of the quiet
bit. `FPIsNan32` emits `OpFUnordNotEqual(x, x)`, which recognizes both NaN
classes. Thus replacing exact mask `3` with `FPIsNan32` preserves the emitter's
classification for signaling and quiet NaNs, including their signs and
payloads. No floating-point constant evaluation is performed by this fold.

High mask bits are semantically ignored by the emitter, but the fold requires
an exact mask so `0x80000003` and all other unrecognized masks remain
`FPCmpClass32`. Dynamic masks also remain unchanged. The standalone structural
regression source is `tests/FloatClassFoldTests.cpp`; it covers the three exact
masks, an unrecognized mask, a high-bit mask, and a dynamic mask.

## Test target dependencies

The test source does not need emulator or GPU code. A CMake target can compile
it with these IR implementation files:

* `src/graphics/shader/recompiler/ir/Block.cpp`
* `src/graphics/shader/recompiler/ir/Value.cpp`
* `src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp`
* `src/graphics/shader/recompiler/ir/passes/ConstantPropagation.cpp`

It also needs the project's usual include directories and link dependencies
and generated `cmake_config.h`. Compile this small standalone target with
`KYTY_FINAL` defined so `EXIT_IF` does not pull in the emulator's assertion
runtime. The test source supplies the one IR runtime hook it needs,
`ValidationEnabled()`, returning false. No emulator, GPU, or fmt link target is
needed. No CMake change is included here.
