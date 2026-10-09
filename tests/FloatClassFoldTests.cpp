// Copyright (c) 2026 KytyPS5 contributors.
// SPDX-License-Identifier: GPL-2.0-only
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"

#include <bit>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics::ShaderRecompiler::IR {
// Program.cpp owns the production setting; this standalone pass test disables validation.
bool ValidationEnabled() {
	return false;
}
} // namespace Libs::Graphics::ShaderRecompiler::IR

namespace {
using namespace Libs::Graphics::ShaderRecompiler::IR;

int failures = 0;

void Expect(bool condition, const char* message) {
	if (!condition) {
		std::printf("FloatClassFoldTests: failed: %s\n", message);
		failures++;
	}
}

Value FoldedResult(Block& block, Value input, Value mask) {
	const Value compare(&block.AppendNewInst(ValueOpcode::FPCmpClass32, {input, mask}));
	auto& reference = block.AppendNewInst(ValueOpcode::Reference, {compare});
	const BlockList blocks {&block};
	ConstantPropagationPass(blocks);
	return reference.Arg(0).Resolve();
}

void TestExactMasks() {
	{
		Block block;
		Expect(FoldedResult(block, Value::F32(1.0f), Value(0u)) == Value(false),
		       "zero class mask folds to false");
	}
	{
		for (const uint32_t bits: {0x7f812345u, 0xff812345u, 0x7fc12345u, 0xffc12345u}) {
			Block block;
			const auto input  = Value::F32(std::bit_cast<float>(bits));
			const auto result = FoldedResult(block, input, Value(3u));
			auto* is_nan      = result.TryInstruction();
			Expect(is_nan != nullptr && is_nan->GetOpcode() == ValueOpcode::FPIsNan32,
			       "signaling and quiet NaN mask lowers to FPIsNan32");
			if (is_nan != nullptr && is_nan->GetOpcode() == ValueOpcode::FPIsNan32) {
				const auto folded_input = is_nan->Arg(0).Resolve();
				Expect(folded_input.IsImmediate() && folded_input.GetType() == Type::F32 &&
				           std::bit_cast<uint32_t>(folded_input.F32Value()) == bits,
				       "NaN sign, quiet bit, and payload are preserved");
			}
		}
	}
	{
		Block block;
		Expect(FoldedResult(block, Value::F32(1.0f), Value(0x3ffu)) == Value(true),
		       "all ten class bits fold to true");
	}
}

void TestUnsupportedMasksRemainClassComparisons() {
	for (const auto mask: {0x400u, 0x80000003u}) {
		Block block;
		const Value input = Value::F32(2.0f);
		const Value compare(&block.AppendNewInst(ValueOpcode::FPCmpClass32,
		                                         {input, Value(mask)}));
		auto& reference = block.AppendNewInst(ValueOpcode::Reference, {compare});
		const BlockList blocks {&block};
		ConstantPropagationPass(blocks);
		const auto result = reference.Arg(0).Resolve();
		auto* retained = result.TryInstruction();
		Expect(retained != nullptr && retained->GetOpcode() == ValueOpcode::FPCmpClass32,
		       "unrecognized or high-bit mask remains unchanged");
	}
}

void TestDynamicMaskRemainsClassComparison() {
	Block block;
	const Value dynamic_mask(&block.AppendNewInst(ValueOpcode::GetBuiltin,
	                                             {Value(0u), Value(0u)}));
	const Value input = Value::F32(2.0f);
	const Value compare(&block.AppendNewInst(ValueOpcode::FPCmpClass32,
	                                         {input, dynamic_mask}));
	auto& reference = block.AppendNewInst(ValueOpcode::Reference, {compare});
	const BlockList blocks {&block};
	ConstantPropagationPass(blocks);
	const auto result = reference.Arg(0).Resolve();
	auto* retained = result.TryInstruction();
	Expect(retained != nullptr && retained->GetOpcode() == ValueOpcode::FPCmpClass32,
	       "dynamic mask remains unchanged");
}
} // namespace

int main() {
	TestExactMasks();
	TestUnsupportedMasksRemainClassComparisons();
	TestDynamicMaskRemainsClassComparison();
	return failures == 0 ? 0 : 1;
}
