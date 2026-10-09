// Copyright (c) 2026 KytyPS5 contributors.
// Derived from Jetsku's KYTY_FUNCTION_ARRAY_SHRINK change, c6d2d8bf3f426ea32f1f5610302075de5bfff527.
// SPDX-License-Identifier: GPL-2.0-only
#include "graphics/host_gpu/spirvLocalArrays.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {
using Libs::Graphics::SpirvLocalArrays::Result;
using Libs::Graphics::SpirvLocalArrays::SetMaxSubgroupSize;
using Libs::Graphics::SpirvLocalArrays::Shrink;
int failures = 0;

void Expect(bool condition, const char* message) {
	if (!condition) { std::printf("SpirvLocalArraysTests: failed: %s\n", message); ++failures; }
}

std::string ComputeModule(const std::string& body, const std::string& initializer = {},
                          const std::string& declarations = {}, const std::string& extra_types = {}) {
	// Debug instructions precede annotations. Int64 is declared for the width regressions.
	// Both inputs are valid compute builtins; WorkgroupId is unknown to the reducer.
	return R"(
OpCapability Shader
OpCapability GroupNonUniform
OpCapability Int64
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %lane_var %workgroup_var
OpExecutionMode %main LocalSize 32 1 1
OpName %array "function_array"
OpDecorate %lane_var BuiltIn SubgroupLocalInvocationId
OpDecorate %workgroup_var BuiltIn WorkgroupId
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%s64 = OpTypeInt 64 1
%u64 = OpTypeInt 64 0
%uvec3 = OpTypeVector %u32 3
%zero = OpConstant %u32 0
%one = OpConstant %u32 1
%mask31 = OpConstant %u32 31
%shift32 = OpConstant %u32 32
%maxu = OpConstant %u32 4294967295
%highbit = OpConstant %u32 2147483648
%signedzero = OpConstant %s32 0
%longzero = OpConstant %s64 0
%ulongzero = OpConstant %u64 0
%length = OpConstant %u32 8192
%array_type = OpTypeArray %u32 %length
%null_array = OpConstantNull %array_type
%ptr_array = OpTypePointer Function %array_type
%ptr_u32 = OpTypePointer Function %u32
%ptr_input = OpTypePointer Input %u32
%ptr_input_vec = OpTypePointer Input %uvec3
)" + extra_types + R"(
%lane_var = OpVariable %ptr_input Input
%workgroup_var = OpVariable %ptr_input_vec Input
%main = OpFunction %void None %fn
%entry = OpLabel
%array = OpVariable %ptr_array Function )" + initializer + "\n" + declarations + R"(
%lane = OpLoad %u32 %lane_var
)" + body + R"(
OpReturn
OpFunctionEnd
)";
}

std::string Access(const std::string& index) {
	return "%element = OpAccessChain %ptr_u32 %array " + index + R"(
OpStore %element %one
%value = OpLoad %u32 %element
)";
}

std::vector<uint32_t> Assemble(const std::string& source) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& pos, const char* msg) {
		std::printf("assembler line %zu: %s\n", pos.line, msg);
	});
	std::vector<uint32_t> binary;
	Expect(tools.Assemble(source, &binary, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS), "assembly succeeds");
	return binary;
}

bool Validate(const std::vector<uint32_t>& binary, bool diagnostics = true) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	if (diagnostics) {
		tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* msg) {
			std::printf("validator: %s\n", msg);
		});
	}
	return tools.Validate(binary);
}

std::vector<uint32_t> ValidModule(const std::string& source) {
	auto module = Assemble(source);
	const bool valid = Validate(module);
	Expect(valid, "valid regression input passes Vulkan 1.3 validation");
	return valid ? module : std::vector<uint32_t> {};
}

// Validation would not detect silently changing an access to a different element. Compare entire
// instructions so the pointer/index IDs, stored values and loaded types/results must stay identical.
std::vector<uint32_t> MemoryInstructions(const std::vector<uint32_t>& module) {
	std::vector<uint32_t> instructions;
	for (size_t offset = 5; offset < module.size();) {
		const uint32_t count = module[offset] >> 16u;
		if (count == 0 || count > module.size() - offset) { return {}; }
		const uint32_t op = module[offset] & 0xffffu;
		if (op == spv::OpLoad || op == spv::OpStore || op == spv::OpAccessChain || op == spv::OpInBoundsAccessChain) {
			instructions.insert(instructions.end(), module.begin() + offset, module.begin() + offset + count);
		}
		offset += count;
	}
	return instructions;
}

void CheckShrink(const std::vector<uint32_t>& module, uint32_t length) {
	if (module.empty()) { return; }
	std::vector<uint32_t> out;
	const Result result = Shrink(module, out);
	Expect(result.changed && result.arrays.size() == 1, "exactly one array shrinks");
	if (!result.changed || result.arrays.size() != 1) { return; }
	Expect(Validate(out), "reduced output passes Vulkan 1.3 validation");
	Expect(result.arrays[0].old_length == 8192 && result.arrays[0].new_length == length, "proven array length");
	Expect(result.arrays[0].element_bytes == 4, "OpTypeInt occupies four bytes");
	Expect(result.arrays[0].name == "function_array", "name retained");
	Expect(result.bytes_before == 8192u * 4u && result.bytes_after == length * 4u, "byte statistics match lengths");
	Expect(MemoryInstructions(module) == MemoryInstructions(out), "all load/store/index operands remain identical");
}

void CheckUnchanged(std::span<const uint32_t> module, const char* reason) {
	std::vector<uint32_t> out {0xdeadbeefu};
	const Result result = Shrink(module, out);
	Expect(!result.changed && result.arrays.empty() && out.empty(), reason);
}

void TestBoundsAndMemoryOperands() {
	SetMaxSubgroupSize(32);
	const auto lane = ValidModule(ComputeModule(Access("%lane")));
	CheckShrink(lane, 32);
	SetMaxSubgroupSize(0);
	SetMaxSubgroupSize(129);
	CheckShrink(lane, 32);
	SetMaxSubgroupSize(128);
	CheckShrink(lane, 128);
	CheckShrink(ValidModule(ComputeModule(Access("%zero"))), 1);
	CheckShrink(ValidModule(ComputeModule("%masked = OpBitwiseAnd %u32 %lane %mask31\n" + Access("%masked"))), 32);
	CheckShrink(ValidModule(ComputeModule(R"(
%sum = OpIAdd %u32 %lane %one
%index = OpISub %u32 %sum %one
)" + Access("%index"))), 128);
	CheckShrink(ValidModule(ComputeModule("%shifted = OpShiftLeftLogical %u32 %lane %one\n" + Access("%shifted"))), 255);
	// Both store and load at different indices must participate in the maximum bound.
	CheckShrink(ValidModule(ComputeModule(R"(
%upper = OpIAdd %u32 %lane %one
%write_pointer = OpAccessChain %ptr_u32 %array %upper
OpStore %write_pointer %lane
%read_pointer = OpInBoundsAccessChain %ptr_u32 %array %zero
%loaded = OpLoad %u32 %read_pointer
)")), 129);
}

void TestUnknownArithmeticAndWidths() {
	for (const auto& expression: {
	         "%index = OpIAdd %u32 %lane %maxu\n",
	         "%index = OpISub %u32 %lane %one\n",
	         "%index = OpShiftLeftLogical %u32 %one %shift32\n",
	         "%index = OpShiftLeftLogical %u32 %highbit %one\n"}) {
		const auto module = ValidModule(ComputeModule(expression + Access("%index")));
		if (!module.empty()) { CheckUnchanged(module, "wrap, underflow or shift >=32 leaves original array"); }
	}
	for (const char* index: {"%signedzero", "%longzero", "%ulongzero"}) {
		const auto module = ValidModule(ComputeModule(Access(index)));
		if (!module.empty()) { CheckUnchanged(module, "signed/64-bit constants do not prove bounds"); }
	}
	const auto unbounded = ValidModule(ComputeModule(R"(
%group = OpLoad %uvec3 %workgroup_var
%index = OpCompositeExtract %u32 %group 0
)" + Access("%index")));
	if (!unbounded.empty()) { CheckUnchanged(unbounded, "unbounded compute builtin leaves original array"); }
}

void TestInitializersAndEscapes() {
	const auto initialized = ValidModule(ComputeModule(Access("%lane"), "%null_array"));
	if (!initialized.empty()) { CheckUnchanged(initialized, "array initializer retains original aggregate type"); }
	const auto whole = ValidModule(ComputeModule(Access("%lane") + R"(
OpStore %array %null_array
%whole = OpLoad %array_type %array
)"));
	if (!whole.empty()) { CheckUnchanged(whole, "whole-array store/load prevents shrinking"); }
	const auto copied_element = ValidModule(ComputeModule(R"(
%element = OpAccessChain %ptr_u32 %array %lane
%alias = OpCopyObject %ptr_u32 %element
OpStore %alias %one
%value = OpLoad %u32 %alias
)"));
	if (!copied_element.empty()) { CheckUnchanged(copied_element, "element pointer alias prevents shrinking"); }
	const auto copied_array = ValidModule(ComputeModule(R"(
%alias = OpCopyObject %ptr_array %array
%element = OpAccessChain %ptr_u32 %alias %lane
OpStore %element %one
%value = OpLoad %u32 %element
)"));
	if (!copied_array.empty()) { CheckUnchanged(copied_array, "array pointer alias prevents shrinking"); }
}

void TestPhiCycle() {
	const auto module = ValidModule(ComputeModule(R"(
OpBranch %header
%header = OpLabel
%index = OpPhi %u32 %zero %entry %next %continue_block
%condition = OpULessThan %bool %index %length
OpLoopMerge %merge %continue_block None
OpBranchConditional %condition %body %merge
%body = OpLabel
%element = OpAccessChain %ptr_u32 %array %index
OpStore %element %one
%value = OpLoad %u32 %element
OpBranch %continue_block
%continue_block = OpLabel
%next = OpIAdd %u32 %index %one
OpBranch %header
%merge = OpLabel
)"));
	if (!module.empty()) { CheckUnchanged(module, "valid cyclic Phi loop remains unknown"); }
}

void ParserSafetyOnly(const std::string& source, const char* reason) {
	// Explicitly illegal semantic fixtures: safe fallback, not valid Vulkan use.
	const auto module = Assemble(source);
	if (module.empty()) { return; }
	Expect(!Validate(module, false), "parser-safety fixture is intentionally invalid");
	CheckUnchanged(module, reason);
}

void TestIllegalPointerInitializersAndArithmeticCycle() {
	ParserSafetyOnly(ComputeModule(Access("%zero"), {},
	                 "%escape = OpVariable %ptr_array_pointer Function %array\n",
	                 "%ptr_array_pointer = OpTypePointer Function %ptr_array\n"),
	                 "another variable's initializer referencing the array is scanned");
	ParserSafetyOnly(ComputeModule(Access("%zero"), {},
	                 "%escape = OpVariable %ptr_element_pointer Function %element\n",
	                 "%ptr_element_pointer = OpTypePointer Function %ptr_u32\n"),
	                 "another variable's initializer referencing an element pointer is scanned");
	// Function OpPtrAccessChain is illegal in Vulkan. Its extra Element operand must not be
	// confused with the array index of OpAccessChain, even when the binary parses safely.
	ParserSafetyOnly(ComputeModule(R"(
%element = OpPtrAccessChain %ptr_u32 %array %zero %lane
OpStore %element %one
%value = OpLoad %u32 %element
)"), "Function OpPtrAccessChain leaves the original module");
	ParserSafetyOnly(ComputeModule("%cycle = OpBitwiseAnd %u32 %cycle %mask31\n" + Access("%cycle")),
	                 "malformed arithmetic cycle cannot become a masked proof");
}

void TestMalformedModules() {
	const auto module = ValidModule(ComputeModule(Access("%lane")));
	if (module.empty()) { return; }
	// Includes truncation at instruction boundaries inside a function. Guard all fixture indexing.
	for (size_t size = 0; size < module.size(); ++size) {
		CheckUnchanged(std::span<const uint32_t>(module.data(), size), "truncated module is rejected safely");
	}
	auto oversized = module;
	oversized[5] = 0xffff0000u | (oversized[5] & 0xffffu);
	CheckUnchanged(oversized, "instruction count beyond module is rejected");
	auto zero_count = module;
	zero_count[5] &= 0xffffu;
	CheckUnchanged(zero_count, "zero instruction count is rejected");
	// Duplicate the result ID of two genuine constants without changing any word counts.
	auto duplicate = module;
	uint32_t first_id = 0;
	bool mutated = false;
	for (size_t offset = 5; offset < duplicate.size();) {
		const uint32_t count = duplicate[offset] >> 16u;
		if (count == 0 || count > duplicate.size() - offset) { break; }
		if ((duplicate[offset] & 0xffffu) == spv::OpConstant && count >= 3) {
			if (first_id == 0) { first_id = duplicate[offset + 2]; }
			else { duplicate[offset + 2] = first_id; mutated = true; break; }
		}
		offset += count;
	}
	Expect(mutated, "duplicate-ID fixture constructed");
	Expect(!Validate(duplicate, false), "duplicate IDs are invalid SPIR-V");
	CheckUnchanged(duplicate, "duplicate result ID prevents all rewriting");
}
} // namespace

int main() {
	TestBoundsAndMemoryOperands();
	TestUnknownArithmeticAndWidths();
	TestInitializersAndEscapes();
	TestPhiCycle();
	TestIllegalPointerInitializersAndArithmeticCycle();
	TestMalformedModules();
	SetMaxSubgroupSize(128);
	if (failures != 0) { std::printf("%d failure(s)\n", failures); return 1; }
	std::puts("spirv_local_arrays: all tests passed");
	return 0;
}
