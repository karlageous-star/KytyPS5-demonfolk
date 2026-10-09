// Copyright (c) 2026 KytyPS5 contributors.
// Derived from Jetsku's KYTY_FUNCTION_ARRAY_SHRINK change, c6d2d8bf3f426ea32f1f5610302075de5bfff527.
// SPDX-License-Identifier: GPL-2.0-only
#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SPIRVLOCALARRAYS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SPIRVLOCALARRAYS_H_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::SpirvLocalArrays {

struct Shrunk {
	uint32_t    variable = 0;
	std::string name;
	uint32_t    old_length = 0;
	uint32_t    new_length = 0;
	uint32_t    element_bytes = 0; // type-derived size; 0 if unknown
};

struct Result {
	bool                changed = false;
	std::vector<Shrunk> arrays;
	uint32_t            unbounded_arrays = 0;
	uint64_t            bytes_before = 0;
	uint64_t            bytes_after = 0;
};

// Maximum subgroup size used to bound SubgroupLocalInvocationId (default 128; valid values: 1..128).
// The caller must supply a bound covering every invocation of the rewritten module.
void SetMaxSubgroupSize(uint32_t size) noexcept;

// Requires an input module validated for the caller's target environment. Writes a rewritten
// module only when changed. Validate every changed output and fall back to the original on failure.
// Validation establishes structural legality, not the range proof. Input and output must not alias.
Result Shrink(std::span<const uint32_t> module, std::vector<uint32_t>& out);

} // namespace Libs::Graphics::SpirvLocalArrays

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_SPIRVLOCALARRAYS_H_
