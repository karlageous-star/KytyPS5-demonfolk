#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct MaskedSelectStats {
	uint32_t collapsed = 0; // select(E, select(E, a, b), c) and select(E, a, select(E, b, c)) collapsed
	uint32_t removed   = 0; // selects on E only observed where E holds: replaced by their true operand
	uint32_t forwarded = 0; // operands select(E, a, b) of a value only observed where E holds: read as a
};

// Every vector register write merges the new value with the old one under exec (Translator::WriteOperand), so inside
// a divergent region each write is a select on the same exec value E, and the next instruction reads that select.
// A lane-local value (pure arithmetic, no memory access, no other lane's value) whose every use is the true operand
// of a select on E, or another such value, is only ever observed in the lanes where E holds: there a select(E, a, b)
// it reads is a, so it reads a. A select on E observed only where E holds is its true operand. What is left is one
// select per register where a region's values leave it. Memory, image, LDS and cross-lane operations keep their
// operands, so no lane that exec turned off reads or writes other memory than before.
MaskedSelectStats ForwardMaskedSelects(const BlockList& blocks);

} // namespace Libs::Graphics::ShaderRecompiler::IR
