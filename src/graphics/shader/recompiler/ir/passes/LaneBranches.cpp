#include "graphics/shader/recompiler/ir/passes/LaneBranches.h"

#include "graphics/shader/recompiler/ir/Value.h"

#include <string_view>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool IsSelect(ValueOpcode opcode) {
	return opcode == ValueOpcode::SelectU1 || opcode == ValueOpcode::SelectU32 || opcode == ValueOpcode::SelectF32;
}

// An operation a region cannot keep under a per-lane branch: it reads other lanes, acts once for the wave, waits for
// the workgroup, or changes memory a number of times the active lanes decide (atomics). Register accesses are gone
// after SSA; refused all the same.
bool RefusedInRegion(ValueOpcode opcode, bool compute_derivatives) {
	switch (opcode) {
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32:
		case ValueOpcode::WqmU64:
		case ValueOpcode::Ballot:
		case ValueOpcode::AnyLane:
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane:
		case ValueOpcode::WriteLane:
		case ValueOpcode::Permlane16U32:
		case ValueOpcode::BpermuteU32:
		case ValueOpcode::SwizzleU32:
		case ValueOpcode::Barrier:
		case ValueOpcode::Sendmsg:
		case ValueOpcode::ImageQueryLod:
		case ValueOpcode::Phi:
		case ValueOpcode::Reference:
		case ValueOpcode::ReferenceU32:
		case ValueOpcode::GetThreadBitScalarRegister:
		case ValueOpcode::SetThreadBitScalarRegister:
		case ValueOpcode::GetScalarMaskTag:
		case ValueOpcode::SetScalarMaskTag:
		case ValueOpcode::GetScalarRegister:
		case ValueOpcode::SetScalarRegister:
		case ValueOpcode::GetGotoVariable:
		case ValueOpcode::SetGotoVariable:
		case ValueOpcode::GetScc:
		case ValueOpcode::SetScc:
		case ValueOpcode::GetExec:
		case ValueOpcode::SetExec:
		case ValueOpcode::GetExecLo:
		case ValueOpcode::SetExecLo:
		case ValueOpcode::GetExecHi:
		case ValueOpcode::SetExecHi:
		case ValueOpcode::GetVcc:
		case ValueOpcode::SetVcc:
		case ValueOpcode::GetVccLo:
		case ValueOpcode::SetVccLo:
		case ValueOpcode::GetVccHi:
		case ValueOpcode::SetVccHi: return true;
		// Implicit derivatives (compute quads) need the whole quad.
		case ValueOpcode::ImageSampleRaw:
		case ValueOpcode::ImageGatherRaw: return compute_derivatives;
		default: break;
	}
	const std::string_view name = ValueOpcodeName(opcode);
	return name.starts_with("BufferAtomic") || name.starts_with("SharedAtomic") || name.starts_with("ImageAtomic");
}

Inst* InstOf(const Value& value) {
	return value.IsEmpty() ? nullptr : value.Resolve().TryInstruction();
}

int BlockIndex(const Program& program, uint32_t id) {
	for (size_t i = 0; i < program.block_info.size(); i++)
		if (program.block_info[i].id == id) return static_cast<int>(i);
	return -1;
}

} // namespace

LaneBranchStats ConvertLaneBranches(Program& program) {
	LaneBranchStats stats;
	if (program.blocks.size() != program.block_info.size()) return stats;
	bool compute_derivatives = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: block->Instructions())
			compute_derivatives = compute_derivatives || inst.GetOpcode() == ValueOpcode::ImageQueryLod;
	}
	for (size_t p = 0; p < program.blocks.size(); p++) {
		auto&       info = program.block_info[p];
		const auto& term = info.terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header ||
		    term.merge_block == UINT32_MAX)
			continue;
		Inst* condition = InstOf(info.condition);
		if (condition == nullptr) continue;
		// s_cbranch_execz: !AnyLane(E) skips to the merge; s_cbranch_execnz: AnyLane(E) enters the region.
		const bool negated = condition->GetOpcode() == ValueOpcode::LogicalNot;
		Inst*      any     = negated ? InstOf(condition->Arg(0)) : condition;
		if (any == nullptr || any->GetOpcode() != ValueOpcode::AnyLane) continue;
		const Value exec      = any->Arg(0).Resolve();
		const Inst* exec_inst = exec.TryInstruction();
		if (exec_inst == nullptr) continue;
		const uint32_t region_id = negated ? term.false_block : term.true_block;
		const uint32_t skip_id   = negated ? term.true_block : term.false_block;
		if (skip_id != term.merge_block || region_id == term.merge_block) continue;
		const int r = BlockIndex(program, region_id);
		const int m = BlockIndex(program, term.merge_block);
		if (r < 0 || m < 0 || static_cast<size_t>(r) == p) continue;
		Block*      branch = program.blocks[p];
		Block*      region = program.blocks[r];
		Block*      merge  = program.blocks[m];
		const auto& region_term = program.block_info[r].terminator;
		if (region_term.kind != CFG::TerminatorKind::Branch || region_term.loop_header ||
		    region_term.true_block != term.merge_block || region->ImmPredecessors().size() != 1 ||
		    region->ImmPredecessors()[0] != branch || merge->ImmPredecessors().size() != 2)
			continue;
		stats.candidates++;
		bool refused = false;
		for (const auto& inst: region->Instructions()) {
			if (RefusedInRegion(inst.GetOpcode(), compute_derivatives)) {
				refused = true;
				break;
			}
		}
		if (refused) {
			stats.refused_op++;
			continue;
		}
		// What the region hands the merge, per phi: a select on E whose old value is what the skipping edge carries
		// (a vector register the region wrote under E), or the skipping edge's own value. Then a lane outside E gets
		// at the merge what it got when the whole wave ran the region. Anything else (a scalar register the region
		// wrote, which every lane sees) keeps the uniform branch.
		for (const auto& inst: merge->Instructions()) {
			if (inst.GetOpcode() != ValueOpcode::Phi) continue;
			Value from_region, from_branch;
			bool  have_region = false, have_branch = false;
			for (size_t operand = 0; operand < inst.NumArgs(); operand++) {
				if (inst.PhiBlock(operand) == region) {
					from_region = inst.Arg(operand).Resolve();
					have_region = true;
				} else if (inst.PhiBlock(operand) == branch) {
					from_branch = inst.Arg(operand).Resolve();
					have_branch = true;
				}
			}
			if (!have_region || !have_branch) {
				refused = true;
				break;
			}
			const Inst* value = from_region.TryInstruction();
			if (value != nullptr && value->Parent() == region) {
				refused = !(IsSelect(value->GetOpcode()) && InstOf(value->Arg(0)) == exec_inst &&
				            value->Arg(2).Resolve() == from_branch);
			} else {
				refused = !(from_region == from_branch);
			}
			if (refused) break;
		}
		if (refused) {
			stats.refused_phi++;
			continue;
		}
		for (auto& inst: region->Instructions()) {
			if (IsSelect(inst.GetOpcode()) && InstOf(inst.Arg(0)) == exec_inst) {
				inst.ReplaceUsesWith(inst.Arg(1).Resolve());
				stats.selects++;
			}
		}
		// The branch on each lane's E (or its negation for the execz form), the condition its Reference keeps.
		Value lane_condition = exec;
		auto  reference      = branch->end();
		for (auto it = branch->begin(); it != branch->end(); ++it) {
			if (it->GetOpcode() == ValueOpcode::Reference && InstOf(it->Arg(0)) == condition) reference = it;
		}
		if (negated) {
			const auto inserted = branch->PrependNewInst(reference, ValueOpcode::LogicalNot, {exec});
			lane_condition      = Value(&*inserted);
		}
		if (reference != branch->end()) reference->SetArg(0, lane_condition);
		info.condition = lane_condition;
		stats.converted++;
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
