#include "graphics/shader/recompiler/ir/passes/LaneBranches.h"

#include "graphics/shader/recompiler/ir/Value.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool IsSelect(ValueOpcode opcode) {
	return opcode == ValueOpcode::SelectU1 || opcode == ValueOpcode::SelectU32 || opcode == ValueOpcode::SelectF32;
}

// An operation a region cannot keep under a per-lane branch: it reads other lanes, acts once for the wave, waits for
// the workgroup, or changes memory a number of times the active lanes decide (atomics). Register accesses are gone
// after SSA; refused all the same. (AnyLane and Ballot of a condition that implies E, phis below the region's entry
// and the conditions of branches inside it are checked apart.)
bool RefusedInRegion(ValueOpcode opcode, bool compute_derivatives) {
	switch (opcode) {
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32:
		case ValueOpcode::WqmU64:
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane:
		case ValueOpcode::WriteLane:
		case ValueOpcode::Permlane16U32:
		case ValueOpcode::BpermuteU32:
		case ValueOpcode::SwizzleU32:
		case ValueOpcode::Barrier:
		case ValueOpcode::Sendmsg:
		case ValueOpcode::ImageQueryLod:
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

// Whether a lane condition holds only where E holds: E itself, or a conjunction with such a conjunct (exec narrowed
// inside the region: s_and_saveexec, v_cmpx).
bool Implies(const Inst* condition, const Inst* exec, int depth = 0) {
	if (condition == nullptr || depth > 32) return false;
	if (condition == exec) return true;
	if (condition->GetOpcode() != ValueOpcode::LogicalAnd) return false;
	return Implies(InstOf(condition->Arg(0)), exec, depth + 1) || Implies(InstOf(condition->Arg(1)), exec, depth + 1);
}

// What a lane outside E read at the merge when the whole wave ran the region (the uniform branch): true when it is
// `skip` on every path. A value is: skip itself; a select whose old value is (its condition implying E), or both of
// whose values are; a phi inside the region all of whose values are (loop phis assumed so while their loop is checked).
struct Region {
	const Inst*                          exec = nullptr;
	std::unordered_set<const Block*>     blocks;
	std::unordered_map<const Inst*, int> memo; // 0 being checked (assumed true), 1 true, 2 false
};

bool SameWhenNotE(Region& region, const Value& value, const Value& skip, int depth = 0) {
	const Value resolved = value.Resolve();
	if (resolved == skip) return true;
	const Inst* inst = resolved.TryInstruction();
	// Lane conditions both false outside E (an exec narrowed inside the region against the exec at the branch).
	if (Implies(inst, region.exec) && Implies(skip.TryInstruction(), region.exec)) return true;
	if (inst == nullptr || depth > 256 || !region.blocks.contains(inst->Parent())) return false;
	if (const auto found = region.memo.find(inst); found != region.memo.end()) return found->second != 2;
	region.memo[inst] = 0;
	bool result = false;
	if (IsSelect(inst->GetOpcode())) {
		result = (Implies(InstOf(inst->Arg(0)), region.exec) && SameWhenNotE(region, inst->Arg(2), skip, depth + 1)) ||
		         (SameWhenNotE(region, inst->Arg(1), skip, depth + 1) &&
		          SameWhenNotE(region, inst->Arg(2), skip, depth + 1));
	} else if (inst->GetOpcode() == ValueOpcode::Phi) {
		result = true;
		for (size_t operand = 0; operand < inst->NumArgs() && result; operand++)
			result = SameWhenNotE(region, inst->Arg(operand), skip, depth + 1);
	}
	region.memo[inst] = result ? 1 : 2;
	return result;
}

constexpr size_t MaxRegionBlocks = 1024;

} // namespace

LaneBranchStats ConvertLaneBranches(Program& program) {
	LaneBranchStats stats;
	if (program.blocks.size() != program.block_info.size()) return stats;
	// KYTY_LANE_DEBUG=<hash>: why each exec branch of that program converts or not.
	static const uint64_t debug_hash = [] {
		const char* value = std::getenv("KYTY_LANE_DEBUG");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : uint64_t {0};
	}();
	const bool debug = debug_hash != 0 && program.shader_hash == debug_hash;
	bool compute_derivatives = false;
	std::unordered_map<const Block*, size_t> block_index;
	for (size_t i = 0; i < program.blocks.size(); i++) {
		block_index[program.blocks[i]] = i;
		for (const auto& inst: program.blocks[i]->Instructions())
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
		if (skip_id != term.merge_block || region_id == term.merge_block) {
			if (debug) std::printf("LANE_DEBUG block %u: the skip edge is not the merge\n", info.id);
			continue;
		}
		const int r = BlockIndex(program, region_id);
		const int m = BlockIndex(program, term.merge_block);
		if (r < 0 || m < 0 || static_cast<size_t>(r) == p) continue;
		Block* branch = program.blocks[p];
		Block* entry  = program.blocks[r];
		Block* merge  = program.blocks[m];
		// The region: every block reached from its entry before the merge. One entry (from the branch), no way back
		// to the branch, no return (s_endpgm ends the whole wave), and the merge entered only from it or the branch.
		Region             region {.exec = exec_inst};
		std::vector<Block*> order {entry};
		region.blocks.insert(entry);
		const char* shape = nullptr;
		for (size_t next = 0; next < order.size() && shape == nullptr; next++) {
			const Block* block = order[next];
			const auto&  block_term = program.block_info[block_index[block]].terminator;
			if (block_term.kind != CFG::TerminatorKind::Branch &&
			    block_term.kind != CFG::TerminatorKind::ConditionalBranch) {
				shape = "a return or an indirect branch inside";
				break;
			}
			for (Block* successor: block->ImmSuccessors()) {
				if (successor == merge) continue;
				if (successor == branch) {
					shape = "a way back to the branch";
					break;
				}
				if (region.blocks.insert(successor).second) order.push_back(successor);
			}
			if (order.size() > MaxRegionBlocks) shape = "too many blocks";
		}
		if (shape == nullptr) {
			if (entry->ImmPredecessors().size() != 1 || entry->ImmPredecessors()[0] != branch) shape = "a second entry";
			for (size_t i = 1; i < order.size() && shape == nullptr; i++) {
				for (const Block* predecessor: order[i]->ImmPredecessors())
					if (!region.blocks.contains(predecessor)) shape = "a block entered from outside";
			}
			bool from_branch = false, from_region = false;
			for (const Block* predecessor: merge->ImmPredecessors()) {
				from_branch = from_branch || predecessor == branch;
				from_region = from_region || region.blocks.contains(predecessor);
				if (predecessor != branch && !region.blocks.contains(predecessor) && shape == nullptr)
					shape = "a merge entered from outside";
			}
			if (shape == nullptr && (!from_branch || !from_region)) shape = "a merge not entered from both";
		}
		if (shape != nullptr) {
			if (debug) std::printf("LANE_DEBUG block %u: region %u: %s\n", info.id, region_id, shape);
			continue;
		}
		stats.candidates++;
		const char* refusal = nullptr;
		for (const Block* block: order) {
			for (const auto& inst: block->Instructions()) {
				const auto opcode = inst.GetOpcode();
				if (opcode == ValueOpcode::Phi) {
					if (block == entry) refusal = "a phi at the entry";
				} else if (opcode == ValueOpcode::AnyLane || opcode == ValueOpcode::Ballot) {
					// Over the active lanes the same as over the wave, when the lanes outside E give false.
					if (!Implies(InstOf(inst.Arg(0)), exec_inst)) refusal = "AnyLane or Ballot of a condition beyond E";
				} else if (opcode == ValueOpcode::Reference) {
					// A branch inside: on a scalar value or AnyLane (checked above): uniform over the active lanes.
				} else if (RefusedInRegion(opcode, compute_derivatives)) {
					refusal = ValueOpcodeName(opcode).data();
				}
				if (refusal != nullptr) break;
			}
			if (refusal != nullptr) break;
		}
		if (refusal != nullptr) {
			if (debug) std::printf("LANE_DEBUG block %u: region %u (%zu blocks) has %s\n", info.id, region_id, order.size(),
			                       refusal);
			stats.refused_op++;
			continue;
		}
		// What the region hands the merge, per phi: for a lane outside E what the skipping edge carries.
		bool refused = false;
		for (const auto& inst: merge->Instructions()) {
			if (inst.GetOpcode() != ValueOpcode::Phi) continue;
			Value skip;
			bool  have_skip = false;
			for (size_t operand = 0; operand < inst.NumArgs(); operand++) {
				if (inst.PhiBlock(operand) == branch) {
					skip      = inst.Arg(operand).Resolve();
					have_skip = true;
				}
			}
			if (!have_skip) {
				refused = true;
				break;
			}
			for (size_t operand = 0; operand < inst.NumArgs() && !refused; operand++) {
				if (inst.PhiBlock(operand) == branch) continue;
				refused = !SameWhenNotE(region, inst.Arg(operand), skip);
				if (refused && debug) {
					const Inst* value = InstOf(inst.Arg(operand));
					std::printf("LANE_DEBUG block %u: region %u (%zu blocks): a merge phi%s gets %s\n", info.id, region_id,
					            order.size(), inst.Flags<Type>() == Type::U1 ? " (U1)" : "",
					            value != nullptr ? std::string(ValueOpcodeName(value->GetOpcode())).c_str() : "a constant");
				}
			}
			if (refused) break;
		}
		if (refused) {
			stats.refused_phi++;
			continue;
		}
		for (Block* block: order) {
			for (auto& inst: block->Instructions()) {
				if (IsSelect(inst.GetOpcode()) && InstOf(inst.Arg(0)) == exec_inst) {
					inst.ReplaceUsesWith(inst.Arg(1).Resolve());
					stats.selects++;
				}
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
		if (debug) std::printf("LANE_DEBUG block %u: converted (region %u, %zu blocks)\n", info.id, region_id, order.size());
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
