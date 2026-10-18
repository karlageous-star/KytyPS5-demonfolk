#include "graphics/shader/recompiler/ir/passes/MaskedSelectForwarding.h"

#include "graphics/shader/recompiler/ir/Value.h"

#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool IsSelect(ValueOpcode opcode) {
	return opcode == ValueOpcode::SelectU1 || opcode == ValueOpcode::SelectU32 || opcode == ValueOpcode::SelectF32;
}

// Pure and lane-local: a lane's result depends only on that lane's operands, with no memory access, no other lane's
// value and no fault (on the GPU an integer division by zero or an FP exception gives a value, never a trap).
bool LaneLocal(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::BitCastU16F16:
		case ValueOpcode::BitCastF16U16:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU16U32:
		case ValueOpcode::ConvertU32U16:
		case ValueOpcode::ConvertU8U32:
		case ValueOpcode::ConvertU32U8:
		case ValueOpcode::ConvertF32F16:
		case ValueOpcode::ConvertF16F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32S32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeConstructF32x2:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::PackHalf2x16:
		case ValueOpcode::PackSnorm2x16:
		case ValueOpcode::PackUnorm2x16:
		case ValueOpcode::PackFloat2x16Rtz:
		case ValueOpcode::FPAbs32:
		case ValueOpcode::FPNeg32:
		case ValueOpcode::FPSaturate32:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32:
		case ValueOpcode::SelectU32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UDiv32:
		case ValueOpcode::SMulHi:
		case ValueOpcode::UMulHi:
		case ValueOpcode::IAbs32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::BitReverse32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::BitCount64:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::FindUMsb64:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::SMin32:
		case ValueOpcode::UMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMinTri32:
		case ValueOpcode::UMinTri32:
		case ValueOpcode::SMaxTri32:
		case ValueOpcode::UMaxTri32:
		case ValueOpcode::SMedTri32:
		case ValueOpcode::UMedTri32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThan64:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThan64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThan64:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPUnordEqual32:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPUnordNotEqual32:
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPUnordLessThan32:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPUnordGreaterThan32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPUnordLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPUnordGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPFma32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32:
		case ValueOpcode::FPMedTri32:
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPRecipSqrt32:
		case ValueOpcode::FPSqrt:
		case ValueOpcode::FPSin:
		case ValueOpcode::FPCos:
		case ValueOpcode::FPExp2:
		case ValueOpcode::FPLog2:
		case ValueOpcode::FPLdexp:
		case ValueOpcode::FPRoundEven32:
		case ValueOpcode::FPFloor32:
		case ValueOpcode::FPCeil32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPFract32: return true;
		default: return false;
	}
}

// The condition instruction of a select, or null (a constant or non-instruction condition is left alone).
Inst* ConditionOf(const Inst& select) {
	return IsSelect(select.GetOpcode()) ? select.Arg(0).TryInstruction() : nullptr;
}

} // namespace

MaskedSelectStats ForwardMaskedSelects(const BlockList& blocks) {
	MaskedSelectStats  stats;
	std::vector<Inst*> order;
	for (auto* block: blocks) {
		for (auto& inst: block->Instructions()) order.push_back(&inst);
	}
	// 1. select(E, select(E, a, b), c) -> select(E, a, c); select(E, a, select(E, b, c)) -> select(E, a, c).
	for (auto* inst: order) {
		const auto* condition = ConditionOf(*inst);
		if (condition == nullptr) continue;
		for (size_t operand = 1; operand <= 2; operand++) {
			for (int depth = 0; depth < 64; depth++) {
				const auto* inner = inst->Arg(operand).Resolve().TryInstruction();
				if (inner == nullptr || ConditionOf(*inner) != condition) break;
				inst->SetArg(operand, inner->Arg(operand).Resolve());
				stats.collapsed++;
			}
		}
	}
	// 2. The condition E each lane-local value is masked by: every use is the true operand of a select on E, or an
	//    operand of a lane-local value masked by E. Uses come after definitions, so a backward walk settles most of it;
	//    loops (phis) never mask.
	std::unordered_map<const Inst*, const Inst*> masked;
	bool changed = true;
	for (int round = 0; round < 8 && changed; round++) {
		changed = false;
		for (auto it = order.rbegin(); it != order.rend(); ++it) {
			const Inst* inst = *it;
			if (!LaneLocal(inst->GetOpcode()) || !inst->HasUses()) continue;
			const Inst* condition = nullptr;
			bool        ok        = true;
			for (const auto& use: inst->Uses()) {
				const Inst* by = nullptr;
				if (use.operand == 1 && IsSelect(use.user->GetOpcode())) {
					by = ConditionOf(*use.user);
				} else if (use.operand != 0 || !IsSelect(use.user->GetOpcode())) {
					if (const auto found = masked.find(use.user); found != masked.end()) by = found->second;
				}
				if (by == nullptr || (condition != nullptr && by != condition)) {
					ok = false;
					break;
				}
				condition = by;
			}
			const auto  found   = masked.find(inst);
			const Inst* current = found != masked.end() ? found->second : nullptr;
			const Inst* next    = ok ? condition : nullptr;
			if (next != current) {
				if (next != nullptr) masked[inst] = next;
				else masked.erase(inst);
				changed = true;
			}
		}
	}
	// 3. A select on E masked by E is its true operand; a value masked by E reads a for an operand select(E, a, b).
	// KYTY_FORWARD_DEAD_ONLY=1: only when every use of that select reads a so (it then goes): a select kept for other uses
	// saves nothing, and its consumers reading a instead keep a and the select both live (two registers for one).
	static const bool dead_only = [] {
		const char* value = std::getenv("KYTY_FORWARD_DEAD_ONLY");
		return value != nullptr && value[0] == '1';
	}();
	std::unordered_map<const Inst*, bool> all_forwarded;
	const auto forwardable = [&](const Inst* select, const Inst* condition) {
		if (!dead_only) return true;
		const auto [found, inserted] = all_forwarded.try_emplace(select, true);
		if (inserted) {
			for (const auto& use: select->Uses()) {
				const auto by = masked.find(use.user);
				if (by == masked.end() || by->second != condition ||
				    (use.operand == 0 && ConditionOf(*use.user) == condition)) {
					found->second = false;
					break;
				}
			}
		}
		return found->second;
	};
	for (auto* inst: order) {
		const auto found = masked.find(inst);
		if (found == masked.end()) continue;
		const auto* condition = found->second;
		if (ConditionOf(*inst) == condition) {
			inst->ReplaceUsesWith(inst->Arg(1).Resolve());
			stats.removed++;
			continue;
		}
		for (size_t operand = 0; operand < inst->NumArgs(); operand++) {
			const auto* source = inst->Arg(operand).Resolve().TryInstruction();
			if (source != nullptr && ConditionOf(*source) == condition && forwardable(source, condition)) {
				inst->SetArg(operand, source->Arg(1).Resolve());
				stats.forwarded++;
			}
		}
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
