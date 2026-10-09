// Copyright (c) 2026 KytyPS5 contributors.
// Derived from Jetsku's KYTY_FUNCTION_ARRAY_SHRINK change, c6d2d8bf3f426ea32f1f5610302075de5bfff527.
// SPDX-License-Identifier: GPL-2.0-only
#include "graphics/host_gpu/spirvLocalArrays.h"

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::SpirvLocalArrays {
namespace {

constexpr uint32_t Magic = 0x07230203u;
std::atomic<uint32_t> g_max_subgroup_size {128};

struct Range {
	uint64_t lo = 0;
	uint64_t hi = UINT32_MAX;
};
constexpr Range Unknown {0, UINT32_MAX};

struct Instruction {
	size_t   offset = 0;
	uint16_t opcode = 0;
	uint16_t count = 0;
};

class Analysis {
public:
	explicit Analysis(std::span<const uint32_t> words):
	    m_words(words), m_max_subgroup_size(g_max_subgroup_size.load(std::memory_order_relaxed)) {}

	bool Parse() {
		if (m_words.size() < 5 || m_words[0] != Magic || m_words[3] == 0 || m_words[4] != 0) {
			return false;
		}
		size_t offset = 5;
		bool in_function = false;
		while (offset < m_words.size()) {
			const uint32_t first = m_words[offset];
			const auto count = static_cast<uint16_t>(first >> 16u);
			const auto op = static_cast<uint16_t>(first & 0xffffu);
			if (count == 0 || count > m_words.size() - offset) {
				return false;
			}
			const Instruction inst {offset, op, count};
			// Exact shapes for declarations and operations whose operands the proof reads.
			switch (op) {
				case spv::OpTypeInt: case spv::OpTypeVector: case spv::OpTypeArray:
				case spv::OpTypePointer:
					if (count != 4) { return false; } break;
				case spv::OpTypeFloat:
					if (count != 3) { return false; } break;
				case spv::OpIAdd: case spv::OpISub: case spv::OpShiftLeftLogical: case spv::OpBitwiseAnd:
					if (count != 5) { return false; } break;
				case spv::OpVariable:
					if (count != 4 && count != 5) { return false; } break;
				case spv::OpLoad:
					if (count < 4) { return false; } break;
				case spv::OpStore:
					if (count < 3) { return false; } break;
				case spv::OpAccessChain: case spv::OpInBoundsAccessChain:
					if (count < 5) { return false; } break;
				case spv::OpFunction:
					if (count != 5 || in_function) { return false; }
					in_function = true; break;
				case spv::OpFunctionEnd:
					if (count != 1 || !in_function) { return false; }
					in_function = false; break;
				default: break;
			}
			bool has_result = false;
			bool has_type = false;
			spv::HasResultAndType(static_cast<spv::Op>(op), &has_result, &has_type);
			if (has_result) {
				const uint32_t result_word = has_type ? 2u : 1u;
				if (result_word >= count) {
					return false;
				}
				const uint32_t id = Word(inst, result_word);
				if (id == 0 || id >= m_words[3] || !m_defs.emplace(id, inst).second) {
					return false;
				}
			}
			if (op == spv::OpDecorate && count >= 4 && Word(inst, 2) == spv::DecorationBuiltIn) {
				if (count != 4 || !m_builtins.emplace(Word(inst, 1), Word(inst, 3)).second) { return false; }
			}
			if (op == spv::OpName && count >= 3) {
				const auto* chars = reinterpret_cast<const char*>(&m_words[offset + 2]);
				const size_t max_chars = (count - 2u) * sizeof(uint32_t);
				size_t length = 0;
				while (length < max_chars && chars[length] != '\0') { ++length; }
				m_names[Word(inst, 1)] = std::string(chars, length);
			}
			m_insts.push_back(inst);
			offset += count;
		}
		return !in_function;
	}
	const std::vector<Instruction>& Instructions() const { return m_insts; }
	uint32_t Word(const Instruction& i, uint32_t n) const { return m_words[i.offset + n]; }
	const Instruction* Def(uint32_t id) const {
		const auto it = m_defs.find(id);
		return it == m_defs.end() ? nullptr : &it->second;
	}
	std::string Name(uint32_t id) const {
		const auto it = m_names.find(id);
		return it == m_names.end() ? std::string {} : it->second;
	}
	bool IsUnsigned32(uint32_t type_id) const {
		const auto* type = Def(type_id);
		return type != nullptr && type->opcode == spv::OpTypeInt && type->count == 4 &&
		       Word(*type, 2) == 32 && Word(*type, 3) == 0;
	}
	bool Constant(uint32_t id, uint32_t& value) const {
		const auto* def = Def(id);
		if (def == nullptr || def->opcode != spv::OpConstant || def->count != 4 ||
		    !IsUnsigned32(Word(*def, 1))) {
			return false;
		}
		value = Word(*def, 3);
		return true;
	}
	uint64_t TypeBytes(uint32_t id, int depth = 0) const {
		const auto* def = Def(id);
		if (def == nullptr || depth > 8) { return 0; }
		switch (def->opcode) {
			case spv::OpTypeInt:
				return def->count == 4 && Word(*def, 2) % 8u == 0 ? Word(*def, 2) / 8u : 0;
			case spv::OpTypeFloat:
				return def->count == 3 && Word(*def, 2) % 8u == 0 ? Word(*def, 2) / 8u : 0;
			case spv::OpTypeArray: {
				uint32_t length = 0;
				if (def->count != 4 || !Constant(Word(*def, 3), length)) { return 0; }
				const auto elem = TypeBytes(Word(*def, 2), depth + 1);
				return length == 0 || elem > UINT64_MAX / length ? 0 : elem * length;
			}
			case spv::OpTypeVector: {
				if (def->count != 4) { return 0; }
				const auto elem = TypeBytes(Word(*def, 2), depth + 1);
				const auto n = Word(*def, 3);
				return n == 0 || elem > UINT64_MAX / n ? 0 : elem * n;
			}
			default: return 0;
		}
	}
	Range ValueRange(uint32_t id, int depth = 0) {
		if (const auto it = m_ranges.find(id); it != m_ranges.end()) { return it->second; }
		if (depth > 64) { return Unknown; }
		if (!m_visiting.insert(id).second) { m_invalid_range_graph = true; return Unknown; }
		const Range result = Compute(id, depth);
		m_visiting.erase(id);
		m_ranges[id] = result;
		return result;
	}
	bool InvalidRangeGraph() const { return m_invalid_range_graph; }
	uint32_t ValueType(uint32_t id) const {
		const auto* def = Def(id);
		if (def == nullptr) { return 0; }
		bool has_result = false, has_type = false;
		spv::HasResultAndType(static_cast<spv::Op>(def->opcode), &has_result, &has_type);
		return has_result && has_type && def->count >= 3 ? Word(*def, 1) : 0;
	}
	bool IsBuiltinLoad(uint32_t value_id, uint32_t builtin) const {
		const auto* load = Def(value_id);
		if (load == nullptr || load->opcode != spv::OpLoad || load->count < 4) { return false; }
		const auto found = m_builtins.find(Word(*load, 3));
		const auto* variable = Def(Word(*load, 3));
		if (found == m_builtins.end() || found->second != builtin || variable == nullptr ||
		    variable->opcode != spv::OpVariable || variable->count != 4 ||
		    Word(*variable, 3) != spv::StorageClassInput) { return false; }
		const auto* pointer = Def(Word(*variable, 1));
		return pointer != nullptr && pointer->opcode == spv::OpTypePointer && pointer->count == 4 &&
		       Word(*pointer, 2) == spv::StorageClassInput && Word(*pointer, 3) == Word(*load, 1);
	}
private:
	Range Compute(uint32_t id, int depth) {
		const auto* def = Def(id);
		if (def == nullptr || def->count < 3 || !IsUnsigned32(Word(*def, 1))) { return Unknown; }
		const auto op = static_cast<spv::Op>(def->opcode);
		uint32_t c = 0;
		if (op == spv::OpConstant) { return Constant(id, c) ? Range {c, c} : Unknown; }
		if (op == spv::OpLoad && IsBuiltinLoad(id, spv::BuiltInSubgroupLocalInvocationId)) {
			return {0, m_max_subgroup_size - 1u};
		}
		// Unsupported SSA operations (especially Phi) are leaves: do not interpret their operands
		// as binary arithmetic. Only unsigned operands of the same type enter this narrow proof.
		if (op != spv::OpIAdd && op != spv::OpISub && op != spv::OpShiftLeftLogical &&
		    op != spv::OpBitwiseAnd) { return Unknown; }
		if (def->count != 5 || ValueType(Word(*def, 3)) != Word(*def, 1) ||
		    ValueType(Word(*def, 4)) != Word(*def, 1)) { return Unknown; }
		const Range a = ValueRange(Word(*def, 3), depth + 1);
		const Range b = ValueRange(Word(*def, 4), depth + 1);
		switch (op) {
			case spv::OpIAdd:
				return a.hi + b.hi <= UINT32_MAX ? Range {a.lo + b.lo, a.hi + b.hi} : Unknown;
			case spv::OpISub:
				return a.lo >= b.hi ? Range {a.lo - b.hi, a.hi - b.lo} : Unknown;
			case spv::OpShiftLeftLogical:
				if (b.hi >= 32 || (static_cast<uint64_t>(a.hi) << b.hi) > UINT32_MAX) { return Unknown; }
				return {a.lo << b.lo, a.hi << b.hi};
			case spv::OpBitwiseAnd:
				return {0, std::min(a.hi, b.hi)};
			default: return Unknown;
		}
	}
	std::span<const uint32_t> m_words;
	uint32_t m_max_subgroup_size;
	bool m_invalid_range_graph = false;
	std::vector<Instruction> m_insts;
	std::unordered_map<uint32_t, Instruction> m_defs;
	std::unordered_map<uint32_t, uint32_t> m_builtins;
	std::unordered_map<uint32_t, std::string> m_names;
	std::unordered_map<uint32_t, Range> m_ranges;
	std::unordered_set<uint32_t> m_visiting;
};

bool MetadataOnly(uint16_t op) {
	switch (op) {
		case spv::OpSourceContinued: case spv::OpSource: case spv::OpSourceExtension:
		case spv::OpName: case spv::OpMemberName: case spv::OpString: case spv::OpLine:
		case spv::OpNoLine: case spv::OpModuleProcessed: case spv::OpDecorate:
		case spv::OpMemberDecorate: case spv::OpDecorateId: case spv::OpDecorationGroup:
		case spv::OpGroupDecorate: case spv::OpGroupMemberDecorate: case spv::OpExtension:
		case spv::OpExtInstImport: case spv::OpMemoryModel: case spv::OpEntryPoint:
		case spv::OpExecutionMode: case spv::OpExecutionModeId: case spv::OpCapability:
		case spv::OpTypeVoid: case spv::OpTypeBool: case spv::OpTypeInt: case spv::OpTypeFloat:
		case spv::OpTypeVector: case spv::OpTypeMatrix: case spv::OpTypeImage: case spv::OpTypeSampler:
		case spv::OpTypeSampledImage: case spv::OpTypeArray: case spv::OpTypeRuntimeArray:
		case spv::OpTypeStruct: case spv::OpTypePointer: case spv::OpTypeFunction:
		case spv::OpConstantTrue: case spv::OpConstantFalse: case spv::OpConstant:
		case spv::OpConstantComposite: case spv::OpConstantNull: case spv::OpSpecConstantTrue:
		case spv::OpSpecConstantFalse: case spv::OpSpecConstant: case spv::OpSpecConstantComposite:
		case spv::OpSpecConstantOp: case spv::OpLabel: case spv::OpBranch:
		case spv::OpSelectionMerge: case spv::OpLoopMerge: case spv::OpSwitch:
		case spv::OpFunction: case spv::OpFunctionEnd: return true;
		default: return false;
	}
}

bool IsAccessChain(const Instruction& inst) {
	return inst.opcode == spv::OpAccessChain || inst.opcode == spv::OpInBoundsAccessChain;
}

} // namespace

void SetMaxSubgroupSize(uint32_t size) noexcept {
	if (size >= 1 && size <= 128) { g_max_subgroup_size.store(size, std::memory_order_relaxed); }
}

Result Shrink(std::span<const uint32_t> module, std::vector<uint32_t>& out) {
	Result result;
	out.clear();
	Analysis a(module);
	if (!a.Parse()) { return result; }
	struct Candidate {
		const Instruction* variable = nullptr;
		uint32_t array = 0, element = 0, length = 0, length_type = 0;
		uint64_t max_index = 0;
		bool safe = true, used = false;
	};
	std::unordered_map<uint32_t, Candidate> candidates;
	for (const auto& inst: a.Instructions()) {
		if (inst.opcode != spv::OpVariable || a.Word(inst, 3) != spv::StorageClassFunction) { continue; }
		const auto* pointer = a.Def(a.Word(inst, 1));
		if (pointer == nullptr || pointer->opcode != spv::OpTypePointer || pointer->count != 4 ||
		    a.Word(*pointer, 2) != spv::StorageClassFunction) { continue; }
		const auto* array = a.Def(a.Word(*pointer, 3));
		uint32_t length = 0;
		if (array == nullptr || array->opcode != spv::OpTypeArray || array->count != 4 ||
		    !a.Constant(a.Word(*array, 3), length) || length == 0) { continue; }
		const auto* length_def = a.Def(a.Word(*array, 3));
		Candidate c;
		c.variable = &inst; c.array = a.Word(*pointer, 3); c.element = a.Word(*array, 2);
		c.length = length; c.length_type = a.Word(*length_def, 1);
		c.safe = inst.count == 4; // an initializer still has the original aggregate type
		result.bytes_before += a.TypeBytes(c.array);
		candidates.emplace(a.Word(inst, 2), c);
	}
	if (candidates.empty()) { return result; }
	// Each variable can only feed a one-index access chain. The derived pointer may only be loaded
	// from or stored through. Word-level scanning deliberately treats unknown instructions as uses.
	for (const auto& inst: a.Instructions()) {
		if (MetadataOnly(inst.opcode)) { continue; }
		for (uint32_t w = 1; w < inst.count; ++w) {
			const uint32_t id = a.Word(inst, w);
			auto it = candidates.find(id);
			if (it == candidates.end()) { continue; }
			if (inst.offset == it->second.variable->offset && w == 2) { continue; } // own result
			if (!IsAccessChain(inst) || w != 3 || inst.count != 5) { it->second.safe = false; }
		}
		if (IsAccessChain(inst) && inst.count == 5) {
			const uint32_t base = a.Word(inst, 3);
			if (candidates.contains(base)) {
				bool has_result = false, has_type = false;
				spv::HasResultAndType(static_cast<spv::Op>(inst.opcode), &has_result, &has_type);
				if (!has_result || !has_type) { candidates[base].safe = false; continue; }
				const uint32_t pointer_id = a.Word(inst, 2);
				for (const auto& use: a.Instructions()) {
					if (MetadataOnly(use.opcode)) { continue; }
					for (uint32_t w = 1; w < use.count; ++w) {
						if (a.Word(use, w) != pointer_id) { continue; }
						if (use.offset == inst.offset && w == 2) { continue; } // result definition
						const bool load_pointer = use.opcode == spv::OpLoad && w == 3;
						const bool store_pointer = use.opcode == spv::OpStore && w == 1;
						if (!load_pointer && !store_pointer) { candidates[base].safe = false; }
					}
				}
				const Range range = a.ValueRange(a.Word(inst, 4));
				if (range.hi >= candidates[base].length) { candidates[base].safe = false; }
				else { candidates[base].used = true; candidates[base].max_index = std::max(candidates[base].max_index, range.hi); }
			}
		}
	}
	if (a.InvalidRangeGraph()) { return {}; } // cyclic arithmetic is not a valid SSA proof
	uint32_t bound = module[3];
	std::unordered_map<size_t, std::vector<uint32_t>> insert_after;
	std::unordered_map<uint32_t, uint32_t> new_type_of;
	for (auto& [id, c]: candidates) {
		if (!c.safe) { ++result.unbounded_arrays; result.bytes_after += a.TypeBytes(c.array); continue; }
		const uint32_t new_length = c.used ? static_cast<uint32_t>(c.max_index + 1u) : 1u;
		const uint64_t element_bytes = a.TypeBytes(c.element);
		if (new_length >= c.length || bound > UINT32_MAX - 3u) { result.bytes_after += a.TypeBytes(c.array); continue; }
		const uint32_t constant_id = bound++, array_id = bound++, pointer_id = bound++;
		const auto* pointer = a.Def(a.Word(*c.variable, 1));
		auto& words = insert_after[pointer->offset];
		words.insert(words.end(), {(4u << 16u) | spv::OpConstant, c.length_type, constant_id, new_length,
		                           (4u << 16u) | spv::OpTypeArray, array_id, c.element, constant_id,
		                           (4u << 16u) | spv::OpTypePointer, pointer_id, spv::StorageClassFunction, array_id});
		new_type_of[id] = pointer_id;
		result.arrays.push_back({id, a.Name(id), c.length, new_length, static_cast<uint32_t>(element_bytes)});
		result.bytes_after += element_bytes * new_length;
	}
	if (new_type_of.empty()) { return result; }
	out.reserve(module.size() + insert_after.size() * 12u);
	out.insert(out.end(), module.begin(), module.begin() + 5);
	out[3] = bound;
	for (const auto& inst: a.Instructions()) {
		const auto begin = module.begin() + inst.offset;
		out.insert(out.end(), begin, begin + inst.count);
		if (inst.opcode == spv::OpVariable) {
			const auto found = new_type_of.find(a.Word(inst, 2));
			if (found != new_type_of.end()) { out[out.size() - inst.count + 1u] = found->second; }
		}
		if (const auto extra = insert_after.find(inst.offset); extra != insert_after.end()) {
			out.insert(out.end(), extra->second.begin(), extra->second.end());
		}
	}
	result.changed = true;
	return result;
}

} // namespace Libs::Graphics::SpirvLocalArrays
