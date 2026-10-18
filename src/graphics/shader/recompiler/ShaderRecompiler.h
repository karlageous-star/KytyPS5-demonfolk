#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint32_t                    scratch_dwords  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	bool                        enable_lod_stats           = false;
	bool                        table_mode                 = false; // IR::EnterTableMode
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
	// KYTY_TRANSLATE_STATS: the guest instructions, and the indexed register accesses among them (V_MOVRELS /
	// V_MOVRELD: each becomes a select over every register from its base to the highest one the program uses).
	uint32_t stat_instructions = 0;
	uint32_t stat_movrels       = 0;
	uint32_t stat_movreld       = 0;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

// Storage buffer accesses without an explicit bounds check where the device checks them the same
// way: robustBufferAccess2 with byte-exact ranges (robustStorageBufferAccessSizeAlignment 1) reads
// zero out of bounds and drops the stores. Set with the device, before any program compiles.
void SetDeviceStorageBufferBounds(bool enabled);

// Storage buffer loads without a branch around them where the device's bounds are not byte-exact
// (AMD: 4-byte) but robust buffer access keeps every load in memory: the load runs for every lane, and a
// select gives zero outside the shader's own bounds check, the same value. Set with the device.
void SetSpeculativeBufferLoads(bool enabled);

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
