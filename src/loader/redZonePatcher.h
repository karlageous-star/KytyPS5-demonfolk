// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_RED_ZONE_PATCHER_H_
#define KYTY_LOADER_RED_ZONE_PATCHER_H_

#include "common/common.h"

#include <span>
#include <vector>

namespace Loader {

struct RedZonePatchResult {
	uint64_t function_count                           = 0;
	uint64_t instruction_count                        = 0;
	uint64_t red_zone_function_count                  = 0;
	uint64_t memory_instruction_count                 = 0;
	uint64_t short_memory_instruction_count           = 0;
	uint64_t patched_memory_instruction_count         = 0;
	uint64_t stack_dependent_memory_instruction_count = 0;
	uint64_t control_flow_memory_instruction_count    = 0;
	uint64_t unrelocatable_memory_instruction_count   = 0;
	uint64_t indirect_red_zone_function_count         = 0;
	uint64_t swept_instruction_count                  = 0;
	uint64_t live_call_count                          = 0;
	// Ordinary guest direct calls the liveness analysis marked live, left as they are: a call
	// writes its return address below RSP on the console too, and lowering RSP around one
	// moves its stack arguments (7th and later) away from where the callee reads them.
	uint64_t native_call_count                        = 0;
};

void RegisterRedZonePatchModule(void* module_ptr, uint64_t module_size, void* trampoline_area_ptr,
                                uint64_t trampoline_area_size);
void UnregisterRedZonePatchModule(void* module_ptr);
// Guest code addresses of red-zone memory instructions no patch protects (a fault there overwrites the
// guest's red zone with the exception frame): the exception handler reports faults at them.
bool IsUnprotectedRedZoneSite(uint64_t address);

// [synthetic_call_begin, synthetic_call_end): what the loader's own direct calls target (the TLS
// patch's `call <handler>`, which can sit where red-zone data is live). Only those calls are moved
// below the red zone.
RedZonePatchResult PatchRedZoneMemoryInstructions(uint64_t segment_addr, uint64_t segment_size,
                                                  std::span<const uintptr_t> function_starts,
                                                  uint64_t synthetic_call_begin = 0,
                                                  uint64_t synthetic_call_end   = 0);

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

} // namespace Loader

#endif /* KYTY_LOADER_RED_ZONE_PATCHER_H_ */
