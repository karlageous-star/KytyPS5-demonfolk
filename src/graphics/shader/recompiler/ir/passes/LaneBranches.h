#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct LaneBranchStats {
	uint32_t candidates   = 0; // branches on AnyLane(E) around a one-block region that falls into the merge
	uint32_t converted    = 0; // ... now branching on each lane's E
	uint32_t selects      = 0; // selects on E those regions no longer need
	uint32_t refused_op   = 0; // a region with a cross-lane, wave, barrier or atomic operation
	uint32_t refused_phi  = 0; // a region handing the merge a value other than a select on E (a scalar register)
};

// KYTY_LANE_BRANCHES: a guest region entered when any lane's exec holds (s_cbranch_execz around it) runs for the whole
// wave with every vector write a select on its exec E. On a native wave64 compute shader the branch can test each
// lane's E instead (the host masks the lanes): inside, the selects on E are their new values, and the merge's phis
// keep the old ones for the lanes that skip. Only where nothing in the region looks at other lanes or acts once for
// the wave, and every value the region hands the merge is a vector register's select on E.
LaneBranchStats ConvertLaneBranches(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
