#pragma once
// kyty_shader_precompile --make-seeds (static-seeds.cpp): the static precompile's seed file, made from the
// game's files. It is tools/local/static-precompile/precompile.py seeds without Python: the same records and file
// format for the same title and host inputs.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace StaticSeeds {

struct Options {
	std::filesystem::path game;            // the folder with eboot.bin
	std::filesystem::path out;             // the seed file to write
	std::filesystem::path states;          // pass-states.json; empty: FindStates()
	std::string           stages = "cs,gfx"; // precompile.py seeds --stages: cs, gfx or both
	std::string           title_id = "PPSA01341"; // the game title ID in the seed identity line
	uint32_t              host_subgroup_size = 32;   // actual device when available; precompile.py default otherwise
	uint32_t              lod_stats_subgroup = 1;    // actual device when available; precompile.py default otherwise
	int64_t               limit  = 0;        // precompile.py seeds --limit: the first N programs only (a smoke test)
};

// pass-states.json where the repository and a release package keep it (tools/local/static-precompile below the
// program's folder, two folders above it in a build tree, or below the working directory), or empty.
std::filesystem::path FindStates();

// Writes the seed file: 0, or 1 where precompile.py stops with an error (nothing is written then).
int Make(const Options& options);

// A seed file of these records and pipeline recipes, as Make writes one (the identity line, the XXH3-64 of the body,
// the body; whole or not at all): the body's size, or 0 where it cannot be written.
uint64_t Write(const std::filesystem::path& path, std::vector<std::vector<uint32_t>> records,
               std::vector<std::vector<uint32_t>> pipelines, const std::string& title_id = "PPSA01341");

} // namespace StaticSeeds
