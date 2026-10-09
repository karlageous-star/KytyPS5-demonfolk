// The static shader and pipeline precompile, a program of its own (tools/local/static-precompile):
//
//   kyty_shader_precompile --game <dir> --seeds <file> [--shard <i>/<n>] [--threads <n>]
//                          [--out <file> | --static-inputs] [--timings <file>] [--no-pipelines]
//   kyty_shader_precompile --game <dir> --merge [--prune]
//   kyty_shader_precompile --game <dir> --seeds <file> --status
//
// compiles the shaders and pipelines of a seed file on a headless Vulkan device (the one the emulator
// creates, without a window) into the static pipeline cache _PipelineCache/static/<title>.bin, which
// the emulator looks up before compiling. With --shard, the i-th of n shares goes into a cache file of
// its own next to it: the NVIDIA driver compiles big shaders nearly one at a time per process, so the
// shares are processes (precompile-windows.ps1 runs them); --merge folds their files into the static
// cache. Where the driver has VK_KHR_pipeline_binary the static cache is the pipelines' binaries instead
// (_PipelineCache/static/<title>.binaries, read by the emulator when it needs a pipeline): every share
// writes its pipelines' binaries to a shard file, and --merge --prune makes the store of the shards' alone
// (what no seed makes any more is dropped; without --prune the store keeps its pipelines too); exit code
// 3: binaries were left out (the driver began compressing with its own dictionary), to be made by two
// shards (2n) instead. --out
// writes what was compiled as a warmup file, for precompile.py coverage; --static-inputs
// writes it where the emulator's shader prefetch reads it (_PipelineCache/static/<title>.shaders);
// --timings each pipeline's compile time (ms, SPIR-V words, the seeds' hashes). --status prints
// whether those inputs and the static cache are this GPU's and driver's ("inputs current|stale",
// "static cache current|stale"; run-windows.ps1 asks before every launch). Run it from the directory
// the emulator runs in.
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "loader/systemContent.h"
#include "static-seeds.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <thread>

using Libs::Graphics::PipelineCache;

static int Usage() {
	std::fprintf(stderr, "usage: kyty_shader_precompile --game <dir> --seeds <file> [--shard <i>/<n>] "
	                     "[--threads <n>] [--out <file> | --static-inputs] [--timings <file>] [--no-pipelines]\n"
	                     "       kyty_shader_precompile --game <dir> --merge [--prune]\n"
	                     "       kyty_shader_precompile --game <dir> --seeds <file> --status\n"
	                     "       kyty_shader_precompile --game <dir> --make-seeds <file> [--states <file>]\n");
	return 2;
}

int main(int argc, char* argv[]) {
	std::filesystem::path          game;
	bool                           merge  = false;
	bool                           prune  = false;
	bool                           status = false;
	PipelineCache::PrecompileOptions options;
	StaticSeeds::Options make_seeds;
	options.threads = std::max(1u, std::thread::hardware_concurrency());
	for (int i = 1; i < argc; i++) {
		const std::string_view arg   = argv[i];
		const char*            value = i + 1 < argc ? argv[i + 1] : nullptr;
		if (arg == "--merge") {
			merge = true;
		} else if (arg == "--prune") {
			prune = true;
		} else if (arg == "--status") {
			status = true;
		} else if (arg == "--no-pipelines") {
			options.pipelines = false;
		} else if (arg == "--static-inputs") {
			options.static_inputs = true;
		} else if (value == nullptr) {
			return Usage();
		} else if (arg == "--game") {
			game = argv[++i];
		} else if (arg == "--seeds") {
			options.seeds = argv[++i];
		} else if (arg == "--out") {
			options.out = argv[++i];
		} else if (arg == "--make-seeds") {
			make_seeds.out = argv[++i];
		} else if (arg == "--states") {
			make_seeds.states = argv[++i];
		} else if (arg == "--timings") {
			options.timings = argv[++i];
		} else if (arg == "--threads") {
			options.threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
		} else if (arg == "--shard") {
			if (std::sscanf(argv[++i], "%u/%u", &options.shard, &options.shards) != 2) return Usage();
		} else {
			return Usage();
		}
	}
	const bool generating = !make_seeds.out.empty();
	if (game.empty() || (!generating && !merge && options.seeds.empty()) ||
	    (generating && (merge || status || !options.seeds.empty())) ||
	    (!generating && !make_seeds.states.empty()) || options.threads == 0 || options.shards == 0 ||
	    options.shard >= options.shards)
		return Usage();

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction = Config::OutputDirection::Silent;
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	// The title the caches are named after.
	const auto param_json = game / "sce_sys" / "param.json";
	if (!std::filesystem::is_regular_file(param_json)) {
		std::fprintf(stderr, "no %s\n", param_json.string().c_str());
		return 1;
	}
	Loader::SystemContentLoadParamSfo(param_json);

	static Libs::Graphics::GraphicContext graphics;
	if (!Libs::Graphics::CreateHeadlessGraphicContext(graphics)) {
		std::fprintf(stderr, "no Vulkan device\n");
		return 1;
	}
	if (generating) {
		make_seeds.game = game;
		if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &make_seeds.title_id) || make_seeds.title_id.empty()) {
			std::fprintf(stderr, "no title ID in param.json\n");
			std::_Exit(1);
		}
		make_seeds.host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
		make_seeds.lod_stats_subgroup = graphics.fragment_subgroup_reduction ? 1u : 0u;
		std::printf("Seed inputs: %s, host compute subgroup %u, fragment subgroup reduction %u\n",
		            make_seeds.title_id.c_str(), make_seeds.host_subgroup_size, make_seeds.lod_stats_subgroup);
		const int code = StaticSeeds::Make(make_seeds);
		std::fflush(nullptr);
		std::_Exit(code);
	}
	if (status) {
		std::printf("inputs %s\nstatic cache %s\n",
		            PipelineCache::StaticInputsCurrent(graphics, options.seeds) ? "current" : "stale",
		            PipelineCache::StaticCacheCurrent(graphics) ? "current" : "stale");
		std::fflush(nullptr);
		std::_Exit(0);
	}
	size_t     left_out = 0;
	const bool done     = merge ? PipelineCache::MergePrecompileShards(graphics, prune)
	                            : PipelineCache::Precompile(graphics, options, left_out);
	std::printf("Shader precompile: %s\n", !done ? "failed" : left_out != 0 ? "partial" : "complete");
	std::fflush(nullptr);
	// 3: pipelines' binaries were left out (PipelineBinaryWriter): run this shard as two.
	std::_Exit(!done ? 1 : left_out != 0 ? 3 : 0);
}
