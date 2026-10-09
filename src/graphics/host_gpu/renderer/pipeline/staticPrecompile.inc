// Included by pipelineCache.cpp in the static precompile program only (KYTY_STATIC_PRECOMPILE).

// The specializations a seed is compiled with: every resource bound to a plain descriptor of the kind
// the shader declares (an image of the declared dimension and the numeric class its use suggests
// (PredictImageNumericClasses), a buffer without stride, and where a
// store packs a format, the one most of them use: one 32-bit integer), and with portable modules each
// indirect image table, whose size is the game's data, in turn in its plain form (one texture) and
// padded to 2, 4, 8 and 16 candidates (BuildResourceSpecialization), for programs with at most two
// tables; a vertex or pixel shader's two tables (texture sets sampled twice) with the same size.
static std::vector<ShaderRecompiler::IR::ResourceSpecialization>
SeedSpecializations(const ShaderRecompiler::IR::Program& program) {
	using ShaderRecompiler::Decoder::ImageDimension;
	using ShaderRecompiler::IR::ImageResource;
	ShaderRecompiler::IR::ResourceSpecialization plain;
	for (const auto& buffer: program.info.buffers) {
		plain.buffers.push_back(buffer.formatted ? ShaderRecompiler::IR::ResourceSpecialization::Buffer {
		                                               .descriptor_format  = Prospero::BufferFormat::k32UInt,
		                                               .descriptor_swizzle = DstSel(4, 0, 0, 0)}
		                                         : ShaderRecompiler::IR::ResourceSpecialization::Buffer {});
	}
	std::vector<uint32_t> tables; // root images of indirect tables, in image order
	const auto            classes = ShaderRecompiler::IR::PredictImageNumericClasses(program);
	for (uint32_t index = 0; index < program.info.images.size(); index++) {
		const auto& image = program.info.images[index];
		plain.images.push_back({
		    .numeric_class     = classes[index],
		    .dimension         = image.dimension == ImageDimension::Unknown ? ImageDimension::Dim2D : image.dimension,
		    .mip_count         = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle    = image.shader_swizzle,
		    .cube              = image.cube,
		});
		if (image.source < program.descriptor_sources.size() &&
		    program.descriptor_sources[image.source].indirect_image.has_value())
			tables.push_back(index);
	}
	// Compute shaders that declare 3D images (outside tables) are often given 2D textures there (most
	// specializations of recorded play the seeds missed), nearly always the last ones of them (95%:
	// 3D 2D, 2D 2D, 3D 3D 2D...): every variant again with the last 1, 2, ... of them 2D; with three of
	// them also the two that keep the last one a volume (2D 2D 3D, 3D 2D 3D: 72 of the 73 other 3D
	// misses of the 10-01 tour). All subsets would be 2^k - 1 variants each instead of k (+ 2).
	const auto with_volumes_as_2d = [&](std::vector<ShaderRecompiler::IR::ResourceSpecialization> out) {
		std::vector<uint32_t> volumes;
		for (uint32_t index = 0; index < program.info.images.size(); index++) {
			if (plain.images[index].dimension == ImageDimension::Dim3D && std::ranges::find(tables, index) == tables.end())
				volumes.push_back(index);
		}
		if (program.stage != ShaderType::Compute || volumes.empty()) return out;
		std::vector<std::vector<uint32_t>> flat_sets; // the volumes given 2D textures, per variant
		for (size_t last = 1; last <= volumes.size(); last++) flat_sets.emplace_back(volumes.end() - last, volumes.end());
		if (volumes.size() == 3) flat_sets.insert(flat_sets.end(), {{volumes[0], volumes[1]}, {volumes[1]}});
		for (size_t variant = 0, count = out.size(); variant < count; variant++) {
			for (const auto& flat_set: flat_sets) {
				auto flat = out[variant];
				for (const auto volume: flat_set) flat.images[volume].dimension = ImageDimension::Dim2D;
				out.push_back(std::move(flat));
			}
		}
		return out;
	};
	if (tables.empty() || tables.size() > 2 || !ShaderRecompiler::IR::PortableShaders()) {
		return with_volumes_as_2d({plain});
	}
	constexpr uint32_t sizes[] = {1, 2, 4, 8, 16};
	const auto         header  = static_cast<uint32_t>(program.srt_reads.size());
	std::vector<ShaderRecompiler::IR::ResourceSpecialization> out;
	const bool same_sizes = program.stage != ShaderType::Compute;
	for (uint32_t combination = 0, count = tables.size() == 1 ? 5u : 25u; combination < count; combination++) {
		if (same_sizes && tables.size() == 2 && combination % 5 != combination / 5) continue;
		auto     specialization = plain;
		uint32_t active         = 0;
		bool     fits           = true;
		for (uint32_t table = 0, digits = combination; table < tables.size(); table++, digits /= 5) {
			const auto size = sizes[digits % 5];
			const auto root = tables[table];
			if (size == 1) continue;
			if (specialization.images.size() + size - 1 > ShaderRecompiler::IR::ShaderInfo::MaxImages) {
				fits = false;
				break;
			}
			auto candidate          = specialization.images[root];
			candidate.indirect_root = root;
			specialization.images.insert(specialization.images.end(), size - 1, candidate);
			auto& image                      = specialization.images[root];
			image.indirect_root              = root;
			image.indirect_mapping_offset    = header + active++;
			image.indirect_search_iterations = ImageResource::RuntimeIndirectSearch;
		}
		if (fits) out.push_back(std::move(specialization));
	}
	// A table's class is its textures' (BuildResourceSpecialization), and the game's tables hold 2D
	// textures where the shader declares others (3D in its particle compute shaders): every variant
	// again with 2D tables.
	if (std::ranges::any_of(tables, [&](uint32_t root) { return plain.images[root].dimension != ImageDimension::Dim2D; })) {
		for (size_t variant = 0, count = out.size(); variant < count; variant++) {
			auto flat = out[variant];
			for (uint32_t index = 0; index < flat.images.size(); index++) {
				if (std::ranges::find(tables, index) != tables.end() ||
				    std::ranges::find(tables, flat.images[index].indirect_root) != tables.end())
					flat.images[index].dimension = ImageDimension::Dim2D;
			}
			out.push_back(std::move(flat));
		}
	}
	return with_volumes_as_2d(std::move(out));
}

// A recorded specialization's indirect image tables in their portable form (BuildResourceSpecialization):
// candidates padded to a power of two, the key mapping found through a header word.
static void PortableIndirectTables(const ShaderRecompiler::IR::Program&         program,
                                   ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	using ShaderRecompiler::IR::ImageResource;
	const auto base = program.info.images.size();
	if (!ShaderRecompiler::IR::PortableShaders() || specialization.images.size() <= base) return;
	std::vector<ShaderRecompiler::IR::ResourceSpecialization::Image> images(specialization.images.begin(),
	                                                                         specialization.images.begin() + base);
	uint32_t active = 0;
	for (uint32_t root = 0; root < base; root++) {
		if (images[root].indirect_root != root) continue;
		const auto count = static_cast<size_t>(std::ranges::count_if(
		    specialization.images, [&](const auto& image) { return image.indirect_root == root; }));
		auto padded = std::bit_ceil(count);
		if (images.size() + padded - 1 > ShaderRecompiler::IR::ShaderInfo::MaxImages) padded = count;
		// Candidates copy the root as the shader declares its table (not the recorded key mapping).
		auto candidate                       = images[root];
		candidate.indirect_root              = root;
		candidate.indirect_mapping_offset    = program.info.images[root].indirect_mapping_offset;
		candidate.indirect_search_iterations = program.info.images[root].indirect_search_iterations;
		images.insert(images.end(), padded - 1, candidate);
		images[root].indirect_mapping_offset    = static_cast<uint32_t>(program.srt_reads.size()) + active++;
		images[root].indirect_search_iterations = ImageResource::RuntimeIndirectSearch;
	}
	specialization.images = std::move(images);
}

// A seed file's compiler inputs (tools/local/static-precompile: derived from the game files, their
// resource specialization left open), each translated and specialized (SeedSpecializations) with
// portable formats, the module the game gets when it meets the program (ProgramCache::Get); a pixel
// shader before the vertex shader it is drawn with (its push data comes first). Their pipelines go
// into the static pipeline cache, or a shard's (every n-th compute seed and graphics recipe) into a
// cache of its own next to it. What was compiled can be written as a warmup file (args.out), to
// compare with what the game compiles.
bool PipelineCache::WarmSeeds(const PrecompileOptions& args, size_t& left_out) {
	std::vector<std::vector<uint32_t>> seed_words, recipe_words;
	if (!LocalShaderWarmup::Cache::ReadSeeds(args.seeds, seed_words, recipe_words)) {
		PipelineCacheLog("Shader seeds: cannot read {}", Common::PathToString(args.seeds));
		return false;
	}
	const auto shard = args.shard, shards = args.shards, threads = args.threads;
	PipelineCacheLog("Shader seeds: {} programs and {} graphics pipelines from {} (shard {}/{})", seed_words.size(),
	                 recipe_words.size(), Common::PathToString(args.seeds), shard, shards);
	const bool create_pipelines = args.pipelines;
	const auto begin            = std::chrono::steady_clock::now();
	const auto elapsed          = [&] {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
	};
	auto&      programs = *m_program_cache;
	const auto decode   = [&](uint32_t index, LocalShaderWarmup::Record& record) {
		LocalShaderWarmup::Reader reader {seed_words[index]};
		return LocalShaderWarmup::Visit(reader, record);
	};
	const auto stage_of = [&](size_t index) { return ShaderType(seed_words[index][0]); };
	// Graphics recipes; every compute seed gets its pipelines anyway.
	std::vector<LocalShaderWarmup::PipelineRecord> recipes;
	for (const auto& words: recipe_words) {
		LocalShaderWarmup::PipelineRecord recipe;
		LocalShaderWarmup::Reader         reader {words};
		if (LocalShaderWarmup::VisitPipeline(reader, recipe) && recipe.compute == LocalShaderWarmup::NoShader)
			recipes.push_back(recipe);
	}
	// This shard's work: every n-th compute seed and graphics recipe, and the programs they need.
	const auto        mine = [&](size_t n) { return n % shards == shard; };
	std::vector<bool> wanted(seed_words.size());
	for (size_t i = 0; i < seed_words.size(); i++) {
		const auto stage = stage_of(i);
		wanted[i]        = (stage == ShaderType::Compute && mine(i)) || (stage == ShaderType::Pixel && shards == 1);
	}
	for (size_t r = 0; r < recipes.size(); r++) {
		if (mine(seed_words.size() + r) && recipes[r].pixel != LocalShaderWarmup::NoShader) wanted[recipes[r].pixel] = true;
	}
	// Runs job(n) for n below count on `threads` workers (this thread one of them).
	const auto parallel = [&](size_t count, const auto& job) {
		std::atomic<size_t> next {0};
		const auto          work = [&] {
			for (size_t n; (n = next.fetch_add(1, std::memory_order_relaxed)) < count;) job(n);
		};
		std::vector<std::jthread> pool;
		for (uint32_t t = 1; t < std::min<size_t>(threads, count); ++t) pool.emplace_back(work);
		work();
	};

	// The permutations of a (seed, push cursor): translation and SPIR-V emission in parallel, the
	// module registration (shader ids) under a lock. A seed the recompiler rejects (an EXIT: code
	// the game never runs, or a specialization it never uses) is skipped and reported.
	using Permutation  = ProgramCache::Permutation;
	using Permutations = std::vector<std::unique_ptr<Permutation>>;
	std::mutex                    finish_mutex;
	std::atomic<size_t>           indirect {0};
	size_t                        compiled = 0;
	std::map<std::string, size_t> rejections;
	std::vector<std::string>      rejected(seed_words.size());
	// The SPIR-V of each permutation (its hash: what the driver cache is keyed by; its size).
	struct Spirv {
		uint64_t hash  = 0;
		size_t   words = 0;
	};
	std::unordered_map<const Permutation*, Spirv> spirv_of;
	const auto compile = [&](uint32_t index, uint32_t push_cursor) {
		Permutations              out;
		LocalShaderWarmup::Record seed;
		if (!decode(index, seed)) return out;
		std::vector<uint32_t> user_data(seed.user_data_count);
		const auto            options = LocalShaderWarmup::Options(seed, user_data);
		try {
			Common::RecoverableExitScope recoverable;
			auto translated = ShaderRecompiler::TranslateProgram(seed.code, options);
			if (std::ranges::any_of(translated.program.descriptor_sources,
			                        [](const auto& source) { return source.indirect_image.has_value(); }))
				indirect.fetch_add(1, std::memory_order_relaxed);
			// A seed that carries a specialization (a recorded warmup file) is compiled with it.
			const bool recorded = !seed.specialization.buffers.empty() || !seed.specialization.images.empty();
			auto       specializations =
			    recorded ? std::vector {seed.specialization} : SeedSpecializations(translated.program);
			if (recorded) PortableIndirectTables(translated.program, specializations[0]);
			for (size_t variant = 0; variant < specializations.size(); variant++) {
				auto& specialization = specializations[variant];
				if (variant != 0) translated = ShaderRecompiler::TranslateProgram(seed.code, options);
				// The module a program gets when the game meets it (ProgramCache::Get): portable formats.
				ShaderRecompiler::IR::CanonicalizeSpecialization(translated.program.info, specialization, translated.program.stage);
				(void)ShaderRecompiler::IR::PortableFormats(translated.program.info, specialization);
				auto result =
				    ShaderRecompiler::CompileProgram(std::move(translated), options, specialization, push_cursor);
				const Spirv     spirv {XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t)),
				                   result.spirv.size()};
				// KYTY_SHADER_DUMP=<dir>: the module too, as <stage>_<hash>_<SPIR-V hash>.spv.
				if (const char* dir = std::getenv("KYTY_SHADER_DUMP"); dir != nullptr && *dir != 0) {
					const auto path = std::filesystem::path(dir) / fmt::format("{}_{:016x}_{:016x}.spv",
					                                                           static_cast<uint32_t>(seed.stage),
					                                                           seed.hash, spirv.hash);
					if (auto* file = std::fopen(Common::PathToString(path).c_str(), "wb")) {
						std::fwrite(result.spirv.data(), sizeof(uint32_t), result.spirv.size(), file);
						std::fclose(file);
					}
				}
				ShaderParams    params {seed.code, user_data, seed.hash, seed.back_code};
				std::lock_guard lock(finish_mutex);
				out.push_back(std::make_unique<Permutation>(
				    programs.FinishPermutation(params, options, std::move(result), std::move(specialization))));
				spirv_of[out.back().get()] = spirv;
				if (++compiled % 1000 == 0) PipelineCacheLog("Shader seeds: compiled {} programs, {} ms", compiled, elapsed());
			}
		} catch (Common::RecoverableExit& exit) {
			std::lock_guard lock(finish_mutex);
			rejections[exit.message]++;
			rejected[index] = std::move(exit.message);
		}
		return out;
	};

	// Pixel and compute shaders: push data from dword 0.
	std::vector<Permutations> first(seed_words.size());
	parallel(seed_words.size(), [&](size_t i) {
		if (wanted[i]) first[i] = compile(static_cast<uint32_t>(i), 0);
	});
	// A pipeline's pixel shader permutations (none: one without a pixel shader), and where its vertex
	// shader's push data starts after each.
	const auto pixel_variants = [&](const LocalShaderWarmup::PipelineRecord& recipe) {
		std::vector<std::pair<const Permutation*, uint32_t>> out;
		if (recipe.pixel == LocalShaderWarmup::NoShader) {
			out.emplace_back(nullptr, 0u);
			return out;
		}
		for (const auto& permutation: first[recipe.pixel]) {
			uint32_t cursor = 0;
			permutation->program.bindings.AdvancePushData(cursor);
			out.emplace_back(permutation.get(), cursor);
		}
		return out;
	};
	std::map<std::pair<uint32_t, uint32_t>, Permutations> vertex;
	for (size_t r = 0; r < recipes.size(); r++) {
		if (!mine(seed_words.size() + r)) continue;
		for (const auto& [pixel, cursor]: pixel_variants(recipes[r])) vertex.try_emplace({recipes[r].vertex, cursor});
	}
	std::vector<decltype(vertex)::value_type*> vertex_jobs;
	for (auto& entry: vertex) vertex_jobs.push_back(&entry);
	parallel(vertex_jobs.size(), [&](size_t n) {
		vertex_jobs[n]->second = compile(vertex_jobs[n]->first.first, vertex_jobs[n]->first.second);
	});
	const auto programs_ms = elapsed();

	// Pipelines into the static cache, or a share's into a cache of its own; the objects themselves
	// are not kept. The static cache is looked up first (CreatePipelineInternal): a rerun compiles
	// only what it misses. With pipeline binaries every pipeline's binaries go to a shard file
	// (.shard<i>of<n>-<seeds>, merged by --merge --prune), copied from the store when it holds them, else captured
	// (from the static cache, else compiled through a cache of the shard's own).
	const bool                            binaries = create_pipelines && m_graphics.pipeline_binaries_enabled;
	std::unique_ptr<PipelineBinaryWriter> writer;
	if (binaries) {
		writer                            = std::make_unique<PipelineBinaryWriter>(m_graphics.device);
		m_graphics.pipeline_binary_writer = writer.get();
	}
	vk::PipelineCache target = binaries ? nullptr : m_static_cache;
	if (create_pipelines && (shards > 1 || binaries)) {
		vk::PipelineCacheCreateInfo info {};
		if (m_graphics.device.createPipelineCache(&info, nullptr, &target) != vk::Result::eSuccess) target = nullptr;
	}
	auto target_file = binaries ? StaticBinariesPath() : StaticCachePath();
	// Named after the seed file too: a run shards several (precompile-windows.ps1: seeds and recorded).
	if (shards > 1 || binaries)
		target_file += fmt::format(".shard{}of{}-{}", shard, shards, Common::PathToString(args.seeds.stem()));
	const auto save = [&] {
		if (!binaries) return SaveStaticCache(target, target_file);
		auto temporary = target_file;
		temporary += ".tmp";
		std::error_code error;
		if (!writer->Save(temporary, StaticBinariesSignature(m_graphics))) return false;
		std::filesystem::rename(temporary, target_file, error);
		PipelineCacheLog("Static pipeline binaries: saved {} pipelines ({} created from the store, {} captured, {} left "
		                 "out; {} held by other shards) to {}",
		                 writer->Pipelines(), writer->Copied(), writer->Captured(), writer->LeftOut(),
		                 writer->ClaimedElsewhere(), Common::PathToString(target_file));
		return !error;
	};
	const auto          driver_cache = target != nullptr ? target : m_driver_cache;
	std::atomic<size_t> created {0};
	// args.timings: a line per pipeline (compile ms, SPIR-V words, the seeds' hashes).
	std::mutex               timings_mutex;
	std::vector<std::string> timings;
	const auto created_one = [&](std::chrono::steady_clock::time_point start, size_t words, uint64_t hash,
	                             uint64_t pixel_hash) {
		if (const auto done = created.fetch_add(1, std::memory_order_relaxed) + 1; done % 1000 == 0)
			PipelineCacheLog("Shader seeds: created {} pipelines, {} ms", done, elapsed());
		if (args.timings.empty()) return;
		const auto      ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		std::lock_guard lock(timings_mutex);
		timings.push_back(fmt::format("{:.1f} {} {:016x} {:016x}", ms, words, hash, pixel_hash));
	};
	const auto words_of = [&](const Permutation* permutation) {
		return permutation != nullptr ? spirv_of.find(permutation)->second.words : size_t {0};
	};
	const auto destroy = [&](Pipeline& pipeline) {
		m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
		m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
		m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	};
	// Saved every ten minutes too: an interrupted precompile loses at most that.
	std::mutex              checkpoint_mutex;
	std::condition_variable checkpoint_wake;
	bool                    pipelines_done = false;
	std::jthread            checkpoints;
	if (create_pipelines && (target != nullptr || binaries)) {
		checkpoints = std::jthread([&] {
			std::unique_lock lock(checkpoint_mutex);
			while (!checkpoint_wake.wait_for(lock, std::chrono::minutes(10), [&] { return pipelines_done; })) {
				lock.unlock();
				if (save())
					PipelineCacheLog("Shader seeds: checkpoint after {} pipelines, {} ms", created.load(), elapsed());
				lock.lock();
			}
		});
	}
	if (create_pipelines) {
		parallel(seed_words.size() + recipes.size(), [&](size_t n) {
			// Once binaries are left out, the rest would be too: for a smaller shard.
			if (!mine(n) || (writer != nullptr && writer->LeftOut() != 0)) return;
			if (n < seed_words.size()) {
				if (stage_of(n) != ShaderType::Compute) return;
				LocalShaderWarmup::Record input;
				if (!decode(static_cast<uint32_t>(n), input)) return;
				for (const auto& permutation: first[n]) {
					input.compute.stage.program = &permutation->program;
					Pipeline   pipeline;
					const auto start = std::chrono::steady_clock::now();
					CreatePipelineInternal(m_graphics, pipeline, input.compute, permutation->handle.module, driver_cache);
					destroy(pipeline);
					created_one(start, words_of(permutation.get()), input.hash, 0);
				}
				return;
			}
			const auto&               recipe = recipes[n - seed_words.size()];
			LocalShaderWarmup::Record vs, ps;
			if (!decode(recipe.vertex, vs) || (recipe.pixel != LocalShaderWarmup::NoShader && !decode(recipe.pixel, ps)))
				return;
			for (const auto& [pixel, cursor]: pixel_variants(recipe)) {
				const auto found = vertex.find({recipe.vertex, cursor});
				if (found == vertex.end()) continue;
				if (pixel) ps.pixel.stage.program = &pixel->program;
				for (const auto& vertex_permutation: found->second) {
					vs.vertex.stage.program = &vertex_permutation->program;
					Pipeline   pipeline;
					const auto start = std::chrono::steady_clock::now();
					CreatePipelineInternal(m_graphics, pipeline, recipe.rendering, recipe.vertex_input, vs.vertex,
					                       vertex_permutation->handle, pixel ? &ps.pixel : nullptr,
					                       pixel ? pixel->handle : ShaderProgram {}, recipe.state, driver_cache,
					                       recipe.native_bindings);
					destroy(pipeline);
					created_one(start, words_of(vertex_permutation.get()) + words_of(pixel), vs.hash,
					            pixel != nullptr ? ps.hash : 0);
				}
			}
		});
	}

	{
		std::lock_guard lock(checkpoint_mutex);
		pipelines_done = true;
	}
	checkpoint_wake.notify_all();
	if (checkpoints.joinable()) checkpoints.join();

	const bool saved_pipelines = !create_pipelines || ((target != nullptr || binaries) && save());
	left_out                          = writer != nullptr ? writer->LeftOut() : 0;
	m_graphics.pipeline_binary_writer = nullptr;
	if (!args.timings.empty()) {
		auto path = args.timings;
		if (shards > 1) path += fmt::format(".shard{}", shard);
		if (auto* file = std::fopen(Common::PathToString(path).c_str(), "w")) {
			for (const auto& line: timings) std::fprintf(file, "%s\n", line.c_str());
			std::fclose(file);
		}
	}
	if (target != nullptr && target != m_static_cache) m_graphics.device.destroyPipelineCache(target, nullptr);
	size_t spirv_words = 0;
	for (const auto& [permutation, spirv]: spirv_of) spirv_words += spirv.words;
	PipelineCacheLog("Shader seeds: {} programs ({} with indirect image tables, {} MiB of SPIR-V) in {} ms; {} pipelines "
	                 "in {} ms ({} threads)",
	                 compiled, indirect.load(), spirv_words >> 18u, programs_ms, created.load(), elapsed() - programs_ms,
	                 threads);
	for (const auto& [message, count]: rejections) PipelineCacheLog("Shader seeds: rejected {}: {}", count, message);

	// What was compiled, as a warmup file (compiler inputs with the specialization used), with the SPIR-V
	// hash of each input (.spirv.txt) and the rejected seeds (.rejected.txt: index, stage, hash, reason).
	if (!args.out.empty() || args.static_inputs) {
		const auto               out = args.static_inputs ? StaticInputsPath() : args.out;
		LocalShaderWarmup::Cache result;
		std::error_code          error;
		std::filesystem::remove(out, error);
		(void)result.Open(out, PipelineCacheTitleId() + ShaderInputDeviceSignature(m_graphics.GetPhysicalDeviceProperties()));
		std::map<uint32_t, uint64_t>                    saved_spirv; // output record -> SPIR-V hash
		std::unordered_map<const Permutation*, uint32_t> saved;
		const auto add = [&](uint32_t index, const Permutation& permutation, uint32_t cursor) {
			LocalShaderWarmup::Record record;
			if (!decode(index, record)) return;
			record.specialization = permutation.specialization;
			record.push_cursor    = cursor;
			const auto added      = result.Add(record);
			if (added == LocalShaderWarmup::NoShader) return;
			saved[&permutation] = added;
			saved_spirv[added]  = spirv_of[&permutation].hash;
		};
		const auto saved_index = [&](const Permutation* permutation) {
			const auto found = saved.find(permutation);
			return found == saved.end() ? LocalShaderWarmup::NoShader : found->second;
		};
		for (uint32_t i = 0; i < first.size(); ++i) {
			for (const auto& permutation: first[i]) {
				add(i, *permutation, 0);
				if (stage_of(i) != ShaderType::Compute || saved_index(permutation.get()) == LocalShaderWarmup::NoShader)
					continue;
				LocalShaderWarmup::PipelineRecord recipe;
				recipe.compute = saved_index(permutation.get());
				result.AddPipeline(recipe);
			}
		}
		for (const auto& [key, permutations]: vertex) {
			for (const auto& permutation: permutations) add(key.first, *permutation, key.second);
		}
		for (const auto& recipe: recipes) {
			for (const auto& [pixel, cursor]: pixel_variants(recipe)) {
				const auto found = vertex.find({recipe.vertex, cursor});
				if (found == vertex.end()) continue;
				const auto pixel_index = pixel ? saved_index(pixel) : LocalShaderWarmup::NoShader;
				if (pixel && pixel_index == LocalShaderWarmup::NoShader) continue;
				for (const auto& vertex_permutation: found->second) {
					auto compiled_recipe   = recipe;
					compiled_recipe.vertex = saved_index(vertex_permutation.get());
					compiled_recipe.pixel  = pixel_index;
					if (compiled_recipe.vertex != LocalShaderWarmup::NoShader) result.AddPipeline(compiled_recipe);
				}
			}
		}
		const bool written = result.Save();
		PipelineCacheLog("Shader seeds: {} {} ({} inputs, {} pipelines)", written ? "wrote" : "could not write",
		                 Common::PathToString(out), result.records.size(), result.pipelines.size());
		// The lists are for coverage (precompile.py), not next to the static cache.
		const auto list = [&](const char* suffix, const auto& write) {
			auto path = out;
			path += suffix;
			if (args.static_inputs) return;
			if (auto* file = std::fopen(Common::PathToString(path).c_str(), "w")) {
				write(file);
				std::fclose(file);
			}
		};
		list(".spirv.txt", [&](std::FILE* file) {
			for (const auto& [index, hash]: saved_spirv)
				std::fprintf(file, "%u %016llx\n", index, static_cast<unsigned long long>(hash));
		});
		list(".rejected.txt", [&](std::FILE* file) {
			for (uint32_t i = 0; i < rejected.size(); ++i) {
				if (rejected[i].empty()) continue;
				LocalShaderWarmup::Record seed;
				(void)decode(i, seed);
				std::fprintf(file, "%u %u %016llx %s\n", i, static_cast<uint32_t>(seed.stage),
				             static_cast<unsigned long long>(seed.hash), rejected[i].c_str());
			}
		});
	}
	for (const auto& permutations: first) {
		for (const auto& permutation: permutations) m_graphics.device.destroyShaderModule(permutation->handle.module, nullptr);
	}
	for (const auto& [key, permutations]: vertex) {
		for (const auto& permutation: permutations) m_graphics.device.destroyShaderModule(permutation->handle.module, nullptr);
	}
	return saved_pipelines;
}

bool PipelineCache::SaveStaticCache(vk::PipelineCache cache, const std::filesystem::path& path) {
	// A checkpoint runs while pipelines are added: what fits is a valid cache too (VK_INCOMPLETE),
	// and the rest waits for the next save.
	size_t size = 0;
	if (m_graphics.device.getPipelineCacheData(cache, &size, nullptr) != vk::Result::eSuccess) return false;
	std::vector<uint8_t> payload(size + size / 8);
	size              = payload.size();
	const auto result = m_graphics.device.getPipelineCacheData(cache, &size, payload.data());
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) return false;
	payload.resize(size);
	const auto      signature = StaticCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const uint64_t  hash      = XXH3_64bits(payload.data(), payload.size());
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	auto temporary = path;
	temporary += ".tmp";
	{
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		file.write(signature.data(), static_cast<std::streamsize>(signature.size()));
		file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
		file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
		if (!file) return false;
	}
	std::filesystem::rename(temporary, path, error);
	PipelineCacheLog("Static pipeline cache: saved {} bytes to {}", payload.size(), Common::PathToString(path));
	return !error;
}

bool PipelineCache::Precompile(GraphicContext& graphics, const PrecompileOptions& options, size_t& left_out) {
	PipelineCache cache(graphics);
	if (cache.m_static_cache == nullptr) cache.InitializeStaticCache(true);
	return cache.WarmSeeds(options, left_out);
}

// Both files are saved whole (a temporary file renamed), so their first line tells the GPU and driver.
bool PipelineCache::StaticCacheCurrent(GraphicContext& graphics) {
	if (graphics.pipeline_binaries_enabled) {
		return PipelineBinaries::Open(graphics.device, StaticBinariesPath(), StaticBinariesSignature(graphics)) != nullptr;
	}
	const auto    signature = StaticCacheSignature(graphics.GetPhysicalDeviceProperties());
	std::ifstream file(StaticCachePath(), std::ios::binary);
	std::string   saved(signature.size(), '\0');
	return file.read(saved.data(), static_cast<std::streamsize>(saved.size())) && saved == signature;
}

bool PipelineCache::StaticInputsCurrent(GraphicContext& graphics, const std::filesystem::path& seeds) {
	const auto      path = StaticInputsPath();
	std::error_code error;
	const auto      written = std::filesystem::last_write_time(path, error);
	if (error) return false;
	const auto seeded = std::filesystem::last_write_time(seeds, error);
	const auto signature = PipelineCacheTitleId() + ShaderInputDeviceSignature(graphics.GetPhysicalDeviceProperties());
	const auto identity = LocalShaderWarmup::Cache::FileIdentity(path);
	return !error && written >= seeded &&
	       (identity == "KytyShaderWarmup3:" + signature || identity == "KytyShaderWarmup4:" + signature);
}

// The store and its shards' files (or only the shards': prune), into a new store; the shard files are
// removed once merged. Their data is copied from the files when the new one is written.
bool PipelineCache::MergeBinaryShards(GraphicContext& graphics, bool prune) {
	const auto                                     path      = StaticBinariesPath();
	const auto                                     signature = StaticBinariesSignature(graphics);
	PipelineBinaryWriter                           writer(graphics.device);
	std::vector<std::unique_ptr<PipelineBinaries>> sources;
	std::vector<std::filesystem::path>             files;
	if (!prune) {
		if (auto store = PipelineBinaries::Open(graphics.device, path, signature)) {
			writer.CopyAll(*store);
			sources.push_back(std::move(store));
		}
	}
	std::error_code error;
	for (const auto& entry: std::filesystem::directory_iterator(path.parent_path(), error)) {
		const auto name = entry.path().filename().string();
		if (!name.starts_with(path.filename().string() + ".shard") || name.ends_with(".tmp")) continue;
		if (auto shard = PipelineBinaries::Open(graphics.device, entry.path(), signature)) {
			writer.CopyAll(*shard);
			sources.push_back(std::move(shard));
			files.push_back(entry.path());
		}
	}
	if (files.empty()) {
		PipelineCacheLog("Static pipeline binaries: no shards to merge");
		return true;
	}
	auto temporary = path;
	temporary += ".tmp";
	const bool saved = writer.Save(temporary, signature);
	sources.clear(); // closes the files: the store may be replaced
	if (saved) std::filesystem::rename(temporary, path, error);
	const bool merged = saved && !error;
	if (merged) {
		for (const auto& file: files) std::filesystem::remove(file, error);
	}
	PipelineCacheLog("Static pipeline binaries: merged {} shards{}, {} pipelines{}", files.size(),
	                 prune ? " (pruned)" : "", writer.Pipelines(), merged ? "" : ": FAILED");
	return merged;
}

// The shard files are removed once merged.
bool PipelineCache::MergePrecompileShards(GraphicContext& graphics, bool prune) {
	if (graphics.pipeline_binaries_enabled) return MergeBinaryShards(graphics, prune);
	PipelineCache cache(graphics);
	if (cache.m_static_cache == nullptr) cache.InitializeStaticCache(true);
	if (cache.m_static_cache == nullptr) return false;
	const auto                         path      = StaticCachePath();
	const auto                         signature = StaticCacheSignature(graphics.GetPhysicalDeviceProperties());
	std::vector<vk::PipelineCache>     shards;
	std::vector<std::filesystem::path> files;
	std::error_code                    error;
	for (const auto& entry: std::filesystem::directory_iterator(path.parent_path(), error)) {
		if (!entry.path().filename().string().starts_with(path.filename().string() + ".shard")) continue;
		const auto payload = ReadPipelineCacheFile(entry.path(), signature);
		if (payload.empty()) continue;
		vk::PipelineCacheCreateInfo info {};
		info.initialDataSize = payload.size();
		info.pInitialData    = payload.data();
		vk::PipelineCache shard = nullptr;
		if (graphics.device.createPipelineCache(&info, nullptr, &shard) != vk::Result::eSuccess) continue;
		shards.push_back(shard);
		files.push_back(entry.path());
	}
	const bool merged = shards.empty() ||
	                    (graphics.device.mergePipelineCaches(cache.m_static_cache, static_cast<uint32_t>(shards.size()),
	                                                         shards.data()) == vk::Result::eSuccess &&
	                     cache.SaveStaticCache(cache.m_static_cache, path));
	if (merged) {
		for (const auto& file: files) std::filesystem::remove(file, error);
	}
	for (const auto shard: shards) graphics.device.destroyPipelineCache(shard, nullptr);
	PipelineCacheLog("Static pipeline cache: merged {} shards", shards.size());
	return merged;
}
