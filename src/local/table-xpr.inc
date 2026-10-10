// --- Table draws (GPU-driven XPR, docs/GPU-DRIVEN-XPR.md) ------------------------------
//
// Included by renderDraw.cpp after native-xpr.inc (local Vulkan recording builds only).
//
// KYTY_TABLE_XPR=1: the XPR draws of a shader pair whose programs allow table mode
// (ShaderRecompiler::IR::EnterTableMode) need no record. For each draw the renderer evaluates the programs' table
// plans (IR::TablePlan): the slots their shaders read and the T#/S# words, from guest memory as it is when the draw
// is translated (as the normal path's SRT evaluation, which the game's command buffers patch between draws); the
// T#/S# words pick the pair's image set - a descriptor set with the images and the samplers, proven current once
// a frame - and each V# is resolved to its buffer's device address (cached while its buffer and region stay the
// same). The slots and buffer addresses go into a block in the table upload ring whose device address is pushed
// with the user data: the shaders read it, no flattened SRT binding, no buffer descriptors.
//
// The normal path, asked to store (TableStore), compiles the pair's table mode programs and the pipeline of each
// draw state, and captures the draw state (NativeXprCaptureState).
//
// KYTY_TABLE_XPR=2 also continues a run (TableContinue): a draw the observer calls clean (no register but user
// data and index state changed) right after a table draw whose emission is the last one in the command buffer
// has the same pair and draw state, with its pipeline and dynamic state still bound.

// A draw state's pipeline.
struct RenderExecutor::TableVariant {
	uint64_t                       state          = 0;
	const PipelineCache::Pipeline* pipeline       = nullptr; // null while a worker compiles it
	bool                           fetch_embedded = false;   // the vertex program fetches (a direct draw's offsets)
	bool                           refused        = false;   // another specialization than the pair's programs
	// The pixel program's export mapping under this state (GetGraphicsPrograms).
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
};

// The descriptor set of one choice of images and samplers (the T#/S# words of every stage).
struct RenderExecutor::TableImageSet {
	struct Image {
		ImageId                 id;
		uint64_t                serial = 0;
		vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
		ImageSubresourceRange   range;
		TextureCache::ImageDesc desc;
		vk::AccessFlags2        access = vk::AccessFlagBits2::eShaderRead;
		bool                    target  = false; // a target of the draw state the set was made under (its access)
		bool                    storage = false; // the shader writes it: GPU-modified at every use (TableTransit)
		bool                    whole   = false; // bound both as a texture and as storage: the whole image's state
	};
	std::vector<uint32_t> words; // without the LOD statistics fields (NativeXprLodStatsMask)
	vk::DescriptorSet     set  = nullptr;
	vk::DescriptorPool    pool = nullptr;
	std::vector<Image>    images;
	uint64_t              validated_frame  = UINT64_MAX;
	uint64_t              resolution_epoch = 0;
	uint64_t              used_frame       = 0;
	// The words select another specialization, or an image no set can hold: the draws take the native path
	// (TableResolveSet tries again after TableImageSet::RetryFrames).
	bool                  refused = false;
	static constexpr uint64_t RetryFrames = 300;
};

struct RenderExecutor::TablePair {
	// Evaluated slots at consecutive addresses (the same base, immediates 4 apart): one read, at the address of the
	// first. Ordered by level: a run whose address is made of slots is read after the runs of those slots.
	struct Run {
		ShaderRecompiler::IR::TablePlan::Slot read;
		uint32_t                              first = 0, count = 0, level = 0;
	};
	uint64_t                                                    vs = 0, ps = 0; // (a dispatch's: its shader, 0)
	// The GS user SGPR the CP writes an indirect draw's start instance into (START_INST_LOC, -1: none): the vertex
	// program is translated for it (ShaderVertexInputInfo::start_instance_sgpr).
	int32_t                                                     start_instance = -1;
	bool                                                        refused  = false;
	std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {}; // vertex, pixel (null: inactive)
	std::array<std::vector<Run>, 2>                             runs;
	std::vector<TableVariant>                                   variants;
	uint64_t                                                    store_frame = 0;
	std::unordered_map<uint64_t, std::unique_ptr<TableImageSet>> sets; // by T#/S# words
	TableImageSet*                                              last_set = nullptr;
	// ShaderMapGeneration under which the shaders at vs and ps were proven those the programs were compiled
	// from (the game registers shaders often; code replaced at an address resets the pair).
	uint64_t                                                    generation = UINT64_MAX;
	// A dispatch's (TableDispatch): its program's input and pipeline, and the normal path's dispatches seen.
	std::unique_ptr<ShaderComputeInputInfo> compute;
	ShaderProgram                           compute_program;
	const PipelineCache::Pipeline*          pipeline  = nullptr;
	uint32_t                                sightings = 0;
	// Per stage, the buffers whose V#s were seen in another format than the programs decode (bits by buffer): the
	// programs are compiled again with those buffers' formats decoded at run time (`respecialize`, TableReset).
	std::array<uint32_t, 2> portable {};
	bool                    respecialize = false;
	// A translation or a pipeline compile the pair's next store needs (TableStore, TableDispatch): until it is done,
	// its draws keep the native path without asking for a store (a request a frame each kept the store budget).
	std::shared_ptr<const std::atomic<bool>> waiting;
};

struct RenderExecutor::TableXpr {
	using Pair    = TablePair;
	using Variant = TableVariant;
	std::unordered_map<uint64_t, Pair> pairs;          // by shader addresses
	std::unordered_map<uint64_t, Pair> dispatches;     // by the compute stage's registers (TableDispatchKey)
	uint64_t collected  = 0;                           // the frame unused image sets were last dropped
	// The next prepared draw of the pair under this state stores (TableStore); the frame's store requests so far.
	Pair*    store       = nullptr;
	uint64_t store_state = 0, store_frame = 0;
	uint32_t store_requests = 0;
	// The draw's T#/S# words (the set's key) and pixel stage LOD statistics words; its runs' addresses.
	std::vector<uint32_t>                words;
	std::vector<uint32_t>                lod_stats;
	std::vector<uint64_t>                addresses;
	// 16 KiB guest pages seen mapped (direct-mapped by page), while the mapping epoch is the same.
	std::array<uint64_t, 256> mapped {};
	uint64_t                  mapping_epoch = UINT64_MAX;
	// The last table draw (TableContinue): its pair, the index of its variant and its draw state.
	Pair*                     last_pair    = nullptr;
	size_t                    last_variant = 0;
	const NativeXprDrawState* last_state   = nullptr;
	const TableImageSet*      last_set     = nullptr; // the set the last emission bound
	// The game buffer the last arguments came from, and the indices' by their address (each mesh has its own): used
	// again while no registration changed and the guest wrote none of its region (ObtainBuffer then returns the same
	// buffer; it streams CPU-dirty ranges).
	struct HeldBuffer {
		uint64_t begin = 0, end = 0, cpu_epoch = 0, registration = 0;
		Buffer*  buffer = nullptr;
	};
	HeldBuffer                   args;
	std::array<HeldBuffer, 8192> indices;
	// A V#'s buffer as the shaders read it (direct-mapped by the V#'s words, one cache line each): its device address
	// and size in dwords (0: none; clamped to the mappings), while no buffer covering its tracker regions was registered
	// or unregistered since `registration` (the registration epoch it was last proven at: a region stamped later moved
	// a buffer) and the guest wrote none of its range (the tracker regions' epoch, else the range's dirty pages). One
	// whose proof also needs the image starts and the mappings (formatted: an image may start at it; clamped: a mapping
	// may extend it) holds while `special` is the table's special generation (0: no such need). (Buffer registrations
	// anywhere used to drop them all: 3800 resolves a frame walking at 1-1, 1000 standing.)
	struct alignas(64) ResolvedBuffer {
		std::array<uint32_t, 4> words {};
		uint32_t                dwords = 0, special = 0;
		vk::DeviceAddress       device = 0;
		uint64_t                address = 0, size = 0, cpu_epoch = 0, registration = 0;
	};
	static constexpr size_t           ResolvedEntries = 65536;
	std::unique_ptr<ResolvedBuffer[]> resolved = std::make_unique<ResolvedBuffer[]>(ResolvedEntries);
	uint32_t                          special_generation = 1; // moves with the image starts and the mappings
	uint64_t                          resolved_start = UINT64_MAX, resolved_mapping = UINT64_MAX;
	std::vector<std::pair<std::array<uint32_t, 4>, ResolvedBuffer*>> entries; // a stage's V#s (TableDraw)
	// The draw's blocks (per stage): its slots (IR::TablePlan::BlockDwords), then its buffers.
	std::array<std::vector<uint32_t>, 2> blocks;
};

// The pair's programs and what was made for them leave (its shaders changed, or a V#'s format did:
// TablePair::portable); the next draw or dispatch compiles them again.
void RenderExecutor::TableReset(TablePair& pair) {
	auto& table = *m_table_xpr;
	for (auto& [words, set]: pair.sets) NativeXprRetireSet(set->pool, set->set);
	if (table.store == &pair) table.store = nullptr;
	if (table.last_pair == &pair) table.last_pair = nullptr;
	table.last_set = nullptr;
	pair = {.vs = pair.vs, .ps = pair.ps, .generation = pair.generation, .sightings = pair.sightings,
	        .portable = pair.portable};
}

RenderExecutor::TableResult RenderExecutor::TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands,
                                                     uint64_t index_base, uint64_t index_bytes,
                                                     vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	if (!m_table_xpr) m_table_xpr = std::make_shared<TableXpr>();
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 0); // census: the table path's phases (b)
	auto&       table = *m_table_xpr;
	auto&       cache = *m_native_xpr;
	const auto& sh    = buffer.GetShaders();
	const auto& vs    = sh.GetVs();
	// The vertex program is the ES stage's (ShaderGetInputInfoVS), without tessellation.
	if (vs.es_regs.data_addr == 0 || vs.ls_regs.data_addr != 0 || vs.hs_regs.data_addr != 0) return TableResult::Native;
	const auto frame      = m_context.FrameNumber();
	const auto vs_address = vs.es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const std::array<uint64_t, 3> pair_key {vs_address, ps_address, static_cast<uint32_t>(vs.start_instance_user_sgpr)};
	auto& pair = table.pairs[XXH3_64bits(pair_key.data(), sizeof(pair_key))];
	if (pair.vs == 0) {
		pair.vs             = vs_address;
		pair.ps             = ps_address;
		pair.start_instance = vs.start_instance_user_sgpr;
	}
	if (pair.vs != vs_address || pair.ps != ps_address || pair.start_instance != vs.start_instance_user_sgpr)
		return TableResult::Native;
	if (const auto generation = ShaderMapGeneration(); generation != pair.generation) {
		if (pair.programs[0] != nullptr &&
		    (ShaderHashAt(vs_address) != pair.programs[0]->shader_hash ||
		     (pair.programs[1] != nullptr && ShaderHashAt(ps_address) != pair.programs[1]->shader_hash))) {
			pair.portable = {};
			TableReset(pair);
		}
		pair.generation = generation;
	}
	if (pair.respecialize) TableReset(pair);
	if (pair.refused) return TableResult::Native;
	// A store at most once a frame per pair, within the frame's budget (kyty_local_table_store_budget): while its
	// programs and pipeline compile, the pair's draws keep the native path.
	const auto request_store = [&] {
		if (pair.waiting) {
			if (!pair.waiting->load(std::memory_order_acquire)) return TableResult::Native;
			pair.waiting.reset();
		}
		if (table.store_frame != frame) {
			table.store_frame    = frame;
			table.store_requests = 0;
		}
		const auto budget = kyty_local_table_store_budget.load(std::memory_order_relaxed);
		if (pair.store_frame == frame || (budget != 0 && table.store_requests >= budget)) return TableResult::Native;
		++table.store_requests;
		pair.store_frame  = frame;
		table.store       = &pair;
		table.store_state = cache.state_key;
		return TableResult::Store;
	};
	const auto variant = std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
	if (variant != pair.variants.end() && variant->refused) return TableResult::Native;
	if (pair.programs[0] == nullptr || variant == pair.variants.end() || variant->pipeline == nullptr)
		return LiveCounters::Add(LiveCounters::TableStoreVariant), request_store();
	// The draw state (TableStore captures it).
	if (cache.last_state == nullptr || cache.last_state_key != cache.state_key) {
		const auto state_it = cache.states.find(cache.state_key);
		if (state_it == cache.states.end()) return request_store();
		cache.last_state_key = cache.state_key;
		cache.last_state     = state_it->second.get();
	}
	const auto& draw_state = *cache.last_state;
	if (!NativeXprTargetsCurrent(draw_state)) return LiveCounters::Add(LiveCounters::TableStoreTargets), request_store();
	return TableDraw(buffer, pair, *variant, draw_state, commands, index_base, index_bytes, index_type, direct);
}

bool RenderExecutor::TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
                                   uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct) {
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) < 2 || !m_table_xpr || !m_native_xpr) return false;
	auto& table = *m_table_xpr;
	if (table.last_pair == nullptr || buffer.IsInvalid() || index_base == 0 || index_bytes == 0 ||
	    commands.size() > 64 || table.last_pair->generation != ShaderMapGeneration() ||
	    table.last_pair->start_instance != buffer.GetShaders().GetVs().start_instance_user_sgpr)
		return false;
	auto&       variant = table.last_pair->variants[table.last_variant];
	const auto& emit    = m_native_xpr->emit;
	if (emit.command != buffer.Handle() || emit.generation != buffer.GraphicsGeneration() ||
	    emit.state != table.last_state || emit.pipeline != variant.pipeline)
		return false;
	Common::LockGuard lock(m_context.GetMutex());
	if (!NativeXprTargetsCurrent(*table.last_state)) return false;
	if (TableDraw(buffer, *table.last_pair, variant, *table.last_state, commands, index_base, index_bytes, index_type,
	              direct) != TableResult::Drawn)
		return false;
	LiveCounters::Add(LiveCounters::TableContinued, std::max<size_t>(commands.size(), 1));
	return true;
}

// The image set of the T#/S# words (`words`) of a draw (under `variant`) or a dispatch (`variant` null): its
// resources as the normal path evaluates them, which must select the pair's table mode programs (the same
// specializations), and its images and samplers as the normal path resolves and acquires them (PrepareBindingsInto,
// RebindImages), in a descriptor set with the page table and the fault buffer. Refused for another specialization,
// or when an image cannot be held in a set (a target the shader cannot read, or writes as storage).
RenderExecutor::TableImageSet* RenderExecutor::TableResolveSet(CommandBuffer& buffer, TablePair& pair,
                                                               const TableVariant*       variant,
                                                               const NativeXprDrawState* draw_state,
                                                               std::span<const uint32_t> words) {
	auto       set      = std::make_unique<TableImageSet>();
	auto&      textures = m_context.GetTextureCache();
	auto&      buffers  = m_context.GetBufferCache();
	const auto stages   = pair.programs[1] != nullptr ? 2u : 1u;
	const auto& sh      = buffer.GetShaders();
	const auto& ctx     = buffer.GetRegisters();
	set->words.assign(words.begin(), words.end());
	set->validated_frame  = m_context.FrameNumber();
	const auto publish = [&] {
		auto& slot = pair.sets[XXH3_64bits(words.data(), words.size() * sizeof(uint32_t))];
		if (slot) NativeXprRetireSet(slot->pool, slot->set);
		slot = std::move(set);
		return slot.get();
	};
	const auto refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports[2] {}; // (dispatches, draws)
		if (reports[pair.ps != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed) < 32)
			std::printf("Table: VS 0x%llx PS 0x%llx image set refused: %s\n", static_cast<unsigned long long>(pair.vs),
			            static_cast<unsigned long long>(pair.ps), reason);
		set->refused = true;
		return publish();
	};
	ShaderVertexInputInfo  vertex_info;
	ShaderPixelInputInfo   pixel_info;
	ShaderComputeInputInfo compute_info;
	std::array<const ShaderStageRuntime*, 2> runtimes {&vertex_info.stage, &pixel_info.stage};
	if (variant != nullptr) {
		ResetNativeVertexInput(vertex_info);
		ResetNativeStageInput(pixel_info);
		(void)m_context.GetPipelineCache().GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
		                                                       buffer.GetUserConfig(), variant->mapping, stages == 2,
		                                                       vertex_info, pixel_info, true, pair.portable);
		if (m_context.GetPipelineCache().TablePending()) return nullptr;
		if (vertex_info.stage.program != pair.programs[0] || (stages == 2 && pixel_info.stage.program != pair.programs[1]))
			return refuse(vertex_info.stage.program != pair.programs[0] ? "another specialization (VS)"
			                                                            : "another specialization (PS)");
	} else {
		ResetNativeStageInput(compute_info);
		compute_info.dispatch_thread_dimensions = pair.compute->dispatch_thread_dimensions;
		(void)m_context.GetPipelineCache().GetComputeProgram(sh.GetCs(), ctx.GetShaderRegisters(), compute_info, nullptr,
		                                                     true, pair.portable[0]);
		if (m_context.GetPipelineCache().TablePending()) return nullptr;
		if (compute_info.stage.program != pair.programs[0]) return refuse("another specialization");
		runtimes[0] = &compute_info.stage;
	}
	std::array<PreparedBindings, 2>                prepared;
	for (uint32_t st = 0; st < stages; ++st) PrepareBindingsInto(*runtimes[st], prepared[st]);
	for (uint32_t st = 0; st < stages; ++st) RebindImages(prepared[st]);
	// Each image's layout and access as the normal path's CommitBindings gives them: a null image (a texture not
	// streamed in yet) as it is bound (not held); a target of the draw state in its attachment layout and access
	// (NativeXprEmitTargets moves it there); one bound both as a texture and as storage (force_general: the deferred
	// decals' G-buffer) wholly in General; storage in General; else read-only.
	std::array<std::vector<TableImageSet::Image>, 2> deps;
	bool                                             ok = true;
	for (uint32_t st = 0; st < stages; ++st) deps[st].reserve(prepared[st].images.size()); // (no regrowth: new sets come in bursts)
	for (uint32_t st = 0; st < stages && ok; ++st)
		for (auto& binding: prepared[st].images) {
			auto&       image   = textures.GetImage(binding.image_id);
			const bool  storage = binding.desc.type == TextureCache::BindingType::Storage;
			const auto  found   = draw_state == nullptr ? decltype(draw_state->targets.end()) {}
			                                                : std::ranges::find(draw_state->targets, binding.image_id,
			                                                                    &NativeXprDrawState::Target::id);
			const auto* target  = draw_state == nullptr || found == draw_state->targets.end() ? nullptr : &*found;
			const auto& view    = binding.desc.view_info;
			auto&       dep     = deps[st].emplace_back();
			dep                 = {binding.image_id, textures.m_slot_images[binding.image_id].serial,
			                       vk::ImageLayout::eUndefined, {view.base_level, view.level_count, view.base_layer,
			                                                     view.layer_count}, binding.desc};
			if (image.info.data.Empty() && !storage && image.backing.state.layout == vk::ImageLayout::eGeneral) {
				binding.layout = vk::ImageLayout::eGeneral;
				continue; // (dep.layout undefined: not held)
			}
			// (A sampled aspect the attachment writes is not shader-readable: the target's access then lacks it. A
			// linear image the shader writes is enrolled for readback at every acquisition: TrackImageDownload.)
			if (image.info.data.Empty() ||
			    (target != nullptr && (storage || !(target->access & vk::AccessFlagBits2::eShaderRead))) ||
			    ((storage || (image.binding.force_general && image.binding.shader_write)) && !image.info.IsTiled())) {
				ok = false;
				break;
			}
			if (target != nullptr) {
				binding.layout = target->layout;
				dep.access     = target->access;
				dep.target     = true;
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				binding.layout = vk::ImageLayout::eGeneral;
				dep.access     = vk::AccessFlagBits2::eShaderRead |
				                 (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlags2 {});
				dep.storage    = image.binding.shader_write;
				dep.whole      = true;
			} else if (storage) {
				binding.layout = vk::ImageLayout::eGeneral;
				dep.access     = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
				dep.storage    = true;
			} else {
				binding.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                      : vk::ImageLayout::eShaderReadOnlyOptimal;
			}
			dep.layout = binding.layout;
		}
	ResetBindings();
	if (!ok) return refuse("an image no set can hold");
	// The set: per stage its bindings (images, samplers, LOD statistics, page table, fault buffer).
	struct Pending {
		vk::WriteDescriptorSet write;
		size_t                 start = 0;
		bool                   image = false;
	};
	std::vector<Pending>                  pending;
	std::vector<vk::DescriptorBufferInfo> buffer_infos;
	std::vector<vk::DescriptorImageInfo>  image_infos;
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	const auto fixed = [](const Buffer* value) -> vk::DescriptorBufferInfo { return {value->Handle(), 0, value->Size()}; };
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		std::vector<uint32_t> occurrences(prepared[st].images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			Pending item;
			item.write.dstBinding      = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			item.write.descriptorType  = NativeDescriptorType(binding.kind);
			item.write.descriptorCount = NativeDescriptorCount(binding);
			item.image = ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			                 ShaderRecompiler::IR::ImageResourceClass::None ||
			             binding.kind == Kind::Samplers;
			item.start = item.image ? image_infos.size() : buffer_infos.size();
			if (binding.kind == Kind::Samplers) {
				for (const auto resource: binding.resources)
					image_infos.emplace_back(prepared[st].samplers.at(resource), nullptr, vk::ImageLayout::eUndefined);
			} else if (item.image) {
				for (const auto resource: binding.resources)
					image_infos.push_back(MakeImageInfo(prepared[st].images.at(resource), occurrences.at(resource)++));
			} else if (binding.kind == Kind::LodStats) {
				buffer_infos.push_back(fixed(buffers.GetLodStatsBuffer()));
			} else if (binding.kind == Kind::BdaPagetable) {
				buffer_infos.push_back(fixed(buffers.GetBdaPageTableBuffer()));
			} else if (binding.kind == Kind::FaultBuffer) {
				buffer_infos.push_back(fixed(buffers.GetFaultBuffer()));
			} else {
				return refuse("a binding table mode does not have");
			}
			pending.push_back(item);
		}
		if (st == 0) set->images.reserve(deps[0].size() + deps[1].size());
		for (const auto& dep: deps[st])
			if (dep.layout != vk::ImageLayout::eUndefined) set->images.push_back(dep);
	}
	set->set = NativeXprAllocateSet((variant != nullptr ? variant->pipeline : pair.pipeline)->descriptor_set_layout,
	                                set->pool);
	std::vector<vk::WriteDescriptorSet> writes;
	for (auto& item: pending) {
		item.write.dstSet = set->set;
		if (item.image) item.write.pImageInfo = image_infos.data() + item.start;
		else item.write.pBufferInfo = buffer_infos.data() + item.start;
		writes.push_back(item.write);
	}
	if (!writes.empty())
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	set->resolution_epoch = textures.ResolutionEpoch();
	LiveCounters::Add(LiveCounters::TableEvaluations);
	return publish();
}

// A set's images are the same objects, registered and current (as NativeXprValidateResources proves a record's):
// once a frame, and again when registrations moved.
bool RenderExecutor::TableValidateSet(TableImageSet& set) {
	auto&      textures = m_context.GetTextureCache();
	const auto frame    = m_context.FrameNumber();
	if (set.validated_frame == frame && set.resolution_epoch == textures.ResolutionEpoch()) return true;
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1e, 7);
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr || image->serial != dep.serial) return false;
		// The guest rewrote it (a texture streaming in): upload it as the normal path's FindTexture does.
		if (image->IsCpuDirty()) {
			if (!image->registered || image->depth_id || image->binding.needs_rebind) return false;
			(void)textures.FindTexture(dep.id, dep.desc);
			image = textures.m_slot_images.try_get(dep.id);
			if (image == nullptr || image->serial != dep.serial || image->IsCpuDirty()) return false;
		}
		if (dep.target) {
			if (!image->registered || image->binding.needs_rebind) return false;
			(void)textures.GetImage(dep.id); // LRU
			continue;
		}
		if (!textures.IsSampledImageCurrent(dep.id, dep.desc)) return false; // (and its LRU, as FindTexture's)
	}
	set.validated_frame  = frame;
	set.resolution_epoch = textures.ResolutionEpoch();
	return true;
}

// The slots of the pair's stages, read into their blocks where the plans address them from the user data and lower
// slots (a pointer read from memory may hold anything: what is read through it must be mapped), and the T#/S# words
// they make (TableXpr::words, the set's key) and the pixel stage's LOD statistics words. False: something unmapped.
bool RenderExecutor::TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	auto&      table     = *m_table_xpr;
	const auto stages    = pair.programs[1] != nullptr ? 2u : 1u;
	auto&      resources = m_context.GetGpuResources();
	if (const auto epoch = resources.MappingEpoch(); epoch != table.mapping_epoch) {
		table.mapped.fill(UINT64_MAX);
		table.mapping_epoch = epoch;
	}
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& words = table.words;
	words.clear();
	table.lod_stats.clear();
	NativeXprReader reader(resources);
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		// (Slots no stage reads keep what an earlier draw left: the block carries them unread.)
		auto& slots = table.blocks[st];
		slots.resize(plan.BlockDwords());
		const auto value = [&](const ShaderRecompiler::IR::TablePlan::Operand& operand) -> uint32_t {
			if (operand.kind == Kind::Immediate) return operand.value;
			if (operand.kind == Kind::Slot) return slots[operand.value];
			const auto index = operand.value - program.user_data_base;
			return index < user_data[st].size() ? user_data[st][index] : 0;
		};
		// A level's addresses are known together: their lines load at once (the guest's tables are cold).
		const auto& runs = pair.runs[st];
		for (size_t begin = 0, end = 0; begin < runs.size(); begin = end) {
			for (end = begin; end < runs.size() && runs[end].level == runs[begin].level; ++end) {
				const auto& run  = runs[end];
				const auto  base = (uint64_t {value(run.read.low)} | uint64_t {value(run.read.high)} << 32u) &
				                  0x0000fffffffffffcull;
				const auto address =
				    base + (value(run.read.offset) & ~3u) + static_cast<uint64_t>(int64_t {run.read.immediate});
				for (auto page = address >> 14u; page <= (address + run.count * 4ull - 1) >> 14u; ++page) {
					auto& seen = table.mapped[page & 255u];
					if (seen == page) continue;
					if (!Libs::LibKernel::Memory::IsFullyMapped(page << 14u, 16384)) return false;
					seen = page;
				}
				for (auto line = address & ~63ull; line < address + run.count * 4ull; line += 64)
					__builtin_prefetch(reinterpret_cast<const void*>(line));
				table.addresses[end] = address;
			}
			for (size_t i = begin; i < end; ++i) reader.Read(table.addresses[i], runs[i].count, slots.data() + runs[i].first);
		}
		// The LOD statistics fields of the T#s (the game reassigns the counters every frame) reach only the shader
		// data (as RebindBuffers computes it), not the set.
		for (size_t i = 0; i < plan.images.size(); ++i) {
			const auto& descriptor = plan.images[i];
			std::array<uint32_t, 8> image {};
			for (uint32_t w = 0; w < descriptor.count; ++w) {
				image[w] = value(descriptor.words[w]);
				words.push_back(image[w] & ~NativeXprLodStatsMask[w]);
			}
			if (i < program.bindings.lod_stats_count) {
				ShaderTextureResource desc;
				std::memcpy(&desc, image.data(), sizeof(desc));
				table.lod_stats.push_back(desc.MipStatsCntEn() ? (0x80000000u | desc.MipStatsCntId() |
				                                                  (uint32_t(desc.BaseLevel()) << 8u) |
				                                                  (uint32_t(desc.MinLodWarn5()) << 12u))
				                                               : 0u);
			}
		}
		for (const auto& descriptor: plan.samplers)
			for (uint32_t w = 0; w < descriptor.count; ++w) words.push_back(value(descriptor.words[w]));
	}
	return true;
}

// The image set of the words TableEvaluate made: the last one when they are the same, else the pair's set for them
// (`variant` null: a dispatch's), proven current. Null: the normal path takes the draw or dispatch.
RenderExecutor::TableImageSet* RenderExecutor::TableSet(CommandBuffer& buffer, TablePair& pair,
                                                        const TableVariant*       variant,
                                                        const NativeXprDrawState* draw_state) {
	auto&       table = *m_table_xpr;
	const auto& words = table.words;
	const auto  frame = m_context.FrameNumber();
	auto*       set   = pair.last_set;
	if (set == nullptr || set->words != words) {
		const auto found = pair.sets.find(XXH3_64bits(words.data(), words.size() * sizeof(uint32_t)));
		set              = found != pair.sets.end() && found->second->words == words ? found->second.get() : nullptr;
		if (set == nullptr || (set->refused && frame >= set->validated_frame + TableImageSet::RetryFrames))
			set = TableResolveSet(buffer, pair, variant, draw_state, words);
		pair.last_set = set;
		if (set == nullptr) return nullptr; // (a program still translating)
	}
	set->used_frame = frame;
	if (set->refused) return LiveCounters::Add(LiveCounters::TableRefusedSets), nullptr;
	if (!TableValidateSet(*set)) {
		// Resolved again at the next draw that uses it.
		if (table.last_set == set) table.last_set = nullptr;
		pair.last_set = nullptr;
		const auto found = pair.sets.find(XXH3_64bits(set->words.data(), set->words.size() * sizeof(uint32_t)));
		NativeXprRetireSet(set->pool, set->set);
		if (found != pair.sets.end() && found->second.get() == set) pair.sets.erase(found);
		return nullptr;
	}
	// Unused image sets leave once a second.
	if (frame >= table.collected + 60) {
		table.collected = frame;
		for (auto* pairs: {&table.pairs, &table.dispatches})
			for (auto& [key, held]: *pairs)
				std::erase_if(held.sets, [&](auto& item) {
					if (item.second->used_frame + 60 >= frame) return false;
					if (held.last_set == item.second.get()) held.last_set = nullptr;
					if (table.last_set == item.second.get()) table.last_set = nullptr;
					NativeXprRetireSet(item.second->pool, item.second->set);
					return true;
				});
	}
	return set;
}

// The pair's blocks (TableEvaluate read their slots): each stage's buffers' device addresses and sizes in dwords
// after its slots, in the table upload ring. A read-only V# is resolved once (TableXpr::resolved); a written one
// is obtained as written at every use, as the normal path's NativeStorageBuffer (the buffer cache tracks the GPU's
// writes). False: a buffer the shaders cannot take.
bool RenderExecutor::TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
                                 std::array<vk::DeviceAddress, 2>& blocks, bool* writes) {
	using Kind         = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto&      table        = *m_table_xpr;
	auto&      buffer_cache = m_context.GetBufferCache();
	auto&      textures     = m_context.GetTextureCache();
	auto&      upload       = buffer_cache.GetTableUploadBuffer();
	const auto stages       = pair.programs[1] != nullptr ? 2u : 1u;
	const auto start_epoch  = textures.StartEpoch();
	if (table.resolved_start != start_epoch || table.resolved_mapping != table.mapping_epoch) {
		table.resolved_start   = start_epoch;
		table.resolved_mapping = table.mapping_epoch;
		++table.special_generation;
	}
	// A tracker region's epoch, read once here (the buffers mostly share one: the line is another core's).
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	// A range none of whose pages the guest wrote since they were uploaded (with no write protection still queued:
	// a page marked clean can change unseen until it lands).
	const auto range_clean = [&](uint64_t address, uint64_t size) {
		for (auto region = address / TRACKER_REGION_SIZE; region <= (address + size - 1) / TRACKER_REGION_SIZE; ++region)
			if (buffer_cache.DeferredProtectionPending(region * TRACKER_REGION_SIZE)) return false;
		return !buffer_cache.IsRegionCpuModified(address, size);
	};
	// A null buffer's device address: valid memory (the shaders' loads of a buffer are not branched around its bounds:
	// out of bounds they load its first dword).
	const vk::DeviceAddress null_device = upload.BufferDeviceAddress();
	// A V#'s buffer word (IR::BufferWord, as IR::BufferDescriptorWord makes it) and stride.
	static const auto format_classes = [] {
		std::array<uint32_t, 128> classes {};
		for (uint32_t format = 0; format < classes.size(); ++format)
			classes[format] = ShaderRecompiler::IR::BufferFormatClass(static_cast<Prospero::BufferFormat>(format));
		return classes;
	}();
	const auto word_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint32_t, uint32_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return {0, 0};
		using Word = ShaderRecompiler::IR::BufferWord;
		return {(descriptor.Stride() <= Word::MaxStride ? descriptor.Stride() : 0u) |
		            format_classes[descriptor.RawFormat()] << Word::LayoutShift |
		            descriptor.DstSelXYZW() << Word::DstSelShift,
		        descriptor.Stride()};
	};
	// The bytes a V# names (0: none).
	const auto requested_of = [](const std::array<uint32_t, 4>& words4) -> uint64_t {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0 || descriptor.Base48() == 0) return 0;
		return descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                : uint64_t {descriptor.NumRecords()};
	};
	// As RenderExecutor::FindBuffers: a V# that is no buffer, or of nothing mapped, is a null buffer (size 0).
	const auto range_of = [](const std::array<uint32_t, 4>& words4) -> std::pair<uint64_t, uint64_t> {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		const auto requested = descriptor.Stride() != 0 ? uint64_t {descriptor.Stride()} * descriptor.NumRecords()
		                                                : uint64_t {descriptor.NumRecords()};
		const auto address   = descriptor.Base48();
		return {address, descriptor.Type() != 0 || address == 0 || requested == 0
		                     ? 0
		                     : Libs::LibKernel::Memory::ClampRangeSize(address, requested)};
	};
	// A V# the program was specialized for (a format and dst_sel it decodes, no swizzle or ADD_TID; strides are the
	// block's): another one takes the normal path, which chooses the program for it.
	const auto specialized = [](const ShaderRecompiler::IR::BufferResource& resource,
	                            const std::array<uint32_t, 4>&             words4) {
		ShaderBufferResource descriptor;
		std::copy_n(words4.begin(), 4, descriptor.fields);
		if (descriptor.Type() != 0) return true; // no buffer: read as a null one
		// Table physical-address U64 atomics have no aligned descriptor prefix.
		// An atomic base at 4 mod 8 therefore lacks the normal path's qword
		// alignment/range semantics. Keep it on the complete normal path until
		// table atomic width and prefix handling are represented explicitly.
		return (!resource.atomic || (descriptor.Base48() & 7u) == 0) &&
		       !descriptor.SwizzleEnabled() && !descriptor.AddTid() &&
		       (!resource.formatted || resource.descriptor_format == Prospero::BufferFormat::kInvalid ||
		        (descriptor.Format() == resource.descriptor_format && descriptor.DstSelXYZW() == resource.descriptor_swizzle));
	};
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program = *pair.programs[st];
		const auto& plan    = program.table_plan;
		auto&       block   = table.blocks[st];
		// The V#s' words and entries first: the entries' lines load together.
		auto& entries = table.entries;
		entries.resize(plan.buffers.size());
		for (size_t i = 0; i < plan.buffers.size(); ++i) {
			auto& [words4, entry] = entries[i];
			for (uint32_t w = 0; w < 4; ++w) {
				const auto& operand = plan.buffers[i][w];
				words4[w] = operand.kind == Kind::Immediate ? operand.value
				            : operand.kind == Kind::Slot    ? block[operand.value]
				            : operand.value - program.user_data_base < user_data[st].size()
				                ? user_data[st][operand.value - program.user_data_base]
				                : 0;
			}
			entry = &table.resolved[XXH3_64bits(words4.data(), sizeof(words4)) & (TableXpr::ResolvedEntries - 1)];
			__builtin_prefetch(entry);
		}
		// The buffers it writes first: a range it also reads through another V# is GPU-written by then, as the
		// normal path's second lookup (RebindBuffers) sees it (not a copy of the guest's bytes in the ring).
		for (size_t n = 0; n < 2 * plan.buffers.size(); ++n) {
			const auto  i         = n < plan.buffers.size() ? n : n - plan.buffers.size();
			const auto& words4    = entries[i].first;
			const auto& resource  = program.info.buffers[i];
			const bool  formatted = resource.formatted;
			if ((resource.written || resource.atomic) != (n < plan.buffers.size())) continue;
			if (!specialized(resource, words4)) {
				if (ShaderRecompiler::IR::RuntimeBufferFormat(resource) && i < 32 && ((pair.portable[st] >> i) & 1u) == 0) {
					pair.portable[st] |= 1u << i;
					pair.respecialize = true;
				}
				return false;
			}
			if (resource.written || resource.atomic) {
				// The shader writes it: marked GPU-written now, whatever an earlier use proved. (A write hazard as
				// HasShaderBufferWrites sees one: a V# of a nonempty range.)
				if (writes != nullptr && resource.written && (uint64_t {words4[0]} | (words4[1] & 0xffffu)) != 0 &&
				    words4[2] != 0)
					*writes = true;
				const auto [address, size] = range_of(words4);
				vk::DeviceAddress device   = null_device;
				if (size != 0) {
					if (address % 4 != 0 || (formatted && textures.HasImageStartingAt(address))) return false;
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, true, formatted);
					region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner)
						return false;
					if (formatted) textures.InvalidateMemoryFromGPU(address, size, "storage-buffer");
					device = owner->BufferDeviceAddress() + offset;
				}
				const auto [word, stride]      = word_of(words4);
				block[plan.BufferDword(i)]     = static_cast<uint32_t>(device);
				block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(device >> 32u);
				block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
				block[plan.BufferDword(i) + 3] = word;
				block[plan.BufferDword(i) + 4] = stride;
				continue;
			}
			auto& held    = *entries[i].second;
			bool  current = held.words == words4 && (held.special == 0 || held.special == table.special_generation);
			if (current && held.size != 0 && held.registration != buffer_cache.RegistrationEpoch()) {
				// Buffers were registered or unregistered since: it holds while none covered its tracker regions.
				for (auto region = held.address / TRACKER_REGION_SIZE, last = (held.address + held.size - 1) / TRACKER_REGION_SIZE;
				     current && region <= last; ++region)
					current = buffer_cache.RegionRegistrationEpoch(region * TRACKER_REGION_SIZE) <= held.registration;
				if (current) held.registration = buffer_cache.RegistrationEpoch();
			}
			if (current && held.size != 0) {
				// Another page of its tracker regions was written: the range itself may still be clean.
				if (const auto epoch = cpu_epoch(held.address, held.size); epoch == 0 || epoch != held.cpu_epoch) {
					LiveCensus::Scope reprove(LiveCensus::DrawPhase, 0x7ab1e, 6);
					current = range_clean(held.address, held.size);
					if (current) held.cpu_epoch = epoch;
				}
			}
			if (!current) {
				LiveCensus::Scope resolve(LiveCensus::DrawPhase, 0x7ab1e, 5);
				// (Kept only once resolved: a refusal leaves nothing a later use could take.)
				held = {};
				TableXpr::ResolvedBuffer fresh {.words = words4, .device = null_device};
				const auto [address, size] = range_of(words4);
				if (formatted || size < requested_of(words4)) fresh.special = table.special_generation;
				if (size != 0) {
					// The shader reads whole dwords from the device address; a buffer an image starts at is made from
					// the image by ObtainBuffer only (SynchronizeBufferFromImage).
					if (address % 4 != 0 || (formatted && textures.HasImageStartingAt(address))) return false;
					const bool cpu_written = buffer_cache.IsRegionCpuModified(address, size) &&
					                         !buffer_cache.IsRegionGpuModified(address, size);
					if (cpu_written && size <= BufferCache::CACHING_PAGESIZE) {
						// As ObtainBuffer: a small range the guest wrote is read from a copy in the ring (this use only,
						// nothing held), not uploaded into its buffer.
						auto [mapped, offset] = upload.Map(size, 16, false);
						if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(address, mapped, size)) {
							upload.Commit();
							const auto [word, stride]      = word_of(words4);
							block[plan.BufferDword(i)]     = static_cast<uint32_t>(upload.BufferDeviceAddress() + offset);
							block[plan.BufferDword(i) + 1] = static_cast<uint32_t>((upload.BufferDeviceAddress() + offset) >> 32u);
							block[plan.BufferDword(i) + 2] = static_cast<uint32_t>(size / 4);
							block[plan.BufferDword(i) + 3] = word;
							block[plan.BufferDword(i) + 4] = stride;
							continue;
						}
					}
					if (cpu_written) buffer_cache.EnsureBufferContents(address, size);
					const auto [owner, offset] = buffer_cache.ObtainBuffer(address, size, false, formatted);
					region_epochs.fill({UINT64_MAX, 0});
					// (The epoch before the dirty check: a write after it moves the epoch.)
					fresh.cpu_epoch = buffer_cache.CpuModificationEpoch(address, size);
					if (owner == nullptr || buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) != owner ||
					    buffer_cache.IsRegionCpuModified(address, size))
						return false;
					fresh.device       = owner->BufferDeviceAddress() + offset;
					fresh.dwords       = static_cast<uint32_t>(std::min<uint64_t>(size / 4, UINT32_MAX));
					fresh.address      = address;
					fresh.size         = size;
					fresh.registration = buffer_cache.RegistrationEpoch(); // (after ObtainBuffer's own registrations)
				}
				held = fresh;
			}
			const auto [word, stride]      = word_of(words4);
			block[plan.BufferDword(i)]     = static_cast<uint32_t>(held.device);
			block[plan.BufferDword(i) + 1] = static_cast<uint32_t>(held.device >> 32u);
			block[plan.BufferDword(i) + 2] = held.dwords;
			block[plan.BufferDword(i) + 3] = word;
			block[plan.BufferDword(i) + 4] = stride;
		}
		if (plan.BlockDwords() != 0) // (a stage with neither slots nor buffers reads no block)
			blocks[st] = upload.BufferDeviceAddress() + upload.Copy(block.data(), block.size() * sizeof(uint32_t), 16);
	}
	return true;
}

// The page table ranges the pair's flat and global loads read made current, as the normal path's PrepareBdaBindings
// does at every use (the read plans' words from the user data and the blocks' slots); an unbounded plan, all of them.
// (Without it, particle simulations at Stonefang read stale pages: readbacks the game crashed on, and indirect
// arguments that lost the device.)
void RenderExecutor::TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data) {
	const auto global = [&](uint32_t st) { return pair.programs[st] != nullptr && pair.programs[st]->table_plan.global_memory; };
	if (!global(0) && !global(1)) return;
	auto&                                               table = *m_table_xpr;
	thread_local ShaderRecompiler::IR::ResourceSnapshot snapshot;
	thread_local std::vector<GuestRange>                ranges, stage_ranges;
	ranges.clear();
	bool bounded = true;
	for (uint32_t st = 0; st < 2 && bounded; ++st) {
		if (!global(st)) continue;
		const auto& program = *pair.programs[st];
		snapshot.user_data.assign(user_data[st].begin(), user_data[st].end());
		snapshot.flattened_srt.assign(table.blocks[st].begin(), table.blocks[st].begin() + program.table_plan.slots.size());
		bounded = ShaderRecompiler::IR::EvaluateBdaReadPlan(program.bda_read_plan, snapshot, stage_ranges);
		ranges.insert(ranges.end(), stage_ranges.begin(), stage_ranges.end());
	}
	auto& resources = m_context.GetGpuResources();
	if (!bounded || !resources.PrepareBdaReadRanges(ranges)) resources.PrepareBda();
}

// A set's images in their layouts (other work may have moved them since the set was last bound), unless a DCC image
// has a fast clear pending (the normal path, whose FindTexture applies it first, takes the draw or dispatch) or the
// draw's targets are not those the set was made for (an image held as a texture became a target, or the reverse).
bool RenderExecutor::TableTransit(const TableImageSet& set, vk::CommandBuffer vk_buffer,
                                  const NativeXprDrawState* draw_state) {
	auto& textures = m_context.GetTextureCache();
	// As CommitBindings': the transitions all precede the one draw or dispatch.
	const TransitGroup transit_group;
	for (const auto& dep: set.images) {
		auto* image = textures.m_slot_images.try_get(dep.id);
		if (image == nullptr) return false;
		// A target of the draw: NativeXprEmitTargets put it in its attachment layout, which the set must have been
		// made for; an image the set holds as a texture must not be one.
		const auto target = draw_state == nullptr
		                        ? decltype(draw_state->targets.end()) {}
		                        : std::ranges::find(draw_state->targets, dep.id, &NativeXprDrawState::Target::id);
		const bool is_target = draw_state != nullptr && target != draw_state->targets.end();
		if (is_target != dep.target || (is_target && (target->layout != dep.layout || target->access != dep.access)))
			return false;
		if (is_target) continue;
		// An image the shader writes is GPU-modified, CPU-clean and current with buffer writes (as the normal path's
		// FindTexture leaves a storage image: it acquires it again otherwise).
		if (dep.storage && !image->SafeToDownload()) return false;
		if (dep.desc.info.metadata.kind == ImageMetadataKind::Dcc) {
			if (image->info.IsVolume()) return false;
			for (uint32_t layer = dep.range.base_layer; layer < dep.range.base_layer + dep.range.layer_count; ++layer)
				if (layer >= 32 || textures.IsMetaCleared(dep.desc.info.metadata.range.address, layer)) return false;
		}
		image->Transit(dep.layout, dep.access, dep.whole ? std::nullopt : std::optional {dep.range}, vk_buffer);
	}
	return true;
}

RenderExecutor::TableResult RenderExecutor::TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
                                                      const NativeXprDrawState& draw_state,
                                                      std::span<const uint64_t> commands, uint64_t index_base,
                                                      uint64_t index_bytes, vk::IndexType index_type,
                                                      const NativeXprDirectDraw* direct) {
	auto&       table  = *m_table_xpr;
	auto&       cache  = *m_native_xpr;
	const auto& sh     = buffer.GetShaders();
	const auto  stages = pair.programs[1] != nullptr ? 2u : 1u;
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {sh.GetVs().gs_user_sgpr.value},
	                                                          std::span<const uint32_t> {sh.GetPs().ps_user_sgpr.value}};
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 1);
	if (!TableEvaluate(pair, user_data)) return TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 2);
	const auto* set = TableSet(buffer, pair, &variant, &draw_state);
	if (set == nullptr) return TableResult::Native;

	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 3);
	// Emission, as NativeXprEmit's: indices and arguments are read by the GPU; lookups may restart the
	// scheduler, so the command buffer handle is taken after them.
	auto&              buffer_cache = m_context.GetBufferCache();
	constexpr uint64_t stride       = sizeof(vk::DrawIndexedIndirectCommand);
	// A tracker region's epoch, read once in the draw (its buffers mostly share one: the line is another core's).
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known = region_epochs[region & 3u];
		if (known.first != region) known = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known.second;
	};
	const auto obtain = [&](TableXpr::HeldBuffer& held, uint64_t address, uint64_t size,
	                        auto&& prepare) -> std::pair<Buffer*, uint64_t> {
		if (held.buffer != nullptr && address >= held.begin && address + size <= held.end &&
		    held.registration == buffer_cache.RegistrationEpoch() && held.cpu_epoch != 0 &&
		    cpu_epoch(held.begin, held.end - held.begin) == held.cpu_epoch)
			return {held.buffer, held.buffer->Offset(address)};
		held = {};
		LiveCensus::Scope miss(LiveCensus::DrawPhase, 0x7ab1e, &held == &table.args ? 10 : 11);
		prepare();
		const auto result = buffer_cache.ObtainBuffer(address, size, false);
		region_epochs.fill({UINT64_MAX, 0}); // (its uploads may have moved them)
		if (result.first != nullptr && buffer_cache.GetBufferIfLive(buffer_cache.FindBuffer(address, size)) == result.first) {
			// The buffer within the tracker region of the address (one region's epoch proves it) when none of it is
			// CPU-dirty, else the range. (The epoch before the dirty check: a write after it moves the epoch.)
			const auto region = address & ~(TRACKER_REGION_SIZE - 1);
			auto       begin  = std::max(result.first->CpuAddress(), region);
			auto       end    = std::min(result.first->CpuAddress() + result.first->Size(), region + TRACKER_REGION_SIZE);
			if (address + size <= end) {
				const auto epoch = buffer_cache.CpuModificationEpoch(begin, end - begin);
				if (buffer_cache.IsRegionCpuModified(begin, end - begin)) {
					begin = address;
					end   = address + size;
				}
				if (epoch != 0 && !buffer_cache.IsRegionCpuModified(begin, end - begin))
					held = {begin, end, epoch, buffer_cache.RegistrationEpoch(), result.first};
			}
		}
		return result;
	};
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes = false;
	// The blocks take each buffer's device address as it resolves: a later lookup that registers a buffer (a merge
	// replaces an earlier one) leaves an earlier address stale, where the normal path binds after all its lookups
	// (RebindBuffers). Such a draw takes the normal path; the next use finds the buffers registered.
	const auto registration = buffer_cache.RegistrationEpoch();
	if (!TableBlocks(pair, user_data, blocks, &writes)) return TableResult::Native;
	TablePrepareBda(pair, user_data);
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 9);
	std::array<std::pair<Buffer*, uint64_t>, 64> arguments;
	std::array<uint32_t, 64>                     group_sizes;
	uint32_t                                     groups = 0;
	for (size_t first = 0; first < commands.size();) {
		size_t count = 1;
		while (first + count < commands.size() && commands[first + count] == commands[first] + count * stride) ++count;
		const auto [owner, offset] = obtain(table.args, commands[first], count * stride, [] {});
		if (owner == nullptr) return TableResult::Native;
		arguments[groups]   = {owner, offset};
		group_sizes[groups] = static_cast<uint32_t>(count);
		++groups;
		first += count;
	}
	auto&      emit       = cache.emit;
	const bool same_index = emit.command == buffer.Handle() && emit.generation == buffer.GraphicsGeneration() &&
	                        emit.index_base == index_base && emit.index_bytes == index_bytes &&
	                        emit.index_type == index_type;
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		indices = obtain(table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u], index_base, index_bytes, [&] {
			if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
			    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
				buffer_cache.EnsureBufferContents(index_base, index_bytes);
		});
		if (indices.first == nullptr) return TableResult::Native;
	}
	if (buffer_cache.RegistrationEpoch() != registration)
		return LiveCounters::Add(LiveCounters::TableRegistered), TableResult::Native;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1e, 4);
	if (buffer.IsInvalid()) return TableResult::Native;
	const auto vk_buffer    = buffer.Handle();
	const bool same_command = emit.command == vk_buffer && emit.generation == buffer.GraphicsGeneration();
	if (!NativeXprEmitTargets(draw_state, vk_buffer)) return TableResult::Native;
	// The set's images in their layouts (a DCC image with a fast clear pending takes the normal path, whose
	// FindTexture applies it first).
	if (!TableTransit(*set, vk_buffer, &draw_state)) return TableResult::Native;
	m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
	if (same_command && emit.state != &draw_state && emit.state != nullptr &&
	    emit.state->dynamic_hash == draw_state.dynamic_hash &&
	    std::memcmp(&emit.state->dynamic, &draw_state.dynamic, sizeof(draw_state.dynamic)) == 0)
		emit.state = &draw_state;
	if (!same_command || emit.state != &draw_state) {
		const auto& state = draw_state.dynamic;
		SetGraphicsDynamicParamsImpl(state.buffer, vk_buffer, state.indexed_viewports, state.colors, state.color_count,
		                             state.depth, VULKAN_HPP_DEFAULT_DISPATCHER);
		if (state.feedback_enabled) vk_buffer.setAttachmentFeedbackLoopEnableEXT(state.feedback_aspects);
	}
	const auto& pipeline = *variant.pipeline;
	if (!same_command || emit.pipeline != &pipeline)
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!same_command || emit.pipeline != &pipeline || table.last_set != set)
		vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1, &set->set, 0,
		                             nullptr);
	// Each stage's user data and the pixel stage's LOD statistics words, where its push data starts.
	ShaderRecompiler::IR::PushData push_data;
	uint32_t                       push_dwords = 0; // the stages' shader data end there: nothing after it is pushed
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& bindings = program.bindings;
		if (!bindings.UsesPushData()) continue;
		push_dwords = std::max(push_dwords, bindings.push_data_start_dword + bindings.ShaderDataDwords());
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[st].size() ? user_data[st][index] : 0;
		}
		if (program.stage == ShaderType::Pixel)
			for (size_t i = 0; i < table.lod_stats.size(); ++i)
				push_data.dwords[bindings.push_data_start_dword + bindings.LodStatsDword() + i] = table.lod_stats[i];
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[st]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[st] >> 32u);
	}
	if (push_dwords != 0)
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                        push_dwords * sizeof(uint32_t), push_data.dwords.data());
	if (!same_index) vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	if (direct != nullptr) {
		// This draw's offsets, as the normal path's ResolveVertexOffset/ResolveInstanceOffset give them: the index
		// offset register, else (a vertex program that fetches) the user data its fetch adds.
		const auto& vertex = *pair.programs[0];
		const auto  word   = [&](int32_t sgpr) {
			const auto index = int64_t {sgpr} - vertex.user_data_base;
			return index >= 0 && index < static_cast<int64_t>(user_data[0].size()) ? user_data[0][index] : 0u;
		};
		const uint32_t index_offset   = buffer.GetUserConfig().GetIndexOffset();
		const uint32_t vertex_offset  = variant.fetch_embedded && index_offset == 0 ? word(vertex.info.vertex_offset_sgpr)
		                                                                            : index_offset;
		const uint32_t first_instance = variant.fetch_embedded ? word(vertex.info.instance_offset_sgpr) : 0u;
		vk_buffer.drawIndexed(direct->index_count, direct->instance_count, 0, static_cast<int32_t>(vertex_offset),
		                      first_instance);
	}
	for (uint32_t group = 0; group < groups; ++group)
		vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
		                              static_cast<uint32_t>(stride));
	// As the normal path's: shaders that write buffers end the rendering, and a barrier orders their writes.
	if (writes) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.command     = vk_buffer;
	emit.generation  = buffer.GraphicsGeneration();
	emit.state       = &draw_state;
	emit.pipeline    = &pipeline;
	emit.index_base  = index_base;
	emit.index_bytes = index_bytes;
	emit.index_type  = index_type;
	table.last_pair    = &pair;
	table.last_variant = static_cast<size_t>(&variant - pair.variants.data());
	table.last_state   = &draw_state;
	table.last_set     = set;
	LiveCounters::Add(LiveCounters::TableDraws, std::max<size_t>(commands.size(), 1));
	return TableResult::Drawn;
}

// The pair's runs (TablePair::Run): the slots its stages evaluate, in reads of consecutive slots, by level.
void RenderExecutor::TableRuns(TablePair& pair) {
	using Kind  = ShaderRecompiler::IR::TablePlan::Operand::Kind;
	auto& table = *m_table_xpr;
	for (uint32_t st = 0; st < 2; ++st) {
		if (pair.programs[st] == nullptr) continue;
		const auto&           plan = pair.programs[st]->table_plan;
		std::vector<uint32_t> levels(plan.slots.size(), 0);
		const auto            same = [](const ShaderRecompiler::IR::TablePlan::Operand& a,
		                     const ShaderRecompiler::IR::TablePlan::Operand& b) {
			return a.kind == b.kind && a.value == b.value;
		};
		pair.runs[st].clear();
		for (uint32_t first = 0; first < plan.slots.size();) {
			if (plan.cpu[first] == 0) {
				++first;
				continue;
			}
			const auto& slot  = plan.slots[first];
			uint32_t    count = 1;
			while (first + count < plan.slots.size() && plan.cpu[first + count] != 0 &&
			       same(plan.slots[first + count].low, slot.low) && same(plan.slots[first + count].high, slot.high) &&
			       same(plan.slots[first + count].offset, slot.offset) &&
			       plan.slots[first + count].immediate == slot.immediate + static_cast<int32_t>(4 * count))
				++count;
			uint32_t level = 0;
			for (const auto* operand: {&slot.low, &slot.high, &slot.offset})
				if (operand->kind == Kind::Slot) level = std::max(level, levels[operand->value] + 1);
			std::fill_n(levels.begin() + first, count, level);
			pair.runs[st].push_back({slot, first, count, level});
			first += count;
		}
		std::ranges::stable_sort(pair.runs[st], {}, &TablePair::Run::level);
		table.addresses.resize(std::max(table.addresses.size(), pair.runs[st].size()));
	}
}

// After a prepared draw (`storable`: an indexed draw without vertex buffers) of a pair the table path asked to
// store: the pair's table mode programs (with this draw's specializations), the pipeline for this draw state,
// compiled by a worker, and the draw state.
void RenderExecutor::TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
                                bool primitive_restart_enable, const RenderState& rendering, bool storable) {
	auto&       table = *m_table_xpr;
	auto* const pair  = std::exchange(table.store, nullptr);
	if (!storable || pair == nullptr) return;
	const auto& sh = buffer.GetShaders();
	// (A request whose packet never reached a prepared draw must not store for another state.)
	if (sh.GetVs().es_regs.data_addr != pair->vs || sh.GetPs().ps_regs.data_addr != pair->ps ||
	    NativeXprStateKey(buffer) != table.store_state)
		return;
	auto&           pipelines   = m_context.GetPipelineCache();
	const auto&     ctx         = buffer.GetRegisters();
	DrawRenderState table_state = state;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t i = 0; i < state.color_count; i++)
		mapping[state.color_info[i].target_slot] = state.color_info[i].export_mapping;
	table_state.programs = pipelines.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
	                                                     buffer.GetUserConfig(), mapping, state.ps_active,
	                                                     table_state.vs_input_info, table_state.ps_input_info, true,
	                                                     pair->portable);
	if (pipelines.TablePending()) {
		pair->waiting = pipelines.TableWaiting(); // (the pair stores again once it is done)
		return;
	}
	const auto* vertex = table_state.vs_input_info.stage.program;
	const auto* pixel  = state.ps_active ? table_state.ps_input_info.stage.program : nullptr;
	const auto  refuse = [&](const char* reason) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
			std::printf("Table: VS 0x%llx PS 0x%llx refused: %s\n", static_cast<unsigned long long>(pair->vs),
			            static_cast<unsigned long long>(pair->ps), reason);
		pair->refused = true;
	};
	if (!table_state.programs.vertex || vertex == nullptr || !vertex->table_mode)
		return refuse("no table mode vertex program");
	if (vertex->stage != ShaderType::Vertex) return refuse("not a vertex program");
	if (state.ps_active && (!table_state.programs.pixel || pixel == nullptr || !pixel->table_mode))
		return refuse("no table mode pixel program");
	const std::array<const ShaderRecompiler::IR::CompiledShaderInfo*, 2> programs {vertex, pixel};
	if (pair->programs[0] == nullptr) {
		pair->programs = programs;
		TableRuns(*pair);
		const auto count = [](const ShaderRecompiler::IR::CompiledShaderInfo* program, bool cpu) -> size_t {
			if (program == nullptr) return 0;
			const auto& plan = program->table_plan;
			return static_cast<size_t>(std::ranges::count(cpu ? plan.cpu : plan.gpu, uint8_t {1}));
		};
		std::printf("Table: VS 0x%llx (0x%016llx) PS 0x%llx (0x%016llx): reads %zu+%zu slots (evaluates %zu+%zu), "
		            "buffers %zu+%zu, images %zu, samplers %zu\n",
		            static_cast<unsigned long long>(pair->vs), static_cast<unsigned long long>(vertex->shader_hash),
		            static_cast<unsigned long long>(pair->ps),
		            static_cast<unsigned long long>(pixel != nullptr ? pixel->shader_hash : 0), count(vertex, false),
		            count(pixel, false), count(vertex, true), count(pixel, true), vertex->table_plan.buffers.size(),
		            pixel != nullptr ? pixel->table_plan.buffers.size() : 0,
		            vertex->table_plan.images.size() + (pixel != nullptr ? pixel->table_plan.images.size() : 0),
		            vertex->table_plan.samplers.size() + (pixel != nullptr ? pixel->table_plan.samplers.size() : 0));
	} else if (pair->programs != programs) {
		// Another specialization (the pixel program's export of another draw state's targets): this state's draws
		// keep the native path.
		const auto found = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
		if (found == pair->variants.end()) pair->variants.push_back({.state = table.store_state, .refused = true});
		return;
	}
	const auto* pipeline = pipelines.TryCreateGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, table_state.vs_input_info, buffer,
	    state.ps_active ? &table_state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    table_state.programs.vertex, table_state.programs.pixel, true);
	if (pipeline == nullptr) pair->waiting = pipelines.PipelineWaiting();
	auto variant = std::ranges::find(pair->variants, table.store_state, &TableXpr::Variant::state);
	if (variant == pair->variants.end()) variant = pair->variants.insert(variant, {.state = table.store_state});
	variant->pipeline       = pipeline;
	variant->mapping        = mapping;
	variant->fetch_embedded = table_state.vs_input_info.fetch_embedded;
	NativeXprCaptureState(buffer, state, rendering, table.store_state);
}

// --- Table dispatches ---------------------------------------------------------------------------------------------
//
// KYTY_TABLE_DISPATCH=1: a compute shader the normal path dispatched twice (its resource tables evaluated in full),
// whose program allows table mode, is dispatched as table draws are drawn: its slots in a block, its read-only V#s
// resolved once, its written ones obtained as written at every dispatch (the buffer cache tracks the GPU's writes),
// its images and samplers in a set; the barriers and the compute chain as the normal path's DispatchDirect. Shaders
// a special path of the normal path took (linear copies, clears), thread-dimension dispatches, frame capture and
// frame generation stay on the normal path.

// The compute stage's registers that select the program (ShaderGetStaticInputInfoCS reads them).
static uint64_t TableDispatchKey(const HW::ComputeShaderInfo& cs) {
	const auto& r = cs.cs_regs;
	const std::array<uint64_t, 4> key {r.data_addr, uint64_t {r.num_thread_x} | uint64_t {r.num_thread_y} << 32u,
	                                   uint64_t {r.num_thread_z} | uint64_t {r.lds_size} << 32u,
	                                   uint64_t {r.wave_size} | uint64_t {r.user_sgpr} << 8u |
	                                       uint64_t {r.tidig_comp_cnt} << 16u | uint64_t {r.tgid_x_en} << 24u |
	                                       uint64_t {r.tgid_y_en} << 25u | uint64_t {r.tgid_z_en} << 26u |
	                                       uint64_t {r.tg_size_en} << 27u | uint64_t {r.scratch_en} << 28u};
	return XXH3_64bits(key.data(), sizeof(key));
}

// The normal path dispatched (`special`: one of its special paths took the dispatch).
void RenderExecutor::TableDispatchSeen(const HW::ComputeShaderInfo& cs, bool special) {
	if (kyty_local_table_dispatch_mode.load(std::memory_order_relaxed) == 0) return;
	if (!m_table_xpr) m_table_xpr = std::make_shared<TableXpr>();
	auto& pair = m_table_xpr->dispatches[TableDispatchKey(cs)];
	pair.vs    = cs.cs_regs.data_addr;
	if (special) pair.refused = true;
	++pair.sightings;
}

bool RenderExecutor::TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                   uint64_t indirect_args) {
	if (!m_table_xpr || FrameCapture::Active() || XprCapture::Enabled() || FrameGen::Enabled()) return false;
	auto&       table = *m_table_xpr;
	const auto& sh    = buffer.GetShaders();
	const auto& cs    = sh.GetCs();
	const auto  found = table.dispatches.find(TableDispatchKey(cs));
	if (found == table.dispatches.end() || found->second.refused || found->second.sightings < 2 ||
	    found->second.vs != cs.cs_regs.data_addr)
		return false;
	auto& pair = found->second;
	LiveCensus::Scope census(LiveCensus::DrawPhase, 0x7ab1f, 0); // census: the dispatches' phases (b)
	if (const auto generation = ShaderMapGeneration(); generation != pair.generation) {
		if (pair.programs[0] != nullptr && ShaderHashAt(pair.vs) != pair.programs[0]->shader_hash) {
			pair.portable = {};
			TableReset(pair);
		}
		pair.generation = generation;
	}
	if (pair.respecialize) TableReset(pair);
	if (pair.waiting) {
		if (!pair.waiting->load(std::memory_order_acquire)) return false;
		pair.waiting.reset();
	}
	auto& pipelines = m_context.GetPipelineCache();
	if (pair.programs[0] == nullptr) {
		// The table mode program of the shader the normal path prepared.
		pair.compute = std::make_unique<ShaderComputeInputInfo>();
		ResetNativeStageInput(*pair.compute);
		pair.compute_program = pipelines.GetComputeProgram(cs, buffer.GetRegisters().GetShaderRegisters(), *pair.compute,
		                                                   nullptr, true, pair.portable[0]);
		const auto* program = pair.compute->stage.program;
		if (pipelines.TablePending()) {
			pair.waiting = pipelines.TableWaiting(); // (asked again once it is done)
			return false;
		}
		if (!pair.compute_program || program == nullptr || !program->table_mode) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
				std::printf("Table: CS 0x%llx refused: no table mode program\n", static_cast<unsigned long long>(pair.vs));
			pair.refused = true;
			return false;
		}
		pair.programs[0] = program;
		TableRuns(pair);
		std::printf("Table: CS 0x%llx (0x%016llx): reads %zu slots (evaluates %zu), buffers %zu (written %zu), "
		            "images %zu, samplers %zu\n",
		            static_cast<unsigned long long>(pair.vs), static_cast<unsigned long long>(program->shader_hash),
		            static_cast<size_t>(std::ranges::count(program->table_plan.gpu, uint8_t {1})),
		            static_cast<size_t>(std::ranges::count(program->table_plan.cpu, uint8_t {1})),
		            program->table_plan.buffers.size(),
		            static_cast<size_t>(std::ranges::count_if(program->info.buffers, [](const auto& item) {
			            return item.written || item.atomic;
		            })),
		            program->table_plan.images.size(), program->table_plan.samplers.size());
	}
	// Restrict the opt-in experiment without omitting work. A rejected shader
	// falls through to DispatchDirect, including all normal special paths.
	// Change the two exact-hash words only while TABLE_DISPATCH is disabled;
	// a live reader must not observe a partially updated selection.
	const auto shader_hash = pair.programs[0]->shader_hash;
	const auto bucket = static_cast<uint32_t>((shader_hash ^ (shader_hash >> 32u)) & 31u);
	const auto selected_hash = uint64_t {kyty_local_table_dispatch_hash_lo.load(std::memory_order_relaxed)} |
	                           uint64_t {kyty_local_table_dispatch_hash_hi.load(std::memory_order_relaxed)} << 32u;
	if ((kyty_local_table_dispatch_buckets.load(std::memory_order_relaxed) & (uint32_t {1} << bucket)) == 0 ||
	    (selected_hash != 0 && selected_hash != shader_hash))
		return false;
	if (pair.pipeline == nullptr) {
		pair.pipeline = pipelines.TryCreateComputePipeline(*pair.compute, pair.compute_program);
		if (pair.pipeline == nullptr) {
			pair.waiting = pipelines.PipelineWaiting();
			return false;
		}
	}
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {cs.cs_user_sgpr.value},
	                                                          std::span<const uint32_t> {}};
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 1);
	if (!TableEvaluate(pair, user_data)) return false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 2);
	const auto* set = TableSet(buffer, pair, nullptr, nullptr);
	if (set == nullptr) return false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 3);
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             storage_writes = false;
	auto&                            cache          = m_context.GetBufferCache();
	const auto                       registration   = cache.RegistrationEpoch(); // (as TableDraw's)
	if (!TableBlocks(pair, user_data, blocks, &storage_writes)) return false;
	TablePrepareBda(pair, user_data);
	vk::Buffer indirect_buffer = nullptr;
	uint64_t   indirect_offset = 0;
	if (indirect_args != 0) {
		// As DispatchDirect's: the arguments' owner, made current.
		cache.EnsureBufferContents(indirect_args, 3u * sizeof(uint32_t));
		auto& owner     = cache.GetBuffer(cache.FindBuffer(indirect_args, 3u * sizeof(uint32_t)));
		indirect_buffer = owner.Handle();
		indirect_offset = owner.Offset(indirect_args);
	}
	if (cache.RegistrationEpoch() != registration) return LiveCounters::Add(LiveCounters::TableRegistered), false;
	phase.reset();
	phase.emplace(LiveCensus::DrawPhase, 0x7ab1f, 4);
	// End a normal dispatch's pending chain before emitting a table dispatch.
	// TableBlocks does not produce the ChainAccess proof DispatchDirect uses;
	// ChainHandle here would silently bypass its dependency checks (and even
	// KYTY_DS_COMPUTE_CHAIN=0). Keep full ordering until table accesses can be
	// proven too. This path remains opt-in through KYTY_TABLE_DISPATCH.
	buffer.EndRendering();
	const auto vk_buffer = buffer.Handle();
	if (!TableTransit(*set, vk_buffer, nullptr)) return false;
	const auto& pipeline = *pair.pipeline;
	const auto& program  = *pair.programs[0];
	storage_writes = storage_writes || std::ranges::any_of(program.info.images, [](const auto& image) {
		return image.written && image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	});
	const auto& bindings = program.bindings;
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, 0, 1, &set->set, 0, nullptr);
	if (bindings.UsesPushData()) {
		ShaderRecompiler::IR::PushData push_data;
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[0].size() ? user_data[0][index] : 0;
		}
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[0]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[0] >> 32u);
		vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                        (bindings.push_data_start_dword + bindings.ShaderDataDwords()) * sizeof(uint32_t),
		                        push_data.dwords.data());
	}
	if (storage_writes) ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (indirect_args != 0) {
		vk::BufferMemoryBarrier args_barrier {};
		args_barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite |
		                                   vk::AccessFlagBits::eMemoryWrite;
		args_barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead;
		args_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.buffer              = indirect_buffer;
		args_barrier.offset              = indirect_offset;
		args_barrier.size                = 3u * sizeof(uint32_t);
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eDrawIndirect,
		                          vk::DependencyFlags {}, 0, nullptr, 1, &args_barrier, 0, nullptr);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
	} else {
		vk_buffer.dispatch(groups_x, groups_y, groups_z);
	}
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	m_context.GetCommandScheduler().CompleteDispatch();
	// (A later table draw binds its pipeline and set again: the graphics state is untouched.)
	LiveCounters::Add(LiveCounters::TableDispatches);
	return true;
}
