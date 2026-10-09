// Included inside DemonsSoulsIdle's private namespace, after Readable().
#if defined(__x86_64__) || defined(_M_X64)
uint64_t collision_guard_cave = 0;
const Program* collision_guard_owner = nullptr;

void InstallCollisionGuard(Program* program) {
	const char* enabled = std::getenv("KYTY_DS_DETACHED_UNLINK_GUARD");
	if (enabled == nullptr || std::strcmp(enabled, "1") != 0 || collision_guard_cave != 0 ||
	    program == nullptr || program->file_name.filename() != "eboot.bin" ||
	    !Libs::Graphics::DemonsSouls::IsSupportedGame()) return;
	std::string title, version;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title) || title != "PPSA01342" ||
	    !Loader::SystemContentParamSfoGetString("APP_VER", &version) || version != "01.004.000") return;
	namespace Guard = Loader::DemonsSoulsCollisionGuard;
	if (program->mapped_size < Guard::FunctionOffset + Guard::FunctionSize ||
	    program->mapped_size > Guard::CaveOffset ||
	    program->base_vaddr > UINT64_MAX - Guard::CaveOffset - PageSize) return;
	const auto function = program->base_vaddr + Guard::FunctionOffset;
	if (!Readable(function, Guard::FunctionSize) ||
	    !Guard::Matches(std::span(reinterpret_cast<const uint8_t*>(function), Guard::FunctionSize))) {
		std::printf("Demon's Souls collision guard: code signature differs; retaining guest code\n");
		return;
	}
	const auto patch_site = program->base_vaddr + Guard::SiteOffset;
	const auto requested = program->base_vaddr + Guard::CaveOffset;
	std::array<uint8_t, Guard::Original.size()> jump {};
	if (!Guard::MakeJump(patch_site, requested, jump)) return;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_collision_guard", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	Xbyak::ClearError();
	Xbyak::CodeGenerator code(PageSize, reinterpret_cast<void*>(allocated));
	Guard::EmitThunk(code, reinterpret_cast<const void*>(program->base_vaddr + Guard::NormalOffset),
	                 reinterpret_cast<const void*>(program->base_vaddr + Guard::DetachedOffset),
	                 &Guard::g_detached_unlinks);
	code.ready();
	if (Xbyak::GetError() ||
	    !Common::VirtualMemory::FlushInstructionCache(allocated, code.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(
	        allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	// RuntimeLinker installs this before initializers/guest workers start. The
	// allocation and owner remain alive until process exit (see lifetime guards).
	std::memcpy(reinterpret_cast<void*>(patch_site), jump.data(), jump.size());
	if (!Common::VirtualMemory::FlushInstructionCache(patch_site, jump.size())) {
		std::memcpy(reinterpret_cast<void*>(patch_site), Guard::Original.data(), Guard::Original.size());
		Common::VirtualMemory::FlushInstructionCache(patch_site, jump.size());
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	collision_guard_owner = program;
	collision_guard_cave = allocated;
	std::printf("Demon's Souls collision guard: installed opt-in detached end-node guard at eboot+0xe762cd\n");
}
#endif
