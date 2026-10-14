#include "loader/demonsSoulsRenderSize.h"

#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace Loader::DemonsSoulsRenderSize {
namespace {

bool Readable(uint64_t address, uint64_t size) {
	return Libs::Graphics::HostMemoryRangeIsReadable(address, size);
}

bool HoldsPair(uint64_t address, uint32_t width, uint32_t height) {
	uint32_t pair[2] {};
	if (!Readable(address, sizeof(pair))) return false;
	std::memcpy(pair, reinterpret_cast<const void*>(address), sizeof(pair));
	return pair[0] == width && pair[1] == height;
}

// This build (USA PPSA01342 player) has no guest code pattern finder: the EU builds' mode case is not looked for.
std::optional<uint64_t> FindRenderConstant(const Program& /*program*/) {
	return std::nullopt;
}

// The one place `bytes` is in the program's loaded segments.
std::optional<uint64_t> FindUniqueData(const Program& program, std::span<const uint8_t> bytes) {
	const auto* elf = program.elf.get();
	if (elf == nullptr || elf->GetEhdr() == nullptr || elf->GetPhdr() == nullptr) return std::nullopt;
	std::vector<uint64_t> found;
	const auto*           phdr = elf->GetPhdr();
	for (size_t i = 0; i < elf->GetEhdr()->e_phnum && found.size() < 2; i++) {
		if (phdr[i].p_type != PT_LOAD || phdr[i].p_filesz < bytes.size()) continue;
		const auto address = program.base_vaddr + phdr[i].p_vaddr;
		if (!Readable(address, phdr[i].p_filesz)) continue;
		const auto* begin = reinterpret_cast<const uint8_t*>(address);
		const auto* end   = begin + phdr[i].p_filesz;
		for (const auto* at = std::search(begin, end, bytes.begin(), bytes.end()); at != end && found.size() < 2;
		     at             = std::search(at + 1, end, bytes.begin(), bytes.end()))
			found.push_back(reinterpret_cast<uint64_t>(at));
	}
	return found.size() == 1 ? std::optional(found[0]) : std::nullopt;
}

// PPSA01342 01.004.000 has no mode case of the builds above: its video settings block (eboot+0x20010a0) holds the
// render size at +0x4 and the output size at +0xc; the performance mode case loads the block in one 32-byte move
// (eboot+0x9454b1). Its mode setter (eboot+0x9449b0) reads the render size from per-mode tables of four modes, not the
// five above (widths eboot+0x1ffbe20, heights +0x1ffd750; the same performance mode index): with the block alone
// patched, the 3D rendered at two sizes (a broken picture, the player not seen, 2026-10-14). Both checked by value.
constexpr uint64_t UsaRenderPair = 0x20010a4, UsaOutputPair = 0x20010ac;
inline constexpr uint32_t UsaModeWidths[4] {1600, 1920, 2560, 3840};
inline constexpr uint32_t UsaModeHeights[4] {900, 1080, 1440, 2160};

} // namespace

void Install(Program* program) {
	const char* setting = std::getenv("KYTY_RENDER_HEIGHT");
	if (setting == nullptr || *setting == 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto requested = static_cast<uint32_t>(std::strtoul(setting, nullptr, 10));
	if (requested == RenderHeight) return;
	const auto size = RenderSizeFor(requested);
	if (size.height == 0) {
		std::printf("Demon's Souls render size: KYTY_RENDER_HEIGHT=%s is not 720-1440; the game's 2560x1440 kept\n", setting);
		return;
	}
	const auto constant = FindRenderConstant(*program);
	const auto widths   = FindUniqueData(*program, std::span(reinterpret_cast<const uint8_t*>(ModeWidths), sizeof(ModeWidths)));
	const auto heights  = FindUniqueData(*program, std::span(reinterpret_cast<const uint8_t*>(ModeHeights), sizeof(ModeHeights)));
	if (!constant && program->mapped_size >= UsaOutputPair + 8) {
		const auto render = program->base_vaddr + UsaRenderPair, output = program->base_vaddr + UsaOutputPair;
		const auto usa_widths =
		    FindUniqueData(*program, std::span(reinterpret_cast<const uint8_t*>(UsaModeWidths), sizeof(UsaModeWidths)));
		const auto usa_heights =
		    FindUniqueData(*program, std::span(reinterpret_cast<const uint8_t*>(UsaModeHeights), sizeof(UsaModeHeights)));
		if (Readable(render, 8) && Readable(output, 8) && HoldsPair(render, RenderWidth, RenderHeight) &&
		    HoldsPair(output, OutputWidth, OutputHeight) && usa_widths && usa_heights) {
			const uint32_t pair[2] {size.width, size.height};
			std::memcpy(reinterpret_cast<void*>(render), pair, sizeof(pair));
			std::memcpy(reinterpret_cast<void*>(*usa_widths + PerformanceMode * sizeof(uint32_t)), &size.width, sizeof(uint32_t));
			std::memcpy(reinterpret_cast<void*>(*usa_heights + PerformanceMode * sizeof(uint32_t)), &size.height,
			            sizeof(uint32_t));
			std::printf("Demon's Souls render size: performance mode renders %ux%u instead of %ux%u (output %ux%u, 01.004.000)\n",
			            size.width, size.height, RenderWidth, RenderHeight, OutputWidth, OutputHeight);
			return;
		}
	}
	if (!constant || !widths || !heights) {
		std::printf("Demon's Souls render size: the video settings of a known build not found; the game's 2560x1440 kept\n");
		return;
	}
	// Installation occurs before module initializers or guest threads run; program memory is writable to the host.
	const uint32_t pair[2] {size.width, size.height};
	std::memcpy(reinterpret_cast<void*>(*constant), pair, sizeof(pair));
	std::memcpy(reinterpret_cast<void*>(*widths + PerformanceMode * sizeof(uint32_t)), &size.width, sizeof(uint32_t));
	std::memcpy(reinterpret_cast<void*>(*heights + PerformanceMode * sizeof(uint32_t)), &size.height, sizeof(uint32_t));
	std::printf("Demon's Souls render size: performance mode renders %ux%u instead of %ux%u (output %ux%u)\n", size.width,
	            size.height, RenderWidth, RenderHeight, OutputWidth, OutputHeight);
}

} // namespace Loader::DemonsSoulsRenderSize
