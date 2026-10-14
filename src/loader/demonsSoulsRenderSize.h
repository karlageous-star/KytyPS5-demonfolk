#pragma once

#include <cstdint>
#include <string_view>

namespace Loader {
struct Program;
// KYTY_RENDER_HEIGHT=<720..1440> (run-windows.ps1 -RenderHeight, the launcher's "Render resolution"): Demon's Souls'
// performance mode - the PS5's default, its 3D rendered at 2560x1440 and upscaled by the game to a 3840x2160 output -
// renders its 3D at that height (16:9) instead. The output, the UI and what the game draws at output size stay 4K;
// GPU-bound PCs (a laptop's RTX 3080: 1-1 standing 20.7 -> 24.5 fps at 1080) trade sharpness for frame rate.
// Unset or 1440: the game as it is. The game's own 1080p output modes (r_Wants4K 1 and 2) are no option: their UI
// keeps the 4K layout (off screen).
namespace DemonsSoulsRenderSize {
void Install(Program* program);

// The video settings' mode 3 case (the setter's and the constructor's): the output size, then the render size, as
// RIP-relative pairs of 32-bit width and height, then a short jump; every build seen (01.005.000, 01.007.000) has it.
inline constexpr std::string_view ModeCase = "c5 fb 10 0d ?? ?? ?? ?? c5 fb 10 05 ?? ?? ?? ?? eb";
inline constexpr uint32_t         OutputWidth = 3840, OutputHeight = 2160, RenderWidth = 2560, RenderHeight = 1440;
// The render size query's per-mode tables (modes 1-5): widths, heights.
inline constexpr uint32_t ModeWidths[5] {1600, 1920, 2560, 3840, 2688};
inline constexpr uint32_t ModeHeights[5] {900, 1080, 1440, 2160, 1512};
inline constexpr int      PerformanceMode = 2; // (index in the tables)

// The 16:9 render size for a height, its width a multiple of 8; {0, 0} for a height out of range.
struct Size {
	uint32_t width  = 0;
	uint32_t height = 0;
};
inline Size RenderSizeFor(uint32_t height) {
	if (height < 720 || height > RenderHeight) return {};
	const uint32_t width = (height * 16u / 9u + 4u) / 8u * 8u;
	return {width, height};
}
} // namespace DemonsSoulsRenderSize
} // namespace Loader
