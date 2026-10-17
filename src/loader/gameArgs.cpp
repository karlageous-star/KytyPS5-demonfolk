#include "loader/gameArgs.h"

#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/systemContent.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace Loader::GameArgs {
namespace {

// The "name=value" entries of a list separated by ';' or ',', each trimmed of its spaces.
std::vector<std::string> Entries(const std::string& list) {
	std::vector<std::string> out;
	for (size_t at = 0; at < list.size();) {
		auto end = list.find_first_of(";,", at);
		if (end == std::string::npos) end = list.size();
		const auto first = list.find_first_not_of(' ', at);
		const auto last  = list.find_last_not_of(' ', end - 1);
		if (first != std::string::npos && first < end && last >= first) out.push_back(list.substr(first, last - first + 1));
		at = end + 1;
	}
	return out;
}

// The name of an entry: what is before its '=', without the spaces before the '='.
std::string NameOf(const std::string& entry) {
	auto       name = entry.substr(0, entry.find('='));
	const auto last = name.find_last_not_of(' ');
	return last == std::string::npos ? std::string() : name.substr(0, last + 1);
}

// KYTY_GRAPHICS_PRESET=medium|low|lowest|high (unset: low, the owner's default since 10-18; high: the game's own
// settings, nothing changes). Demon's Souls' own console variables for
// what a frame draws. Low and Lowest are the Graphics Low and Lowest presets of the chenxiao launcher: the level of
// detail as for a 720 or 480-line picture (r_forceViewportHeightForPixelSizeCalc), sun shadows to 125 or 60 m
// (sunShadowMaxDistance, 250 by default), and Lowest also without ambient shadowing or spot light shadows. Medium is
// between the game's settings and Low: detail as for 900 lines, sun shadows to 180 m. r_forceShadowLOD is never set:
// it draws objects without their shadow meshes, so they vanish (the cart and the barricades at the Gates of Boletaria).
struct Preset {
	const char* name;
	const char* convars;
};
constexpr Preset presets[] = {
	{"medium", "r_forceViewportHeightForPixelSizeCalc=900;sunShadowMaxDistance=180"},
	{"low", "r_forceViewportHeightForPixelSizeCalc=720;sunShadowMaxDistance=125"},
	{"lowest", "r_forceViewportHeightForPixelSizeCalc=480;sunShadowMaxDistance=60;"
	           "r_enableAmbientShadowing=false;r_enableSpotShadows=false"},
};

// The entries of the chosen preset (low when unset; none for high, or when the game is not Demon's Souls). Called once, from
// ExtraLines, so the choice is logged once.
std::vector<std::string> PresetEntries() {
	const char* env  = std::getenv("KYTY_GRAPHICS_PRESET");
	const char* name = env == nullptr || *env == '\0' ? "low" : env;
	if (std::strcmp(name, "high") == 0) return {};
	const Preset* chosen = nullptr;
	for (const auto& preset: presets) {
		if (std::strcmp(preset.name, name) == 0) chosen = &preset;
	}
	if (chosen == nullptr) {
		std::printf("Graphics preset %s is not medium, low, lowest or high: nothing changed\n", name);
		return {};
	}
	std::string title;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title) ||
	    !Libs::Graphics::DemonsSouls::IsSupportedTitle(title)) {
		std::printf("Graphics preset %s is for Demon's Souls only: nothing changed\n", name);
		return {};
	}
	std::printf("Graphics preset: %s (%s)\n", name, chosen->convars);
	return Entries(chosen->convars);
}

// One "+name=value" line per entry of KYTY_GAME_CONVARS, then the preset's entries that KYTY_GAME_CONVARS does not
// set for the same name (its own entries win).
const std::string& ExtraLines() {
	static const std::string lines = [] {
		const char* value = std::getenv("KYTY_GAME_CONVARS");
		const auto  user  = Entries(value != nullptr ? value : "");
		std::string out;
		for (const auto& entry: user) out += "+" + entry + "\n";
		for (const auto& entry: PresetEntries()) {
			bool set = false;
			for (const auto& own: user) set = set || NameOf(own) == NameOf(entry);
			if (!set) out += "+" + entry + "\n";
		}
		return out;
	}();
	return lines;
}

// The engine opens it as "$/packagecmdlineargs.txt" (lower case): the name is compared without case.
bool IsArgsFile(const std::filesystem::path& real) {
	if (ExtraLines().empty()) return false;
	auto name = real.filename().string();
	for (auto& c: name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return name == "packagecmdlineargs.txt";
}

std::string Contents(const std::filesystem::path& real) {
	std::ifstream in(real, std::ios::binary);
	std::string   text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (!text.empty() && text.back() != '\n') text += '\n';
	return text + ExtraLines();
}

} // namespace

std::filesystem::path RedirectRead(const std::filesystem::path& real) {
	if (!IsArgsFile(real)) return {};
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	// The copy goes next to the temporary files; one that cannot be written leaves the file as it is.
	std::error_code error;
	const auto      copy = std::filesystem::temp_directory_path(error) / "kyty-PackageCmdLineArgs.txt";
	if (error) return {};
	const auto    text = Contents(real);
	std::ofstream out(copy, std::ios::binary | std::ios::trunc);
	out.write(text.data(), static_cast<std::streamsize>(text.size()));
	out.close();
	if (!out) return {};
	std::printf("Game args: %s read with\n%s", real.string().c_str(), ExtraLines().c_str());
	std::fflush(stdout);
	return copy;
}

std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real) {
	if (!IsArgsFile(real)) return std::nullopt;
	return Contents(real).size();
}

} // namespace Loader::GameArgs
