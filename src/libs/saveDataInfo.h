#ifndef EMULATOR_SRC_LIBS_SAVEDATAINFO_H_
#define EMULATOR_SRC_LIBS_SAVEDATAINFO_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace Libs::SaveData {

// What the system's save data list shows of a slot: the texts the game last gave it (SaveDataSetParam; empty for a
// slot no session of this emulator has saved since they were kept) and its modification time (0: no such slot).
struct SaveSlotInfo {
	std::string title;
	std::string sub_title;
	std::string detail;
	int64_t     mtime = 0;
};

SaveSlotInfo GetSaveSlotInfo(std::string_view dir_name);

} // namespace Libs::SaveData

#endif /* EMULATOR_SRC_LIBS_SAVEDATAINFO_H_ */
