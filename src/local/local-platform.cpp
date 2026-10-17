#include "local-platform.h"

#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <string>
#else
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace LocalPlatform {

// "1,2,3" -> bit mask of CPUs 0..63; 0 for an empty or invalid list.
static uint64_t CpuListMask(const char* list) {
	uint64_t mask = 0;
	if (list == nullptr) return 0;
	for (const char* p = list; *p != '\0';) {
		char*      end = nullptr;
		const long cpu = std::strtol(p, &end, 10);
		if (end == p) break;
		if (cpu >= 0 && cpu < 64) mask |= uint64_t(1) << cpu;
		p = (*end == ',') ? end + 1 : end;
	}
	return mask;
}

#if defined(_WIN32)

static double FileTimeSeconds(const FILETIME& time) {
	const uint64_t ticks = (uint64_t(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
	return static_cast<double>(ticks) / 1e7;
}

static double HandleCpuSeconds(HANDLE thread) {
	FILETIME creation {}, exit {}, kernel {}, user {};
	if (GetThreadTimes(thread, &creation, &exit, &kernel, &user) == 0) return 0;
	return FileTimeSeconds(kernel) + FileTimeSeconds(user);
}

void SetThreadName(const char* name) {
	wchar_t wide[64] {};
	for (size_t i = 0; name[i] != '\0' && i + 1 < std::size(wide); ++i) wide[i] = static_cast<wchar_t>(name[i]);
	(void)SetThreadDescription(GetCurrentThread(), wide);
}

void ReleaseRedirectedConsole() {
	if (GetConsoleWindow() != nullptr && GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_DISK &&
	    GetFileType(GetStdHandle(STD_ERROR_HANDLE)) == FILE_TYPE_DISK)
		(void)FreeConsole();
}

uint32_t ThreadId() {
	return static_cast<uint32_t>(GetCurrentThreadId());
}

void PinThreadToCpuList(const char* list) {
	if (const auto mask = CpuListMask(list); mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(mask));
}

bool PinThreadHandleToCpuList(uint64_t handle, const char* list) {
	const auto mask = CpuListMask(list);
	return handle != 0 && mask != 0 &&
	       SetThreadAffinityMask(reinterpret_cast<HANDLE>(handle), DWORD_PTR(mask)) != 0;
}

// CPU set ids of the logical processors in the mask (all of them for mask 0 = no restriction).
static bool SetProcessDefaultCpuMask(uint64_t mask) {
	if (mask == 0) return SetProcessDefaultCpuSets(GetCurrentProcess(), nullptr, 0) != 0;
	ULONG length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(length);
	auto* first = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
	if (length == 0 || !GetSystemCpuSetInformation(first, length, &length, GetCurrentProcess(), 0)) return false;
	std::vector<ULONG> ids;
	for (ULONG offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 && info->CpuSet.LogicalProcessorIndex < 64 &&
		    (mask >> info->CpuSet.LogicalProcessorIndex) & 1u)
			ids.push_back(info->CpuSet.Id);
		offset += info->Size;
	}
	return !ids.empty() && SetProcessDefaultCpuSets(GetCurrentProcess(), ids.data(), static_cast<ULONG>(ids.size())) != 0;
}

bool SetProcessDefaultCpuList(const char* list) {
	return SetProcessDefaultCpuMask(CpuListMask(list));
}

uint64_t CurrentThreadHandle() {
	HANDLE handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_SET_INFORMATION | THREAD_SUSPEND_RESUME |
	                               THREAD_GET_CONTEXT,
	                           FALSE, GetCurrentThreadId());
	return reinterpret_cast<uint64_t>(handle);
}

uint64_t OpenThreadForSampling(uint32_t thread_id) {
	HANDLE handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_SET_INFORMATION | THREAD_SUSPEND_RESUME |
	                               THREAD_GET_CONTEXT,
	                           FALSE, thread_id);
	return reinterpret_cast<uint64_t>(handle);
}

void CloseThreadForSampling(uint64_t handle) {
	if (handle != 0) CloseHandle(reinterpret_cast<HANDLE>(handle));
}

// Unwind tables of the loaded images, collected before sampling: the sampler must not take
// loader or function-table locks (RtlLookupFunctionEntry does) while the sampled thread,
// which may hold them, is suspended. RtlVirtualUnwind itself takes no lock.
namespace {
struct UnwindImage {
	uint64_t                base = 0, end = 0;
	const RUNTIME_FUNCTION* table = nullptr;
	size_t                  count = 0;
};
std::vector<UnwindImage> g_unwind_images;

const RUNTIME_FUNCTION* FindUnwindEntry(uint64_t pc, uint64_t* image_base) {
	for (const auto& image: g_unwind_images) {
		if (pc < image.base || pc >= image.end) continue;
		const auto rva = static_cast<DWORD>(pc - image.base);
		size_t     lo = 0, hi = image.count;
		while (lo < hi) {
			const size_t mid = (lo + hi) / 2;
			if (image.table[mid].EndAddress <= rva) lo = mid + 1;
			else hi = mid;
		}
		if (lo == image.count || image.table[lo].BeginAddress > rva) return nullptr;
		*image_base = image.base;
		// Chained entries (UNW_FLAG_CHAININFO) are resolved by RtlVirtualUnwind.
		return &image.table[lo];
	}
	return nullptr;
}
} // namespace

void PrepareSampling() {
	g_unwind_images.clear();
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
	if (snapshot == INVALID_HANDLE_VALUE) return;
	MODULEENTRY32W entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Module32FirstW(snapshot, &entry); ok; ok = Module32NextW(snapshot, &entry)) {
		const auto base = reinterpret_cast<uint64_t>(entry.modBaseAddr);
		const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
		if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
		if (nt->Signature != IMAGE_NT_SIGNATURE ||
		    nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_EXCEPTION)
			continue;
		const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
		if (directory.VirtualAddress == 0 || directory.Size < sizeof(RUNTIME_FUNCTION)) continue;
		g_unwind_images.push_back({base, base + entry.modBaseSize,
		                           reinterpret_cast<const RUNTIME_FUNCTION*>(base + directory.VirtualAddress),
		                           directory.Size / sizeof(RUNTIME_FUNCTION)});
	}
	CloseHandle(snapshot);
}

bool SampleThread(uint64_t handle, uint64_t stack_low, uint64_t stack_high, uint64_t* out, size_t words) {
	auto* thread = reinterpret_cast<HANDLE>(handle);
	if (thread == nullptr || words < 2 || SuspendThread(thread) == static_cast<DWORD>(-1)) return false;
	CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	const bool got       = GetThreadContext(thread, &context) != 0;
	if (got && stack_high == 0) {
		// Unknown stack (a guest thread on its own stack): walk only inside the committed
		// region that holds rsp, so no read can fault while the thread is suspended.
		MEMORY_BASIC_INFORMATION region {};
		if (VirtualQuery(reinterpret_cast<const void*>(context.Rsp), &region, sizeof(region)) != 0 &&
		    region.State == MEM_COMMIT) {
			stack_low  = context.Rsp;
			stack_high = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
		}
	}
	if (got) {
		// Layout of the Linux SIGPROF sampler: pc, then callers. With unwind tables
		// (PrepareSampling) the callers come from RtlVirtualUnwind, so frames without a
		// frame pointer (ntdll, the driver, leaf code) are walked too; code without unwind
		// data (guest code) ends the walk with the word at rsp and the frame-pointer chain.
		for (size_t i = 0; i < words; ++i) out[i] = 0;
		out[0]      = context.Rip;
		size_t next = 1;
		for (int depth = 0; next < words && depth < 64; ++depth) {
			const uint64_t rsp = context.Rsp;
			if (rsp < stack_low || rsp + 8 > stack_high) break;
			uint64_t    image_base = 0;
			const auto* entry      = g_unwind_images.empty() ? nullptr : FindUnwindEntry(context.Rip, &image_base);
			if (entry == nullptr) {
				// No unwind data: the word at rsp (a leaf's caller), then frame pointers.
				if (next < words) out[next++] = *reinterpret_cast<const uint64_t*>(rsp);
				uint64_t rbp = context.Rbp;
				while (next < words) {
					if (rbp < rsp || rbp >= stack_high || stack_high - rbp < 16 || (rbp & 7u) != 0) break;
					const auto* frame = reinterpret_cast<const uint64_t*>(rbp);
					out[next++]       = frame[1];
					rbp               = frame[0] <= rbp ? 0 : frame[0];
				}
				break;
			}
			// A function with a frame register unwinds from that register: one caught before
			// its prologue set it (or mid exception dispatch) holds anything, and
			// RtlVirtualUnwind would read through it (a profiling run died at address 0x48).
			const auto* unwind = reinterpret_cast<const uint8_t*>(image_base + entry->UnwindData);
			if (const uint32_t frame_register = unwind[3] & 0x0fu; frame_register != 0) {
				const DWORD64* registers = &context.Rax;
				const uint64_t frame     = registers[frame_register] - (unwind[3] >> 4u) * 16ull;
				if (frame < stack_low || frame >= stack_high) break;
			}
			void*   handler_data = nullptr;
			DWORD64 establisher  = 0;
			RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, const_cast<RUNTIME_FUNCTION*>(entry), &context,
			                 &handler_data, &establisher, nullptr);
			if (context.Rip == 0 || context.Rsp <= rsp) break;
			out[next++] = context.Rip;
		}
	}
	ResumeThread(thread);
	return got;
}

bool ModuleOf(const void* address, uint64_t* base, char* path, size_t path_size) {
	HMODULE module = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       static_cast<LPCSTR>(address), &module) == 0)
		return false;
	*base = reinterpret_cast<uint64_t>(module);
	if (path_size != 0 && GetModuleFileNameA(module, path, static_cast<DWORD>(path_size)) == 0) path[0] = '\0';
	return true;
}

double ThreadCpuSeconds(uint64_t handle) {
	return handle != 0 ? HandleCpuSeconds(reinterpret_cast<HANDLE>(handle)) : 0;
}

uint64_t CurrentThreadCycles() {
	ULONG64 cycles = 0;
	return QueryThreadCycleTime(GetCurrentThread(), &cycles) != 0 ? cycles : 0;
}

int CurrentThreadPriority() {
	return GetThreadPriority(GetCurrentThread());
}

namespace {
struct KeptFile {
	HANDLE   handle = INVALID_HANDLE_VALUE;
	uint64_t size   = 0;
};
std::mutex                                g_kept_files_mutex;
std::unordered_map<std::string, KeptFile> g_kept_files; // (never closed: the game's files, read-only)
} // namespace

uint64_t OpenKeptReadFile(const std::string& path, uint64_t* size) {
	{
		const std::lock_guard lock(g_kept_files_mutex);
		if (const auto found = g_kept_files.find(path); found != g_kept_files.end()) {
			*size = found->second.size;
			return reinterpret_cast<uint64_t>(found->second.handle);
		}
	}
	// Opened outside the lock: a 16 GB laptop's opens took 100-390 ms while the game streamed a level in, and each
	// read of a file kept open already (an audio stream's next piece) waited for them behind it (open=150-250 ms).
	const int wide_size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
	if (wide_size <= 0) return 0;
	std::wstring wide(static_cast<size_t>(wide_size), L' ');
	MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), wide_size);
	HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
	                            FILE_ATTRIBUTE_NORMAL, nullptr);
	if (handle == INVALID_HANDLE_VALUE) return 0;
	LARGE_INTEGER length {};
	if (GetFileSizeEx(handle, &length) == 0) {
		CloseHandle(handle);
		return 0;
	}
	const std::lock_guard lock(g_kept_files_mutex);
	const auto [kept, added] = g_kept_files.try_emplace(path, KeptFile {handle, static_cast<uint64_t>(length.QuadPart)});
	if (!added) CloseHandle(handle); // (another thread opened it meanwhile: its handle is the one kept)
	*size = kept->second.size;
	return reinterpret_cast<uint64_t>(kept->second.handle);
}

uint32_t ReadOpenFileAt(uint64_t file, uint64_t offset, void* buffer, uint32_t size) {
	// A positional read (the handle is synchronous: it returns when done), so threads share the handle.
	OVERLAPPED at {};
	at.Offset     = static_cast<DWORD>(offset);
	at.OffsetHigh = static_cast<DWORD>(offset >> 32u);
	DWORD got     = 0;
	if (ReadFile(reinterpret_cast<HANDLE>(file), buffer, size, &got, &at) == 0) return 0;
	return got;
}

double NamedThreadsCpuSeconds(const char* name) {
	std::wstring wanted;
	for (const char* p = name; *p != '\0'; ++p) wanted.push_back(static_cast<wchar_t>(*p));
	double       total    = 0;
	const DWORD  process  = GetCurrentProcessId();
	HANDLE       snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return 0;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process) continue;
		HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		PWSTR description = nullptr;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			if (wanted == description) total += HandleCpuSeconds(thread);
			LocalFree(description);
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	return total;
}

std::vector<std::pair<uint32_t, std::string>> ProcessThreads() {
	std::vector<std::pair<uint32_t, std::string>> threads;
	const DWORD process  = GetCurrentProcessId();
	HANDLE      snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return threads;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process) continue;
		HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		PWSTR       description = nullptr;
		std::string name;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			for (const wchar_t* c = description; *c != L'\0'; ++c) name.push_back(*c < 128 ? static_cast<char>(*c) : '?');
			LocalFree(description);
		}
		threads.emplace_back(entry.th32ThreadID, std::move(name));
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	return threads;
}

bool FlushProcessWriteBuffers() {
	::FlushProcessWriteBuffers();
	return true;
}

bool CurrentThreadStack(uint64_t* low, uint64_t* high) {
	ULONG_PTR stack_low = 0, stack_high = 0;
	GetCurrentThreadStackLimits(&stack_low, &stack_high);
	*low  = stack_low;
	*high = stack_high;
	return stack_high > stack_low;
}

void MakeBackgroundThread(const char* avoid_cpus) {
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	DWORD_PTR process = 0, system = 0;
	if (const auto avoid = static_cast<DWORD_PTR>(CpuListMask(avoid_cpus));
	    avoid != 0 && GetProcessAffinityMask(GetCurrentProcess(), &process, &system) != 0 && (process & ~avoid) != 0)
		(void)SetThreadAffinityMask(GetCurrentThread(), process & ~avoid);
}

uint64_t PhysicalMemory() {
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	return GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 0;
}

uint64_t TrimPrivateWorkingSet(uint64_t min_address) {
	uint64_t                 asked = 0;
	MEMORY_BASIC_INFORMATION info {};
	for (const auto* at = reinterpret_cast<const uint8_t*>(min_address);
	     VirtualQuery(at, &info, sizeof(info)) == sizeof(info);
	     at = static_cast<const uint8_t*>(info.BaseAddress) + info.RegionSize) {
		// Plain read-write only: write-combined upload memory (PAGE_READWRITE | PAGE_WRITECOMBINE) stays.
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && info.Protect == PAGE_READWRITE) {
			(void)VirtualUnlock(info.BaseAddress, info.RegionSize);
			asked += info.RegionSize;
		}
	}
	return asked;
}

void MakeCriticalThread() {
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}

uint64_t OpenScratchFile() {
	wchar_t directory[MAX_PATH + 1] {}, path[MAX_PATH + 1] {};
	if (GetTempPathW(MAX_PATH + 1, directory) == 0 || GetTempFileNameW(directory, L"kyt", 0, path) == 0) return 0;
	HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
	                          FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
	if (file == INVALID_HANDLE_VALUE) {
		(void)DeleteFileW(path);
		return 0;
	}
	return reinterpret_cast<uint64_t>(file);
}

// ReadFile or WriteFile at an offset, in pieces a DWORD can count.
template <typename Call, typename Byte>
static bool ScratchTransfer(uint64_t file, uint64_t offset, Byte* data, size_t size, Call call) {
	while (size != 0) {
		OVERLAPPED at {};
		at.Offset         = static_cast<DWORD>(offset);
		at.OffsetHigh     = static_cast<DWORD>(offset >> 32u);
		DWORD      done   = 0;
		const auto length = static_cast<DWORD>(size < (1u << 30u) ? size : (1u << 30u));
		if (call(reinterpret_cast<HANDLE>(file), data, length, &done, &at) == 0 || done == 0) return false;
		data += done;
		offset += done;
		size -= done;
	}
	return true;
}

bool WriteScratchFile(uint64_t file, uint64_t offset, const void* data, size_t size) {
	return file != 0 && ScratchTransfer(file, offset, static_cast<const uint8_t*>(data), size, WriteFile);
}

bool ReadScratchFile(uint64_t file, uint64_t offset, void* data, size_t size) {
	return file != 0 && ScratchTransfer(file, offset, static_cast<uint8_t*>(data), size, ReadFile);
}

void CloseScratchFile(uint64_t file) {
	if (file != 0) CloseHandle(reinterpret_cast<HANDLE>(file));
}

uint64_t OpenFileForReading(const char* path) {
	wchar_t wide[MAX_PATH * 2] {};
	if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, MAX_PATH * 2) == 0) return 0;
	HANDLE file = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
	                          FILE_FLAG_RANDOM_ACCESS, nullptr);
	return file == INVALID_HANDLE_VALUE ? 0 : reinterpret_cast<uint64_t>(file);
}

#else

void SetThreadName(const char* name) {
	(void)pthread_setname_np(pthread_self(), name);
}

uint32_t ThreadId() {
	return static_cast<uint32_t>(syscall(SYS_gettid));
}

static bool MaskToSet(uint64_t mask, cpu_set_t* set) {
	CPU_ZERO(set);
	for (int cpu = 0; cpu < 64; ++cpu)
		if ((mask >> cpu) & 1u) CPU_SET(cpu, set);
	return mask != 0;
}

void PinThreadToCpuList(const char* list) {
	cpu_set_t set;
	if (MaskToSet(CpuListMask(list), &set)) (void)sched_setaffinity(0, sizeof(set), &set);
}

bool PinThreadHandleToCpuList(uint64_t handle, const char* list) {
	cpu_set_t set;
	return handle != 0 && MaskToSet(CpuListMask(list), &set) &&
	       pthread_setaffinity_np(static_cast<pthread_t>(handle), sizeof(set), &set) == 0;
}

bool SetProcessDefaultCpuList(const char* /*list*/) {
	return false;
}

static double ClockSeconds(clockid_t clock) {
	timespec now {};
	if (clock_gettime(clock, &now) != 0) return 0;
	return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) / 1e9;
}

uint64_t CurrentThreadHandle() {
	return static_cast<uint64_t>(pthread_self());
}

double ThreadCpuSeconds(uint64_t handle) {
	clockid_t clock {};
	if (pthread_getcpuclockid(static_cast<pthread_t>(handle), &clock) != 0) return 0;
	return ClockSeconds(clock);
}

uint64_t CurrentThreadCycles() {
	return 0;
}

int CurrentThreadPriority() {
	return 0;
}

uint64_t OpenKeptReadFile(const std::string& /*path*/, uint64_t* /*size*/) {
	return 0;
}

uint32_t ReadOpenFileAt(uint64_t /*file*/, uint64_t /*offset*/, void* /*buffer*/, uint32_t /*size*/) {
	return 0;
}

std::vector<std::pair<uint32_t, std::string>> ProcessThreads() {
	return {};
}

double NamedThreadsCpuSeconds(const char* name) {
	double     total = 0;
	const long ticks = sysconf(_SC_CLK_TCK);
	if (auto* dir = opendir("/proc/self/task")) {
		while (auto* entry = readdir(dir)) {
			if (entry->d_name[0] == '.') continue;
			char path[64], comm[32] {};
			std::snprintf(path, sizeof(path), "/proc/self/task/%s/comm", entry->d_name);
			if (auto* f = std::fopen(path, "re")) {
				if (std::fgets(comm, sizeof(comm), f)) comm[std::strcspn(comm, "\n")] = 0;
				std::fclose(f);
			}
			if (std::strcmp(comm, name) != 0) continue;
			std::snprintf(path, sizeof(path), "/proc/self/task/%s/stat", entry->d_name);
			if (auto* f = std::fopen(path, "re")) {
				char line[1024] {};
				if (std::fgets(line, sizeof(line), f)) {
					const char*        rest  = std::strrchr(line, ')');
					unsigned long long utime = 0, stime = 0;
					if (rest && std::sscanf(rest + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu",
					                        &utime, &stime) == 2)
						total += static_cast<double>(utime + stime) / static_cast<double>(ticks);
				}
				std::fclose(f);
			}
		}
		closedir(dir);
	}
	return total;
}

bool CurrentThreadStack(uint64_t* low, uint64_t* high) {
	pthread_attr_t attr;
	bool           found = false;
	if (pthread_getattr_np(pthread_self(), &attr) == 0) {
		void*  stack = nullptr;
		size_t size  = 0;
		if (pthread_attr_getstack(&attr, &stack, &size) == 0) {
			*low  = reinterpret_cast<uint64_t>(stack);
			*high = *low + size;
			found = true;
		}
		pthread_attr_destroy(&attr);
	}
	return found;
}

void MakeBackgroundThread(const char* avoid_cpus) {
	(void)setpriority(PRIO_PROCESS, static_cast<id_t>(ThreadId()), 10);
	cpu_set_t set;
	if (const auto avoid = CpuListMask(avoid_cpus); avoid != 0 && sched_getaffinity(0, sizeof(set), &set) == 0) {
		for (int cpu = 0; cpu < 64; ++cpu)
			if ((avoid >> cpu) & 1u) CPU_CLR(cpu, &set);
		if (CPU_COUNT(&set) != 0) (void)sched_setaffinity(0, sizeof(set), &set);
	}
}

// Handles are the file descriptor + 1 (0: none).
uint64_t PhysicalMemory() {
	const long pages = sysconf(_SC_PHYS_PAGES);
	const long size  = sysconf(_SC_PAGE_SIZE);
	return pages > 0 && size > 0 ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(size) : 0;
}

uint64_t TrimPrivateWorkingSet(uint64_t /*min_address*/) {
	return 0;
}

void MakeCriticalThread() {}

// Handles are the file descriptor + 1 (0: none).
uint64_t OpenScratchFile() {
	const char* directory = std::getenv("TMPDIR");
	std::string path      = std::string(directory != nullptr && *directory != '\0' ? directory : "/tmp") + "/kyty-XXXXXX";
	const int   fd        = mkstemp(path.data());
	if (fd < 0) return 0;
	(void)unlink(path.c_str());
	return static_cast<uint64_t>(fd) + 1;
}

bool WriteScratchFile(uint64_t file, uint64_t offset, const void* data, size_t size) {
	for (const auto* bytes = static_cast<const uint8_t*>(data); file != 0 && size != 0;) {
		const auto done = pwrite(static_cast<int>(file - 1), bytes, size, static_cast<off_t>(offset));
		if (done <= 0) return false;
		bytes += done;
		offset += static_cast<uint64_t>(done);
		size -= static_cast<size_t>(done);
	}
	return file != 0;
}

bool ReadScratchFile(uint64_t file, uint64_t offset, void* data, size_t size) {
	for (auto* bytes = static_cast<uint8_t*>(data); file != 0 && size != 0;) {
		const auto done = pread(static_cast<int>(file - 1), bytes, size, static_cast<off_t>(offset));
		if (done <= 0) return false;
		bytes += done;
		offset += static_cast<uint64_t>(done);
		size -= static_cast<size_t>(done);
	}
	return file != 0;
}

void CloseScratchFile(uint64_t file) {
	if (file != 0) close(static_cast<int>(file - 1));
}

uint64_t OpenFileForReading(const char* path) {
	const int fd = open(path, O_RDONLY | O_CLOEXEC);
	return fd < 0 ? 0 : static_cast<uint64_t>(fd) + 1;
}

#endif

} // namespace LocalPlatform
