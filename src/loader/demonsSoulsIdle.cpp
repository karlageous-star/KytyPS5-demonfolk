#include "loader/demonsSoulsIdle.h"
#include "loader/demonsSoulsCollisionGuard.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/guestCode.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <xbyak/xbyak_util.h>
#endif

// KYTY_SPIN_BACKOFF_US=<n> (live kyty_local_spin_backoff_us, default 50): the workers' TSC helper waits this long
// during each otherwise-busy spin round; 0 retains the game's original spin.
extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_spin_backoff_us {[] {
	const char* value = std::getenv("KYTY_SPIN_BACKOFF_US");
	return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 50u;
}()};
}

namespace Loader::DemonsSoulsIdle {
namespace {
constexpr uint64_t     PageSize = 0x4000, CaveOffset = 0x8000000;
uint64_t               cave = 0, site = 0, wait_site = 0, spin_site = 0;
std::array<uint8_t, 5> installed_call {}, original_call {}, installed_wait {}, spin_installed {}, spin_original {};

void KYTY_SYSV_ABI WaitForWork() {
	Common::Thread::SleepMicroWithoutSpinning(50);
}

void KYTY_SYSV_ABI YieldMainWait() {
	std::this_thread::yield();
}

void KYTY_SYSV_ABI SpinBackoff() {
	if (const auto us = kyty_local_spin_backoff_us.load(std::memory_order_relaxed); us != 0)
		Common::Thread::SleepMicroWithoutSpinning(us);
}

bool Readable(uint64_t address, uint64_t size) {
	// Executable modules are private runtime allocations, not GPU backing aliases.
	return Libs::Graphics::HostMemoryRangeIsReadable(address, size);
}

#include "demonsSoulsEmptyPoll.inc"
#include "demonsSoulsCollisionGuard.inc"

// Another build of the game (another region or version) has the same code elsewhere: the call's
// bytes after its displacement, landing on the poll's prologue up to its RIP-relative displacement.
// Only a single such site counts.
bool FindSites(const Program& program, uint64_t* call, uint64_t* poll) {
	// The structured pattern covers the 01.005.000 and 01.007.000 worker loops when their fixed RVAs differ.
	if (const auto match = GuestCode::FindUnique(program, GuestCode::Pattern(IdleCall))) {
		const uint64_t candidate_call = *match + IdleCallAt;
		int32_t        displacement = 0;
		std::memcpy(&displacement, reinterpret_cast<const void*>(candidate_call + 1), sizeof(displacement));
		const uint64_t candidate_poll = candidate_call + 5 + static_cast<int64_t>(displacement);
		const GuestCode::Pattern prologue(PollPrologue);
		const uint64_t           base = program.base_vaddr, end = base + program.mapped_size;
		if (base <= UINT64_MAX - program.mapped_size && end >= prologue.Size() && candidate_poll >= base &&
		    candidate_poll <= end - prologue.Size() && Readable(candidate_poll, prologue.Size()) &&
		    prologue.Matches(reinterpret_cast<const uint8_t*>(candidate_poll))) {
			*call = candidate_call;
			*poll = candidate_poll;
			return true;
		}
	}
	const std::span<const uint8_t> tail(CallBytes.data() + 5, CallBytes.size() - 5);
	const std::span<const uint8_t> prologue(PollBytes.data(), PollBytes.size() - 1);
	const std::boyer_moore_horspool_searcher searcher(tail.begin(), tail.end());
	const uint64_t                           base = program.base_vaddr, end = base + program.mapped_size;
	constexpr uint64_t                       Chunk = 0x100000;
	size_t                                   found = 0;
	for (uint64_t at = base; at < end; at += Chunk) {
		// Chunks overlap by a pattern less a byte: a match is in exactly one of them.
		const auto size = std::min(Chunk + tail.size() - 1, end - at);
		if (!Readable(at, size)) continue;
		const auto* bytes = reinterpret_cast<const uint8_t*>(at);
		for (const auto* match = std::search(bytes, bytes + size, searcher); match != bytes + size;
		     match           = std::search(match + 1, bytes + size, searcher)) {
			const auto from = reinterpret_cast<uint64_t>(match) - 5;
			if (from < base || !Readable(from, 5) || *reinterpret_cast<const uint8_t*>(from) != 0xe8) continue;
			int32_t displacement = 0;
			std::memcpy(&displacement, reinterpret_cast<const void*>(from + 1), sizeof(displacement));
			const auto to = from + 5 + static_cast<int64_t>(displacement);
			if (to < base || to > end - PollBytes.size() || !Readable(to, PollBytes.size()) ||
			    !std::equal(prologue.begin(), prologue.end(), reinterpret_cast<const uint8_t*>(to)))
				continue;
			if (found++ != 0) return false;
			*call = from;
			*poll = to;
		}
	}
	return found == 1;
}

// The worker wait's TSC helper call (when its caller and helper match the audited pattern).
uint64_t FindSpinCall(const Program& program) {
	const auto found = GuestCode::FindUnique(program, GuestCode::Pattern(SpinCall));
	if (!found) return 0;
	const uint64_t call = *found + SpinCallAt;
	int32_t        displacement = 0;
	std::memcpy(&displacement, reinterpret_cast<const void*>(call + 1), sizeof(displacement));
	const uint64_t helper = call + 5 + static_cast<int64_t>(displacement);
	const GuestCode::Pattern body(TscHelper);
	const uint64_t           base = program.base_vaddr, end = base + program.mapped_size;
	return base <= UINT64_MAX - program.mapped_size && end >= body.Size() && helper >= base &&
	               helper <= end - body.Size() && Readable(helper, body.Size()) &&
	               body.Matches(reinterpret_cast<const uint8_t*>(helper))
	           ? call
	           : 0;
}
} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	InstallCollisionGuard(program);
	if (cave != 0 || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin" || program->mapped_size > CaveOffset ||
	    program->base_vaddr > UINT64_MAX - CaveOffset - PageSize)
		return;
	auto call = program->base_vaddr + CallOffset;
	auto poll = program->base_vaddr + PollOffset;
	std::array<uint8_t, CallBytes.size()> call_bytes {};
	std::array<uint8_t, PollBytes.size()> poll_bytes {};
	const bool at_offsets = program->mapped_size >= PollOffset + PollBytes.size() &&
	                        Readable(call, call_bytes.size()) && Readable(poll, poll_bytes.size());
	if (at_offsets) {
		std::memcpy(call_bytes.data(), reinterpret_cast<const void*>(call), call_bytes.size());
		std::memcpy(poll_bytes.data(), reinterpret_cast<const void*>(poll), poll_bytes.size());
	}
	bool usa = false;
	if (!(at_offsets && Matches(call_bytes, poll_bytes))) {
		const auto usa_call = program->base_vaddr + CallOffsetUsa, usa_poll = program->base_vaddr + PollOffsetUsa;
		if (program->mapped_size >= PollOffsetUsa + PollBytesUsa.size() && Readable(usa_call, call_bytes.size()) &&
		    Readable(usa_poll, poll_bytes.size())) {
			std::memcpy(call_bytes.data(), reinterpret_cast<const void*>(usa_call), call_bytes.size());
			std::memcpy(poll_bytes.data(), reinterpret_cast<const void*>(usa_poll), poll_bytes.size());
			if (MatchesUsa(call_bytes, poll_bytes)) {
				call = usa_call;
				poll = usa_poll;
				usa  = true;
			}
		}
	}
	if (!usa && !(at_offsets && Matches(call_bytes, poll_bytes)) && !FindSites(*program, &call, &poll)) {
		std::printf("Demon's Souls idle wait: code signature differs; retaining guest code\n");
		return;
	}
	const auto requested = program->base_vaddr + CaveOffset;
	const auto allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    requested, PageSize, Common::VirtualMemory::Mode::ExecuteReadWrite,
	    "demons_souls_idle_wait", true);
	if (allocated != requested) {
		if (allocated) Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	Xbyak::ClearError();
	Xbyak::CodeGenerator code(PageSize, reinterpret_cast<void*>(allocated));
	EmitThunk(code, reinterpret_cast<const void*>(poll),
	          reinterpret_cast<const void*>(&WaitForWork));
	const uint64_t spin_call = FindSpinCall(*program);
	uint64_t       spin_thunk = 0;
	if (spin_call != 0) {
		code.align(16);
		spin_thunk = reinterpret_cast<uint64_t>(code.getCurr());
		EmitSpinThunk(code, reinterpret_cast<const void*>(&SpinBackoff));
	}
	// KYTY_DS_MAIN_WAIT_YIELD=1: the main thread's job wait yields its CPU each round (demonsSoulsIdle.h).
	uint64_t wait_call = 0, wait_thunk = 0;
	if (const char* yield = std::getenv("KYTY_DS_MAIN_WAIT_YIELD"); usa && yield != nullptr && *yield == '1') {
		const auto at = program->base_vaddr + MainWaitOffsetUsa;
		if (program->mapped_size >= MainWaitOffsetUsa + MainWaitBytesUsa.size() && Readable(at, MainWaitBytesUsa.size()) &&
		    std::memcmp(reinterpret_cast<const void*>(at), MainWaitBytesUsa.data(), MainWaitBytesUsa.size()) == 0) {
			wait_call = at;
			code.align(16);
			wait_thunk = reinterpret_cast<uint64_t>(code.getCurr());
			EmitYieldThunk(code, reinterpret_cast<const void*>(&YieldMainWait));
		} else {
			std::printf("Demon's Souls main wait yield: the wait loop of 01.004.000 not found; retaining guest code\n");
		}
	}
	code.ready();
	if (Xbyak::GetError() ||
	    !Common::VirtualMemory::FlushInstructionCache(allocated, code.getSize()) ||
	    !Libs::LibKernel::Memory::ProtectGuestMemory(
	        allocated, PageSize, Common::VirtualMemory::Mode::ExecuteRead, nullptr)) {
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	installed_call[0]       = 0xe8;
	const auto displacement = static_cast<int32_t>(allocated - (call + 5));
	std::memcpy(installed_call.data() + 1, &displacement, sizeof(displacement));
	std::memcpy(original_call.data(), reinterpret_cast<const void*>(call), original_call.size());
	// Installation occurs before module initializers or guest worker threads run.
	// An external patch with different bytes is deliberately left untouched.
	// SetProgramMemoryProtection keeps executable code writable for loader patches.
	std::memcpy(reinterpret_cast<void*>(call), installed_call.data(), installed_call.size());
	if (!Common::VirtualMemory::FlushInstructionCache(call, installed_call.size())) {
		std::memcpy(reinterpret_cast<void*>(call), original_call.data(), original_call.size());
		Common::VirtualMemory::FlushInstructionCache(call, installed_call.size());
		Libs::LibKernel::Memory::FreeGuestMemory(allocated, PageSize);
		return;
	}
	cave = allocated;
	site = call;
	std::printf("Demon's Souls idle wait: installed portable 50 us backoff (call at eboot+0x%llx)\n",
	     static_cast<unsigned long long>(call - program->base_vaddr));
	if (wait_call != 0) {
		installed_wait[0]    = 0xe8;
		const auto wait_disp = static_cast<int32_t>(wait_thunk - (wait_call + 5));
		std::memcpy(installed_wait.data() + 1, &wait_disp, sizeof(wait_disp));
		std::memcpy(reinterpret_cast<void*>(wait_call), installed_wait.data(), installed_wait.size());
		if (Common::VirtualMemory::FlushInstructionCache(wait_call, installed_wait.size())) {
			wait_site = wait_call;
			std::printf("Demon's Souls main wait yield: installed (eboot+0x%llx)\n",
			            static_cast<unsigned long long>(wait_call - program->base_vaddr));
		} else {
			std::memcpy(reinterpret_cast<void*>(wait_call), MainWaitBytesUsa.data(), installed_wait.size());
			Common::VirtualMemory::FlushInstructionCache(wait_call, installed_wait.size());
		}
	}
	if (spin_call != 0) {
		spin_installed[0] = 0xe8;
		const auto spin_disp = static_cast<int32_t>(spin_thunk - (spin_call + 5));
		std::memcpy(spin_installed.data() + 1, &spin_disp, sizeof(spin_disp));
		std::memcpy(spin_original.data(), reinterpret_cast<const void*>(spin_call), spin_original.size());
		std::memcpy(reinterpret_cast<void*>(spin_call), spin_installed.data(), spin_installed.size());
		if (Common::VirtualMemory::FlushInstructionCache(spin_call, spin_installed.size())) {
			spin_site = spin_call;
			std::printf("Demon's Souls job wait: the spin's TSC read at eboot+0x%llx backs off %u us (live "
			            "kyty_local_spin_backoff_us)\n",
			            static_cast<unsigned long long>(spin_call - program->base_vaddr),
			            kyty_local_spin_backoff_us.load(std::memory_order_relaxed));
		} else {
			std::memcpy(reinterpret_cast<void*>(spin_call), spin_original.data(), spin_original.size());
			Common::VirtualMemory::FlushInstructionCache(spin_call, spin_original.size());
		}
	}
	// Both USA opt-ins replace eboot+0x86eb39. Keep the main-wait yield if it was installed; the
	// empty-poll patch must not allocate a process-lifetime thunk for a site it cannot own.
	const char* empty_poll = std::getenv("KYTY_DS_EMPTY_POLL_BACKOFF");
	if (wait_site != 0 && wait_site == program->base_vaddr + MainWaitOffsetUsa && empty_poll != nullptr &&
	    std::strcmp(empty_poll, "1") == 0) {
		std::printf("Demon's Souls empty poll: shares its site with the main-wait yield; keeping the main-wait yield\n");
	} else {
		InstallEmptyPoll(program);
	}
#endif
}

void Clear() {
#if defined(__x86_64__) || defined(_M_X64)
	// RuntimeLinker::Clear has no guest-thread quiescence protocol.
	EXIT_IF(empty_poll_cave != 0 || collision_guard_cave != 0);
#endif
	if (!cave) return;
	if (wait_site != 0) {
		std::array<uint8_t, 5> wait_bytes {};
		std::memcpy(wait_bytes.data(), reinterpret_cast<const void*>(wait_site), wait_bytes.size());
		if (wait_bytes == installed_wait) {
			std::memcpy(reinterpret_cast<void*>(wait_site), MainWaitBytesUsa.data(), installed_wait.size());
			Common::VirtualMemory::FlushInstructionCache(wait_site, installed_wait.size());
		}
		wait_site = 0;
	}
	std::array<uint8_t, 5> bytes {};
	std::memcpy(bytes.data(), reinterpret_cast<const void*>(site), bytes.size());
	if (bytes == installed_call) {
		std::memcpy(reinterpret_cast<void*>(site), original_call.data(), bytes.size());
		Common::VirtualMemory::FlushInstructionCache(site, bytes.size());
	}
	if (spin_site != 0) {
		std::memcpy(bytes.data(), reinterpret_cast<const void*>(spin_site), bytes.size());
		if (bytes == spin_installed) {
			std::memcpy(reinterpret_cast<void*>(spin_site), spin_original.data(), bytes.size());
			Common::VirtualMemory::FlushInstructionCache(spin_site, bytes.size());
		}
	}
	Libs::LibKernel::Memory::FreeGuestMemory(cave, PageSize);
	cave = site = spin_site = 0;
}
bool HasProcessLifetimePatch() {
#if defined(__x86_64__) || defined(_M_X64)
	return empty_poll_cave != 0 || collision_guard_cave != 0;
#else
	return false;
#endif
}
bool OwnsProcessLifetimePatch(const Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	return (empty_poll_cave != 0 && program == empty_poll_owner) ||
	       (collision_guard_cave != 0 && program == collision_guard_owner);
#else
	return false;
#endif
}
} // namespace Loader::DemonsSoulsIdle
