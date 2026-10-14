#pragma once

#include "common/abi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#if defined(__x86_64__) || defined(_M_X64)
#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>
#endif

namespace Loader {
struct Program;
namespace DemonsSoulsIdle {
void Install(Program* program);
void Clear();
// The opt-in empty-poll and collision-guard thunks are process-lifetime patches. Guest exits use
// quick_exit with the requested status, as WindowRun already does, instead of
// restoring/freeing code that another guest thread may still be executing.
[[nodiscard]] bool HasProcessLifetimePatch();
[[nodiscard]] bool OwnsProcessLifetimePatch(const Program* program);

inline constexpr uint64_t                CallOffset = 0x83b59b, PollOffset = 0x83b880;
inline constexpr std::array<uint8_t, 24> CallBytes {0xe8, 0xe0, 0x02, 0x00, 0x00, 0x84, 0xc0, 0x74,
                                                    0xd7, 0x4c, 0x8b, 0x7d, 0xc8, 0x49, 0x8b, 0xb7,
                                                    0x90, 0x00, 0x00, 0x00, 0x48, 0x85, 0xf6, 0x74};
inline constexpr std::array<uint8_t, 24> PollBytes {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                    0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec,
                                                    0x88, 0x00, 0x00, 0x00, 0x4c, 0x8b, 0x35, 0x25};

// The same call and poll in the USA release PPSA01342 01.004.000: the job-table structure has the
// pointer at +0xb0 instead of +0x90 and the poll a 0xc8-byte frame instead of 0x88, so neither the
// fixed offsets nor the search above find it. Found by a wildcard scan of that eboot (one site;
// 0x86e000 is the page where 51% of its guest CPU samples were spinning, floppa-ps5 X72).
inline constexpr uint64_t                CallOffsetUsa = 0x86e963, PollOffsetUsa = 0x86ecd0;
inline constexpr std::array<uint8_t, 24> CallBytesUsa {0xe8, 0x68, 0x03, 0x00, 0x00, 0x84, 0xc0, 0x74,
                                                       0xcf, 0x4c, 0x8b, 0x7d, 0xc8, 0x49, 0x8b, 0xb7,
                                                       0xb0, 0x00, 0x00, 0x00, 0x48, 0x85, 0xf6, 0x74};
inline constexpr std::array<uint8_t, 24> PollBytesUsa {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                                       0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec,
                                                       0xc8, 0x00, 0x00, 0x00, 0x4c, 0x8b, 0x35, 0xcd};

inline bool Matches(std::span<const uint8_t> call, std::span<const uint8_t> poll) {
	return std::ranges::equal(call, CallBytes) && std::ranges::equal(poll, PollBytes);
}
inline bool MatchesUsa(std::span<const uint8_t> call, std::span<const uint8_t> poll) {
	return std::ranges::equal(call, CallBytesUsa) && std::ranges::equal(poll, PollBytesUsa);
}

// KYTY_DS_MAIN_WAIT_YIELD=1 (USA 01.004.000): the main thread's wait for its jobs (eboot+0x86eb00..0x86ecba) spins
// on the job counters, a pause, a TSC read, the job-queue spinlock and a try-run-job: a whole core of a 6-core PC (100%
// of the main thread's samples in gameplay, its 13 job workers at 16-22% each). Each round's "pause; mov r13, r15"
// (eboot+0x86eb39, before the lock is taken) becomes a call of a thunk that gives the CPU to a ready thread
// (SwitchToThread: nothing when none is ready), then does them. Checked by its bytes.
inline constexpr uint64_t                MainWaitOffsetUsa = 0x86eb39;
inline constexpr std::array<uint8_t, 17> MainWaitBytesUsa {0xf3, 0x90, 0x4d, 0x89, 0xfd, 0x45, 0x89, 0xe7, 0x4e,
                                                           0x8b, 0xa4, 0xf3, 0x00, 0x0f, 0x00, 0x00, 0xe8};

#if defined(__x86_64__) || defined(_M_X64)
// The poll's observable register state is retained even on the idle path.
// In particular this is not an ordinary C++ function replacement: its caller
// may rely on registers the original poll was known not to modify.
inline void EmitThunk(Xbyak::CodeGenerator& c, const void* poll, const void* sleep) {
	using namespace Xbyak::util;
	Xbyak::Label done;
	c.sub(rsp, 8);
	c.call(poll);
	c.add(rsp, 8);
	c.test(al, al);
	c.jnz(done, Xbyak::CodeGenerator::T_NEAR);
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11})
		c.push(reg);
	// Entry RSP is 8 mod 16; the 80-byte save above retains that alignment.
	c.sub(rsp, 520);
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(sleep));
	c.call(rax); // Explicit SysV callback bridges to the platform sleep API.
	c.fxrstor64(ptr[rsp]);
	c.add(rsp, 520);
	for (const auto& reg: {r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.popfq();
	c.L(done);
	c.ret();
}

// The main-wait thunk: every register and the flags kept, `yield` called (SysV), then the replaced instructions.
inline void EmitYieldThunk(Xbyak::CodeGenerator& c, const void* yield) {
	using namespace Xbyak::util;
	c.pushfq();
	for (const auto& reg: {rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11})
		c.push(reg);
	// Entry RSP is 8 mod 16 (the function body keeps RSP 16-aligned); the 80-byte save above retains that alignment.
	c.sub(rsp, 520);
	constexpr uint8_t save_fp[] {0x48, 0x0f, 0xae, 0x04, 0x24}; // FXSAVE64 [rsp]
	c.db(save_fp, sizeof(save_fp));
	c.mov(rax, reinterpret_cast<uint64_t>(yield));
	c.call(rax);
	c.fxrstor64(ptr[rsp]);
	c.add(rsp, 520);
	for (const auto& reg: {r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax})
		c.pop(reg);
	c.popfq();
	c.pause();
	c.mov(r13, r15);
	c.ret();
}
#endif
} // namespace DemonsSoulsIdle
} // namespace Loader
