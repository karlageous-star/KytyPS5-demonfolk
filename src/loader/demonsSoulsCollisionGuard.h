#pragma once

#include "common/abi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>

#if defined(__x86_64__) || defined(_M_X64)
#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif
#include <xbyak/xbyak.h>
#endif

namespace Loader::DemonsSoulsCollisionGuard {
// USA PPSA01342 01.004.000 only. This is a mitigation of an invalid
// temporary-list state, not a repair of the cause of that state.
inline constexpr uint64_t FunctionOffset = 0xe76140, FunctionSize = 0x456;
inline constexpr uint64_t SiteOffset = 0xe762cd, NormalOffset = 0xe762d5, DetachedOffset = 0xe762b7;
inline constexpr uint64_t CaveOffset = 0x80c0000;
inline constexpr uint64_t FunctionFingerprint = 0xdcf5496bb1a60ba5ull;
inline constexpr std::array<uint8_t, 8> Original {0x49, 0x8b, 0x4f, 0x50, 0x49, 0x8b, 0x47, 0x58};
inline std::atomic<uint64_t> g_detached_unlinks {0};
static_assert(std::atomic<uint64_t>::is_always_lock_free && sizeof(g_detached_unlinks) == 8);

inline uint64_t Fingerprint(std::span<const uint8_t> bytes) {
	uint64_t hash = 14695981039346656037ull;
	for (auto byte: bytes) hash = (hash ^ byte) * 1099511628211ull;
	return hash;
}

inline bool Matches(std::span<const uint8_t> function) {
	return function.size() == FunctionSize && Fingerprint(function) == FunctionFingerprint &&
	       std::ranges::equal(function.subspan(SiteOffset - FunctionOffset, Original.size()), Original);
}

inline bool MakeJump(uint64_t from, uint64_t to, std::array<uint8_t, Original.size()>& bytes) {
	if (from > UINT64_MAX - 5) return false;
	const auto next = from + 5;
	if ((to >= next && to - next > INT32_MAX) ||
	    (to < next && next - to > uint64_t {INT32_MAX} + 1)) return false;
	const int32_t delta = to >= next ? static_cast<int32_t>(to - next)
	                                 : static_cast<int32_t>(-static_cast<int64_t>(next - to));
	bytes.fill(0x90);
	bytes[0] = 0xe9;
	for (unsigned i = 0; i < 4; ++i) bytes[i + 1] = static_cast<uint32_t>(delta) >> (8 * i);
	return true;
}

// Called only by the render thread. Emit once per changed count even when
// a guarded event did not exceed the slow-frame logging threshold.
inline void Report() {
	static uint64_t reported = 0;
	const auto total = g_detached_unlinks.load(std::memory_order_relaxed);
	if (total == reported) return;
	std::printf("Demon's Souls collision guard: detached_unlinks=%llu (invalid end-node removal bypassed)\n",
	            static_cast<unsigned long long>(total));
	std::fflush(stdout);
	reported = total;
}

#if defined(__x86_64__) || defined(_M_X64)
inline void EmitThunk(Xbyak::CodeGenerator& c, const void* normal, const void* detached, const void* counter) {
	using namespace Xbyak::util;
	Xbyak::Label unchanged;
	// Reproduce the two displaced loads. A normal member, including a valid
	// tail with next==0, follows the original unlink instructions unchanged.
	c.mov(rcx, qword[r15 + 0x50]);
	c.mov(rax, qword[r15 + 0x58]);
	// Do not use any of the guest red zone or call a host ABI bridge.
	c.lea(rsp, ptr[rsp - 128]);
	c.pushfq();
	c.push(r11);
	c.test(rax, rax);
	c.jnz(unchanged);
	c.test(rcx, rcx);
	c.jnz(unchanged);
	// The observed fault was the end endpoint at node+0x20. Do not turn
	// unrelated null links or other malformed nodes into successful removals.
	c.lea(r11, ptr[r15 + 0x20]);
	c.cmp(r14, r11);
	c.jne(unchanged);
	c.mov(r11, reinterpret_cast<uint64_t>(counter));
	c.lock();
	c.inc(qword[r11]);
	c.pop(r11);
	c.popfq();
	c.lea(rsp, ptr[rsp + 128]);
	c.jmp(detached);
	c.L(unchanged);
	c.pop(r11);
	c.popfq();
	c.lea(rsp, ptr[rsp + 128]);
	c.jmp(normal);
}
#endif
} // namespace Loader::DemonsSoulsCollisionGuard
