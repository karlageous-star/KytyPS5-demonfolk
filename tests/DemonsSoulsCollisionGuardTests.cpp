#include "loader/demonsSoulsCollisionGuard.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <xbyak/xbyak_util.h>
#endif

namespace {
namespace Guard = Loader::DemonsSoulsCollisionGuard;

void Check(bool pass, const char* message) {
	if (!pass) {
		std::fprintf(stderr, "collision guard: %s\n", message);
		std::abort();
	}
}

void TestJumpEncoding() {
	constexpr uint64_t From = 0x100000000ull;
	constexpr uint64_t Next = From + 5;
	std::array<uint8_t, Guard::Original.size()> bytes {};

	Check(Guard::MakeJump(From, Next + INT32_MAX, bytes), "positive rel32 limit accepted");
	Check(bytes == std::array<uint8_t, 8> {0xe9, 0xff, 0xff, 0xff, 0x7f, 0x90, 0x90, 0x90},
	      "positive rel32 encoding and NOP fill");

	Check(Guard::MakeJump(From, Next - (uint64_t {INT32_MAX} + 1), bytes),
	      "negative rel32 limit accepted");
	Check(bytes == std::array<uint8_t, 8> {0xe9, 0x00, 0x00, 0x00, 0x80, 0x90, 0x90, 0x90},
	      "negative rel32 encoding and NOP fill");

	Check(!Guard::MakeJump(From, Next + uint64_t {INT32_MAX} + 1, bytes),
	      "positive rel32 overflow rejected");
	Check(!Guard::MakeJump(From, Next - (uint64_t {INT32_MAX} + 2), bytes),
	      "negative rel32 overflow rejected");
	Check(!Guard::MakeJump(UINT64_MAX - 4, UINT64_MAX, bytes), "from plus instruction length overflow rejected");
}

void TestSignatureRejection() {
	// Keep the test fixture synthetic: never embed the complete game routine.
	std::vector<uint8_t> function(Guard::FunctionSize, 0);
	constexpr auto       site = Guard::SiteOffset - Guard::FunctionOffset;
	std::copy(Guard::Original.begin(), Guard::Original.end(), function.begin() + site);
	Check(!Guard::Matches(std::span<const uint8_t>(function)), "synthetic routine is not mistaken for the game build");

	function[site + 3] ^= 1;
	Check(!Guard::Matches(std::span<const uint8_t>(function)), "mutated displaced-load signature rejected");
	Check(!Guard::Matches(std::span<const uint8_t>(function).first(function.size() - 1)),
	      "short routine signature rejected");
}

void TestSignatureFixture(const char* path) {
	constexpr uint64_t SelfTextDataOffset = 0x10a30;
	static_assert(SelfTextDataOffset <= UINT64_MAX - Guard::FunctionOffset);
	std::ifstream self(path, std::ios::binary);
	Check(self.good(), "cannot open optional SELF fixture");
	self.seekg(static_cast<std::streamoff>(SelfTextDataOffset + Guard::FunctionOffset));
	Check(self.good(), "cannot seek to optional SELF routine");
	std::vector<uint8_t> function(Guard::FunctionSize);
	self.read(reinterpret_cast<char*>(function.data()), static_cast<std::streamsize>(function.size()));
	Check(self.gcount() == static_cast<std::streamsize>(function.size()), "optional SELF routine is truncated");
	Check(Guard::Matches(std::span<const uint8_t>(function)), "optional SELF routine fingerprint/signature mismatch");
}

#if defined(__x86_64__) || defined(_M_X64)
struct Node {
	std::array<std::byte, 0x50> padding {};
	uint64_t                   next_at_50 = 0;
	uint64_t                   prev_at_58 = 0;
};
static_assert(offsetof(Node, next_at_50) == 0x50 && offsetof(Node, prev_at_58) == 0x58);

struct Snapshot {
	uint64_t              rax = 0;
	uint64_t              rcx = 0;
	uint64_t              r11 = 0;
	uint64_t              rbx = 0;
	uint64_t              rbp = 0;
	uint64_t              r12 = 0;
	uint64_t              r13 = 0;
	uint64_t              r14 = 0;
	uint64_t              r15 = 0;
	uint64_t              saved_rsp = 0;
	uint64_t              marker = 0;
	uint64_t              flags_before = 0;
	uint64_t              flags_after = 0;
	uint32_t              stack_balanced = 0;
	std::array<std::array<uint8_t, 32>, 16> ymm {};
};

constexpr uint64_t NormalMarker   = 0x4e4f524d414cull;
constexpr uint64_t DetachedMarker = 0x444554414348ull;
constexpr uint64_t UnlinkMarker   = 0x554e4c494e4bull;
constexpr uint64_t RbxSentinel    = 0x1122334455667788ull;
constexpr uint64_t RbpSentinel    = 0x2233445566778899ull;
constexpr uint64_t R13Sentinel    = 0x33445566778899aaull;
constexpr uint64_t R11Sentinel    = 0x445566778899aabbull;

constexpr auto MakeYmmSeeds() {
	std::array<std::array<uint8_t, 32>, 16> seeds {};
	for (size_t reg = 0; reg < seeds.size(); ++reg) {
		for (size_t byte = 0; byte < seeds[reg].size(); ++byte) {
			seeds[reg][byte] = static_cast<uint8_t>(reg * 29 + byte * 17 + 3);
		}
	}
	return seeds;
}
alignas(32) constexpr auto YmmSeeds = MakeYmmSeeds();

using Runner = KYTY_SYSV_ABI void (*)(Snapshot*, const Node*, const void*, const void*, uint8_t*);

Runner MakeRunner(Xbyak::CodeGenerator& code) {
	using namespace Xbyak::util;

	const auto* run = code.getCurr();
	// The generated runner switches to a private stack so its red-zone canaries
	// cannot overlap compiler frames. The thunk gets an ordinary call return slot.
	code.push(rbp);
	code.push(rbx);
	code.push(r12);
	code.push(r13);
	code.push(r14);
	code.push(r15);
	code.mov(r12, rdi);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, saved_rsp))], rsp);
	code.mov(r14, rdx);
	code.mov(r15, rsi);
	code.mov(rbx, RbxSentinel);
	code.mov(rbp, RbpSentinel);
	code.mov(r13, R13Sentinel);
	code.mov(r11, R11Sentinel);
	for (size_t i = 0; i < YmmSeeds.size(); ++i) {
		code.mov(rax, reinterpret_cast<uint64_t>(YmmSeeds[i].data()));
		code.vmovdqu(Xbyak::Ymm(static_cast<int>(i)), ptr[rax]);
	}
	code.lea(rsp, ptr[r8 + 144]); // call-site RSP is 16-byte aligned
	code.mov(r10, rcx);
	code.cmp(r10, r10); // Seed a known arithmetic-flags state immediately before the call.
	code.pushfq(); // This temporary is above the guarded red-zone.
	code.pop(r9);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, flags_before))], r9);
	code.call(r10);

	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, rax))], rax);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, rcx))], rcx);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, r11))], r11);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, rbx))], rbx);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, rbp))], rbp);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, r12))], r12);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, r13))], r13);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, r14))], r14);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, r15))], r15);
	for (size_t i = 0; i < YmmSeeds.size(); ++i) {
		code.vmovdqu(ptr[r12 + static_cast<int32_t>(offsetof(Snapshot, ymm) + i * 32)],
		             Xbyak::Ymm(static_cast<int>(i)));
	}
	code.pushfq(); // The thunk's return slot has been popped; this is outside the red-zone too.
	code.pop(r9);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, flags_after))], r9);
	code.lea(rdx, ptr[r8 + 144]);
	code.cmp(rsp, rdx);
	code.sete(al);
	code.movzx(eax, al);
	code.mov(dword[r12 + static_cast<int32_t>(offsetof(Snapshot, stack_balanced))], eax);
	code.mov(rax, qword[r12 + static_cast<int32_t>(offsetof(Snapshot, saved_rsp))]);
	code.mov(rsp, rax);
	code.pop(r15);
	code.pop(r14);
	code.pop(r13);
	code.pop(r12);
	code.pop(rbx);
	code.pop(rbp);
	code.ret();
	return reinterpret_cast<Runner>(const_cast<uint8_t*>(run));
}

void RunCase(Runner runner, const void* thunk, Node& node, const void* endpoint, uint64_t expected_prev,
             uint64_t expected_next, uint64_t expected_marker, uint64_t expected_count, uint64_t& counter,
             bool check_flags = true) {
	Snapshot snapshot {};
	alignas(16) std::array<uint8_t, 256> stack {};
	stack.fill(0xa5);
	constexpr size_t RedzoneStart = 32 + 8;
	constexpr size_t RedzoneSize  = 128;
	runner(&snapshot, &node, endpoint, thunk, stack.data() + 32);

	Check(snapshot.rax == expected_prev && snapshot.rcx == expected_next, "RAX=prev and RCX=next loads preserved");
	Check(snapshot.r11 == R11Sentinel, "R11 preserved on both continuations");
	Check(snapshot.marker == expected_marker, "correct continuation selected");
	Check(snapshot.stack_balanced == 1, "RSP balanced through the generated runner");
	Check(std::all_of(stack.begin() + RedzoneStart, stack.begin() + RedzoneStart + RedzoneSize,
	                  [](uint8_t byte) { return byte == 0xa5; }),
	      "guest red-zone canary untouched");
	if (check_flags) Check(snapshot.flags_before == snapshot.flags_after, "RFLAGS preserved through thunk");
	Check(snapshot.ymm == YmmSeeds, "all XMM/YMM registers, including upper halves, preserved");
	Check(snapshot.rbx == RbxSentinel && snapshot.rbp == RbpSentinel && snapshot.r13 == R13Sentinel,
	      "host nonvolatile GPR sentinels preserved");
	Check(snapshot.r12 == reinterpret_cast<uint64_t>(&snapshot), "host nonvolatile R12 preserved");
	Check(snapshot.r14 == reinterpret_cast<uint64_t>(endpoint) && snapshot.r15 == reinterpret_cast<uint64_t>(&node),
	      "host nonvolatile R14/R15 preserved");
	Check(counter == expected_count, "counter increments only for the observed detached endpoint");
}

void TestGeneratedThunk() {
	Xbyak::util::Cpu cpu;
	if (!cpu.has(Xbyak::util::Cpu::tAVX)) {
		std::puts("collision guard: generated YMM regression skipped (AVX unavailable)");
		return;
	}

	using namespace Xbyak::util;
	Xbyak::ClearError();
	Xbyak::CodeGenerator code(8192);
	const auto* normal_marker = code.getCurr();
	code.mov(r10, NormalMarker);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, marker))], r10);
	code.ret();
	const auto* normal_unlink = code.getCurr();
	Xbyak::Label tail_done;
	code.mov(qword[rax + 0x50], rcx); // Actual first instruction at NormalOffset.
	code.test(rcx, rcx);
	code.jz(tail_done); // Actual branch to DetachedOffset for a valid tail.
	code.mov(qword[rcx + 0x58], rax);
	code.L(tail_done);
	code.mov(r10, UnlinkMarker);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, marker))], r10);
	code.ret();
	const auto* detached = code.getCurr();
	code.mov(r10, DetachedMarker);
	code.mov(qword[r12 + static_cast<int32_t>(offsetof(Snapshot, marker))], r10);
	code.ret();
	alignas(8) uint64_t counter = 0;
	const auto* marker_thunk = code.getCurr();
	Guard::EmitThunk(code, normal_marker, detached, &counter);
	const auto* unlink_thunk = code.getCurr();
	Guard::EmitThunk(code, normal_unlink, detached, &counter);
	const auto runner = MakeRunner(code);
	code.ready();
	Check(!Xbyak::GetError(), "generated thunk and runner assembled");

	Node node {}, previous {}, next {};
	constexpr uint64_t PreviousCanary = 0x1111222233334444ull;
	constexpr uint64_t NextCanary     = 0x5555666677778888ull;
	previous.next_at_50 = PreviousCanary;
	next.prev_at_58     = NextCanary;
	node.next_at_50     = reinterpret_cast<uint64_t>(&next);
	node.prev_at_58     = reinterpret_cast<uint64_t>(&previous);
	RunCase(runner, unlink_thunk, node, reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&node) + 0x20),
	        node.prev_at_58, node.next_at_50, UnlinkMarker, 0, counter, false);
	Check(previous.next_at_50 == reinterpret_cast<uint64_t>(&next) &&
	          next.prev_at_58 == reinterpret_cast<uint64_t>(&previous),
	      "doubly linked normal continuation updates both neighbors");

	previous.next_at_50 = PreviousCanary;
	node.next_at_50     = 0;
	node.prev_at_58     = reinterpret_cast<uint64_t>(&previous);
	RunCase(runner, unlink_thunk, node, reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&node) + 0x20),
	        node.prev_at_58, 0, UnlinkMarker, 0, counter, false);
	Check(previous.next_at_50 == 0, "valid tail continuation updates previous.next to null");

	node.next_at_50 = reinterpret_cast<uint64_t>(&next);
	node.prev_at_58 = 0;
	RunCase(runner, marker_thunk, node, reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&node) + 0x20),
	        0, node.next_at_50, NormalMarker, 0, counter); // marker avoids the real null-prev store

	node.next_at_50 = 0;
	node.prev_at_58 = 0;
	RunCase(runner, marker_thunk, node, reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&node) + 0x20),
	        0, 0, DetachedMarker, 1, counter);
	RunCase(runner, marker_thunk, node, reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&node) + 0x28),
	        0, 0, NormalMarker, 1, counter); // both-null on a different endpoint remains on normal path
}
#endif
} // namespace

int main(int argc, char** argv) {
	Check(argc <= 2, "usage: demons_souls_collision_guard_tests [original-usa-self]");
	TestJumpEncoding();
	TestSignatureRejection();
	if (argc == 2) TestSignatureFixture(argv[1]);
#if defined(__x86_64__) || defined(_M_X64)
	TestGeneratedThunk();
#endif
	std::puts("Demon's Souls collision guard tests passed");
}
