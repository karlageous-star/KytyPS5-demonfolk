#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "live-counters.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <pthread.h>
#include <sched.h>
#elif defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

extern "C" {
// KYTY_HOT_PAGES (live; default 0 until measured): memoryTracker.cpp.
extern volatile std::atomic<uint32_t> kyty_local_hot_pages_mode;
}

namespace Libs::Graphics {

// KYTY_HOT_PAGES: a page the guest CPU writes again FaultsInFrame times within one guest frame (each a write fault, a
// protection change back and a re-protection at the next upload) stays writable and CPU-dirty: every upload over it
// copies it, no fault or protection change. Demon's Souls rewrites its constant rings several times a frame (Gates of
// Boletaria: ~900 write faults and 2,000-2,900 protection changes a frame, S47). Pages written once a frame never qualify
// (they would be uploaded at every use instead of once). Hot pages go back to tracking when the decay epoch moves, when
// the switch is off, and when the GPU writes them.
namespace HotPages {
inline std::atomic<uint32_t> g_frame {1}; // guest frames (TextureCache::RunGarbageCollector)
inline std::atomic<uint32_t> g_epoch {1}; // decay
constexpr uint32_t           FaultsInFrame = 3;
} // namespace HotPages

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		if (m_owner.load(std::memory_order_relaxed) == thread) {
			EXIT("recursive region tracking lock\n");
		}
		uint32_t spins = 0;
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			if (m_owner.load(std::memory_order_relaxed) == thread) {
				EXIT("recursive region tracking lock while contended\n");
			}
			// Wait with plain loads: repeated locked writes would keep the line away from the
			// owner and slow the very critical section being waited for. Past a short spin the
			// waiter yields its CPU: guest threads streaming textures fault into the texture
			// cache at once, and pure spinners starved a descheduled owner (the render thread,
			// 100+ ms frames with its samples parked inside the critical section).
			do {
				if (++spins < 512) {
					_mm_pause();
				} else {
					YieldCpu();
				}
			} while (m_lock.test(std::memory_order_relaxed));
		}
		m_owner.store(thread, std::memory_order_relaxed);
		// Odd while held: ReadShared retries a read that overlapped a holder.
		m_seq.store(m_seq.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
	}
	void unlock() noexcept {
		if (m_owner.load(std::memory_order_relaxed) != CurrentThread()) {
			EXIT("region tracking lock released by non-owner\n");
		}
		m_seq.store(m_seq.load(std::memory_order_relaxed) + 1, std::memory_order_release);
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}
	// A read of state only holders of this lock write, without taking it: valid when no holder
	// was inside while it ran (the sequence even and unchanged), else done again under the lock.
	// The query then costs no locked instruction and leaves the lock's line to the writers.
	template <typename Read>
	auto ReadShared(Read&& read) {
		const auto before = m_seq.load(std::memory_order_acquire);
		if ((before & 1u) == 0) {
			auto result = read();
			std::atomic_thread_fence(std::memory_order_acquire);
			if (m_seq.load(std::memory_order_relaxed) == before) return result;
		}
		lock();
		auto result = read();
		unlock();
		return result;
	}

	static void YieldCpu() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SwitchToThread();
#else
		sched_yield();
#endif
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// TEB ClientId.UniqueThread, what GetCurrentThreadId returns: one load instead of a
		// call into kernel32 on every lock and unlock.
		return static_cast<uint32_t>(__readgsqword(0x48));
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
	std::atomic_uint32_t m_seq {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	[[nodiscard]] uint64_t CpuModificationEpoch() const {
		// Pages marked clean whose write protection is still queued can change unseen:
		// nothing is provable until the worker applied it (and moved the epoch).
		if (m_deferred_protects.load(std::memory_order_acquire) != 0) return 0;
		return m_cpu_epoch.load(std::memory_order_acquire);
	}
	// KYTY_ASYNC_REPROTECT: a write protection of pages in this region is queued.
	void BeginDeferredProtection() noexcept { m_deferred_protects.fetch_add(1, std::memory_order_acq_rel); }
	[[nodiscard]] bool DeferredProtectionPending() const noexcept {
		return m_deferred_protects.load(std::memory_order_acquire) != 0;
	}
	// The upload worker (or the render thread as a fallback) applies it.
	void ApplyDeferredProtection(uint64_t vaddr, uint64_t size) {
		m_page_manager.ReapplyProtection(vaddr, size);
		m_cpu_epoch.fetch_add(1, std::memory_order_acq_rel);
		m_deferred_protects.fetch_sub(1, std::memory_order_acq_rel);
	}

	template <DirtySource source, bool all = false>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const auto& bits        = GetBits<source>();
		if constexpr (all) return bits.AllInRange(start, end);
		return bits.AnyInRange(start, end);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Gpu && enable) {
			// Hot pages the GPU now writes: tracked again (their bytes went up with the upload before this),
			// ahead of the conflict check below: a hot page is CPU-dirty by design.
			if (m_hot.AnyInRange(start, end)) {
				for (size_t page = start; page < end; ++page) {
					if (m_hot.Get(page)) {
						m_hot.Unset(page);
						m_cpu_dirty.Unset(page);
					}
				}
				UpdateCpuProtection<true>();
			}
		}
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.AnyInRange(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (m_cpu_dirty.AnyInRange(start, end)) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Cpu && enable) {
			NoteCpuWrites(start, end);
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			if constexpr (enable) {
				// Invalidate cached clean proofs before another CPU thread can write
				// through the relaxed host protection. A cache miss takes this lock.
				m_cpu_epoch.fetch_add(1, std::memory_order_release);
				BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
			}
			UpdateCpuProtection<!enable>();
		} else {
			UpdateGpuProtection<enable>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (!GetBits<source>().AnyInRange(start, end)) return;
		RegionBits mask(GetBits<source>(), start, end);
		if constexpr (clear) {
			GetBits<source>().UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu && clear) {
			// Hot pages stay dirty (and writable): uploaded again at the next use. Their writes are not seen, so the
			// clean proofs and the BDA dirty regions take them as written again.
			if (m_hot.Any() && RefreshHot() && m_hot.AnyInRange(start, end)) {
				for (size_t page = start; page < end; ++page) {
					if (m_hot.Get(page)) m_cpu_dirty.Set(page);
				}
				m_cpu_epoch.fetch_add(1, std::memory_order_release);
				BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
			}
			UpdateCpuProtection<true>();
			ForEachRange(mask, std::forward<Func>(func));
			return;
		}
		if constexpr (source == DirtySource::Gpu && clear) {
			UpdateGpuProtection<false>();
		}
		ForEachRange(mask, std::forward<Func>(func));
	}

	// Own cache line: guest fault handling updates the epoch and bitmaps below.
	alignas(64) TrackingSpinLock lock;

private:
	template <bool track>
	void UpdateCpuProtection() {
		auto mask  = m_cpu_dirty ^ m_writable;
		m_writable = m_cpu_dirty;
		if (mask.None()) {
			return;
		}
		LiveCounters::Add(track ? LiveCounters::Reprotect : LiveCounters::Unprotect);
		LiveCounters::Add(track ? LiveCounters::ReprotectPages : LiveCounters::UnprotectPages, mask.Count());
		if (track && LiveCounters::g_granules_on.load(std::memory_order_relaxed)) {
			for (const auto [start, end]: mask) {
				LiveCounters::AddGranule(m_cpu_addr + start * TRACKER_PAGE_SIZE, LiveCounters::GReprotectPages, end - start);
			}
		}
		m_page_manager.UpdatePageWatchersForRegion<track>(m_cpu_addr, mask);
	}

	template <bool track>
	void UpdateGpuProtection() {
		auto readable = ~m_gpu_dirty;
		auto mask     = readable ^ m_readable;
		m_readable    = readable;
		if (mask.None()) {
			return;
		}
		if constexpr (track) {
			m_page_manager.UpdatePageWatchersForRegion<true, true>(m_cpu_addr, mask);
		} else {
			m_page_manager.UpdatePageWatchersForRegion<false, true>(m_cpu_addr, mask);
		}
	}

	// Whether hot pages are still kept (false: the switch is off or the decay epoch moved, and they were dropped).
	bool RefreshHot() {
		const uint32_t epoch = HotPages::g_epoch.load(std::memory_order_relaxed);
		if (kyty_local_hot_pages_mode.load(std::memory_order_relaxed) == 0 || m_hot_epoch != epoch) {
			m_hot.Clear();
			m_hot_epoch = epoch;
			return false;
		}
		return true;
	}

	// Pages going from clean to CPU-dirty (a write fault or an invalidation): counted per guest frame.
	void NoteCpuWrites(size_t start, size_t end) {
		if (kyty_local_hot_pages_mode.load(std::memory_order_relaxed) == 0) {
			if (m_hot.Any()) m_hot.Clear();
			return;
		}
		(void)RefreshHot();
		if (!m_fault_log) m_fault_log = std::make_unique<std::array<uint16_t, TRACKER_REGION_PAGES>>();
		const auto frame = static_cast<uint16_t>(HotPages::g_frame.load(std::memory_order_relaxed) & 0x0fffu);
		for (size_t page = start; page < end; ++page) {
			if (m_cpu_dirty.Get(page)) continue;
			auto&          entry = (*m_fault_log)[page];
			const uint32_t count = (entry & 0x0fffu) == frame ? (entry >> 12u) + 1u : 1u;
			entry                = static_cast<uint16_t>((std::min<uint32_t>(count, 15u) << 12u) | frame);
			if (count >= HotPages::FaultsInFrame && !m_hot.Get(page)) {
				m_hot.Set(page);
				LiveCounters::Add(LiveCounters::HotPages);
			}
		}
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	template <typename Func>
	void ForEachRange(const RegionBits& bits, Func&& func) const {
		for (const auto [start, end]: bits) {
			func(m_cpu_addr + start * TRACKER_PAGE_SIZE, (end - start) * TRACKER_PAGE_SIZE);
		}
	}

	alignas(64) PageManager& m_page_manager;
	uint64_t              m_cpu_addr = 0;
	std::atomic<uint64_t> m_cpu_epoch {1};
	std::atomic<uint32_t> m_deferred_protects {0};
	RegionBits            m_cpu_dirty;
	RegionBits            m_gpu_dirty;
	RegionBits            m_writable;
	RegionBits            m_readable;
	RegionBits            m_hot; // KYTY_HOT_PAGES
	uint32_t              m_hot_epoch = 0;
	// Per page: the guest frame (low 12 bits) and its clean-to-dirty count (high 4); made at the first counted write.
	std::unique_ptr<std::array<uint16_t, TRACKER_REGION_PAGES>> m_fault_log;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
