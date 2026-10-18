#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"
#include "live-census.h"
#include "live-counters.h"
#include "async-upload.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"

#include <bit>
#include <chrono>
#include <optional>
#include <utility>
#include <x86intrin.h>

#include "common/assert.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "kernel/memory.h"

extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_bda_dirty_regions_mode {0};
// 1: watchers removed while memory is released skip the host protection change there
// (PageManager::SetUnmappingRange); a write fault on a page nothing watches unprotects it.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_unmap_protect_skip_mode {0};
// KYTY_WINDOW_STREAM (performance-switches.h, opt-in): 1 takes the next 64 KiB at once for a write fault in the 16 KiB
// window right after the one the same thread's last window fault ended in, within 2 ms (a sequential stream of writes).
// 0 (default) keeps the 16 KiB window. Opt-in because it invalidates 4x the pages (uploads +9%, on the upload worker).
// Measured by chenxiao 87280083 (1-1 route, 120 cap, A-B-A over 29 points): window faults -13.6%, fps +0.7%, 1% low +3.8%.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_window_stream_mode {0};
// KYTY_FAULT_SCAN_FRAME (live): 1 scans the BDA fault bitmap (FaultManager::ProcessFaultBuffer: 8 MiB read and cleared,
// ~170 us of GPU time) once a frame (at the GPU flip, else every 32 submissions) instead of after every submission
// that ran a BDA shader (7 a frame in the tunnel). Its records only mark buffers for the per-frame retirement.
// Default 1 (S3, in-session: +0.35 FPS, ahead in all five rounds); 0 scans after every such submission.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_fault_scan_frame_mode {1};
}

namespace Libs::Graphics {

GpuResourceManager::GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler), m_buffer_cache(graphics, scheduler, m_page_manager, m_texture_cache, this),
      m_texture_cache(graphics, scheduler, m_page_manager, m_buffer_cache) {}

GpuResourceManager::~GpuResourceManager() = default;

void GpuResourceManager::BeginAliasWrite() noexcept {
	{
		const auto old = m_preparation_alias_epoch.fetch_add((uint64_t{1} << 32u) + 1u,
		    std::memory_order_acq_rel);
		EXIT_IF(uint32_t(old >> 32u) == UINT32_MAX || uint32_t(old) == UINT32_MAX);
	}
}

void GpuResourceManager::EndAliasWrite() noexcept {
	{
		const auto old = m_preparation_alias_epoch.fetch_sub(1u, std::memory_order_release);
		EXIT_IF(uint32_t(old) == 0);
	}
}

uint64_t GpuResourceManager::PreparationAliasEpoch() const noexcept {
	const auto value = m_preparation_alias_epoch.load(std::memory_order_acquire);
	return uint32_t(value) == 0 ? value : 0;
}

bool GpuResourceManager::TryInvalidateCpuWriteWindow(uint64_t fault) {
	// A small window amortizes faults from sequential CPU writes. Expanding a
	// fault is allowed only for mapped, unaliased pages with no GPU/image owner.
	constexpr uint64_t window_size  = 4 * TRACKER_PAGE_SIZE;
	constexpr uint64_t stream_size  = 16 * TRACKER_PAGE_SIZE;
	const auto         window_begin = fault & ~(window_size - 1);
	if (kyty_local_window_stream_mode.load(std::memory_order_relaxed) != 0) {
		// The window after the one this thread's last window fault ended in, within 2 ms (~6.4M TSC ticks at 3.2 GHz):
		// a stream, whose next 64 KiB goes at once (the next fault in it is the window after that).
		thread_local uint64_t last_end = 0, last_tsc = 0;
		const uint64_t        now      = __rdtsc();
		const bool            stream   = window_begin == last_end && now - last_tsc < 6'400'000;
		if (stream && TryInvalidateCpuWriteWindow(fault, window_begin, stream_size)) {
			last_end = window_begin + stream_size;
			last_tsc = now;
			return true;
		}
		if (!TryInvalidateCpuWriteWindow(fault, window_begin, window_size)) return false;
		last_end = window_begin + window_size;
		last_tsc = now;
		return true;
	}
	return TryInvalidateCpuWriteWindow(fault, window_begin, window_size);
}

bool GpuResourceManager::TryInvalidateCpuWriteWindow(uint64_t fault, uint64_t window_begin, uint64_t window_size) {
	GuestRange selected {};
	uint64_t   mapping_epoch;
	{
		std::shared_lock mapped_lock(m_mapped_ranges_mutex);
		mapping_epoch = m_mapping_epoch;
		m_mapped_ranges.ForEachIntersection(window_begin, window_size, [&](RangeSet::Range range) {
			if (fault >= range.address && fault - range.address < range.size) {
				const auto begin =
				    (range.address + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
				const auto end = (range.address + range.size) & ~(TRACKER_PAGE_SIZE - 1);
				if (begin < end) selected = {begin, end - begin};
			}
		});
	}
	if (selected.size <= TRACKER_PAGE_SIZE || fault < selected.address ||
	    fault - selected.address >= selected.size ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(selected.address, selected.size))
		return false;
	// Kernel backing queries must not run while holding the resource-map lock.
	std::shared_lock mapped_lock(m_mapped_ranges_mutex);
	if (mapping_epoch != m_mapping_epoch ||
	    !m_mapped_ranges.Contains(selected.address, selected.size))
		return false;
	return m_buffer_cache.TryInvalidateCpuWriteWindow(fault, selected.address, selected.size);
}

bool GpuResourceManager::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		// (A write to a watched page of a view a partial unmap is mapping again: unseen while such a view came back
		// writable.)
		const bool in_remap = LibKernel::Memory::InRemapWindow(fault_vaddr);
		if (TryInvalidateCpuWriteWindow(fault_vaddr)) {
			LiveCounters::Add(LiveCounters::WindowFault);
			if (in_remap) LiveCounters::Add(LiveCounters::RemapWindowWrites);
			return true;
		}
		LiveCounters::Add(LiveCounters::WriteFault);
		// No watcher holds the page, so no cache expects this fault: the protection outlived its
		// watchers and would fault forever. It gets the guest's own protection back.
		if (m_page_manager.RestoreIfUnwatched(fault_vaddr)) {
			LiveCounters::Add(LiveCounters::StaleProtectRepairs);
			return true;
		}
		if (in_remap) LiveCounters::Add(LiveCounters::RemapWindowWrites);
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		LiveCounters::Add(LiveCounters::ReadFault);
		if (LiveCensus::g_render) LiveCounters::Add(LiveCounters::RenderReadFaults);
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool GpuResourceManager::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool GpuResourceManager::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void GpuResourceManager::MapMemory(uint64_t vaddr, uint64_t size) {
	const auto map = [this, vaddr, size] {
		m_texture_cache.MapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Add(vaddr, size);
		++m_mapping_epoch;
		if (size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size)) ++m_bda_ranges_epoch;
	};
	if (m_gpu) {
		// A pending readback writes its backing when it finishes, which a new mapping aliasing
		// mapped memory would show; memory nothing else maps has no readback pending.
		const bool aliased = !LibKernel::Memory::IsUniqueGuestBackingRange(vaddr, size);
		m_gpu->SendCommandSync([this, map, aliased] {
			if (aliased) m_buffer_cache.DrainGuestReadback();
			map();
		});
	} else {
		map();
	}
}

void GpuResourceManager::UnmapMemory(uint64_t vaddr, uint64_t size, bool releasing) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size, releasing] {
		// Released memory loses its host protection with the mapping: the watchers removed
		// below need not restore it page run by page run first.
		struct UnmappingRange {
			explicit UnmappingRange(uint64_t begin, uint64_t end) { PageManager::SetUnmappingRange(begin, end); }
			~UnmappingRange() { PageManager::SetUnmappingRange(0, 0); }
		};
		std::optional<UnmappingRange> unmapping;
		if (releasing && kyty_local_unmap_protect_skip_mode.load(std::memory_order_relaxed) != 0) {
			unmapping.emplace(vaddr, vaddr + size);
		}
		// Readbacks write guest memory when they finish: the ones into the range finish first
		// (the others do not touch it).
		m_buffer_cache.DrainGuestReadback(vaddr, size);
		// Pending upload jobs on the range read its backing (or change its protection): they finish before
		// it changes owner. Jobs elsewhere keep running.
		AsyncUpload::DrainRange(vaddr, size);
		// The GPU only works on the caches' own buffers and images, which are freed after the
		// work in flight completes (deferred operations), so the unmap need not wait for the GPU
		// (it used to finish all of it: up to a frame of GPU work, for every guest unmap). What
		// writes guest memory after the GPU are the readbacks drained above and GPU-written
		// images' downloads, which complete on the priority thread: wait for those over the range.
		// (Deferred fault-buffer processing skips ranges no longer mapped.)
		const auto download_tick = m_scheduler.Active() ? m_texture_cache.PendingDownloadTick(vaddr, size) : 0;
		if (download_tick != 0) {
			const auto start = std::chrono::steady_clock::now();
			if (!m_scheduler.IsFree(download_tick)) m_scheduler.Wait(download_tick);
			m_scheduler.WaitPriorityOperations(download_tick);
			LiveCounters::Add(LiveCounters::UnmapFinishes);
			LiveCounters::Add(LiveCounters::UnmapFinishUs,
			                  static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                            std::chrono::steady_clock::now() - start)
			                                            .count()));
		}
		const bool registered = size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		const auto epoch = ++m_mapping_epoch;
		if (registered) ++m_bda_ranges_epoch;
		if (size == 0) return;
		constexpr uint64_t leaf_mask = (uint64_t {1} << UnmapLeafBits) - 1;
		for (uint64_t g = vaddr >> UnmapGranuleBits, last = (vaddr + size - 1) >> UnmapGranuleBits; g <= last; ++g) {
			auto& leaf = m_unmap_epochs[g >> UnmapLeafBits];
			if (!leaf) leaf = std::make_unique<uint64_t[]>(leaf_mask + 1);
			leaf[g & leaf_mask] = epoch;
		}
	};
	if (m_gpu == nullptr) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

bool GpuResourceManager::UnmappedSince(uint64_t epoch, uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0) return false;
	constexpr uint64_t leaf_mask = (uint64_t {1} << UnmapLeafBits) - 1;
	for (uint64_t g = vaddr >> UnmapGranuleBits, last = (vaddr + size - 1) >> UnmapGranuleBits; g <= last; ++g) {
		const auto leaf = m_unmap_epochs.find(g >> UnmapLeafBits);
		if (leaf == m_unmap_epochs.end()) {
			g |= leaf_mask; // never unmapped in this leaf
			continue;
		}
		if (leaf->second[g & leaf_mask] > epoch) return true;
	}
	return false;
}

void GpuResourceManager::RefreshBdaRanges() {
	const auto registered = m_buffer_cache.RegistrationEpoch();
	if (m_bda_mapping_epoch == m_bda_ranges_epoch && m_bda_registration_epoch == registered) return;
	LiveCounters::Add(LiveCounters::BdaRebuilds);
	std::vector<RangeSet::Range> ranges;
	m_buffer_cache.CollectMappedRegisteredRanges(m_mapped_ranges, ranges);
	// An unchanged request keeps its proof: SynchronizeRegionRequest checks it against its region's CPU and
	// registration epochs (an unmap marks the range CPU-dirty, which moves the CPU epoch).
	std::swap(m_bda_region_requests, m_old_bda_region_requests);
	m_bda_region_requests.clear();
	size_t old = 0;
	for (const auto& range : ranges) {
		const auto end = range.address + range.size;
		for (auto start = range.address; start < end;) {
			const auto finish = std::min(end, (start / TRACKER_REGION_SIZE + 1) * TRACKER_REGION_SIZE);
			auto& request = m_bda_region_requests.emplace_back(BufferCache::SyncRegionRequest {start, finish - start});
			while (old < m_old_bda_region_requests.size() && m_old_bda_region_requests[old].address < start) ++old;
			if (old < m_old_bda_region_requests.size() && m_old_bda_region_requests[old].address == start &&
			    m_old_bda_region_requests[old].size == request.size) {
				request = m_old_bda_region_requests[old];
			} else if (start / TRACKER_REGION_SIZE < BdaDirtyRegions::Regions) {
				// A new request starts unproven.
				BdaDirtyRegions::Mark(start / TRACKER_REGION_SIZE);
			}
			start = finish;
		}
	}
	m_bda_mapping_epoch = m_bda_ranges_epoch;
	m_bda_registration_epoch = registered;
}

bool GpuResourceManager::PrepareBdaReadRanges(std::span<const GuestRange> ranges) {
	if (ranges.empty() || ranges.size() > 128 ||
	    std::ranges::any_of(ranges, [](const auto& range) { return !range.Valid(); })) return false;
	std::shared_lock lock(m_mapped_ranges_mutex);
	RefreshBdaRanges();
	LiveCounters::Add(LiveCounters::BdaRangeCalls);
	LiveCounters::Add(LiveCounters::BdaRanges, ranges.size());
	for (const auto range : ranges) LiveCounters::Add(LiveCounters::BdaRangeMiB, range.size >> 20u);
	if (kyty_local_bda_dirty_regions_mode.load(std::memory_order_relaxed) != 0) {
		// KYTY_BDA_DIRTY_REGIONS: only regions whose bit is set can hold a stale request.
		for (const auto range : ranges) SynchronizeDirtyBdaRegions(range);
		m_fault_process_pending = true;
		m_bda_used              = true;
		return true;
	}
	for (const auto range : ranges) {
		auto it = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), range.address,
		    [](const auto& request, uint64_t begin) { return request.address + request.size <= begin; });
		for (; it != m_bda_region_requests.end() && it->address < range.End(); ++it)
			m_buffer_cache.SynchronizeRegionRequest(*it);
	}
	m_fault_process_pending = true;
	m_bda_used              = true;
	return true;
}

void GpuResourceManager::SynchronizeDirtyBdaRegions(GuestRange range) {
	const auto first = range.address / TRACKER_REGION_SIZE;
	if (first >= BdaDirtyRegions::Regions) return;
	const auto last = std::min((range.End() - 1) / TRACKER_REGION_SIZE, BdaDirtyRegions::Regions - 1);
	for (auto word = first / 64; word <= last / 64; ++word) {
		auto mask = ~uint64_t {0};
		if (word == first / 64) mask &= ~uint64_t {0} << (first % 64);
		if (word == last / 64) mask &= ~uint64_t {0} >> (63 - last % 64);
		auto& bits = BdaDirtyRegions::g_bits[word];
		for (auto pending = bits.load(std::memory_order_acquire) & mask; pending != 0; pending &= pending - 1) {
			const auto bit = uint64_t {1} << std::countr_zero(pending);
			// Clear before synchronizing: a CPU write after the epoch read sets it again.
			if ((bits.fetch_and(~bit, std::memory_order_acq_rel) & bit) == 0) continue;
			const auto region_begin = (word * 64 + static_cast<uint64_t>(std::countr_zero(pending))) * TRACKER_REGION_SIZE;
			auto it = std::lower_bound(m_bda_region_requests.begin(), m_bda_region_requests.end(), region_begin,
			    [](const auto& request, uint64_t begin) { return request.address + request.size <= begin; });
			bool unproven = false;
			for (; it != m_bda_region_requests.end() && it->address < region_begin + TRACKER_REGION_SIZE; ++it) {
				m_buffer_cache.SynchronizeRegionRequest(*it);
				unproven |= it->cpu_epoch == 0;
			}
			// A request left unproven by its own uploads' queued write protection needs no other visit:
			// a write before the protection lands is in the copies taken after it, and a write after it
			// faults (and marks the region).
			if (unproven && !m_buffer_cache.DeferredProtectionPending(region_begin))
				BdaDirtyRegions::Mark(region_begin / TRACKER_REGION_SIZE);
		}
	}
}

void GpuResourceManager::PrepareBda() {
	// Unknown shader addresses cannot prove disjointness from an in-flight readback.
	LiveCounters::Add(LiveCounters::BdaFullSyncs);
	m_buffer_cache.DrainGuestReadback();
	std::shared_lock lock(m_mapped_ranges_mutex);
	RefreshBdaRanges();
	for (auto& request : m_bda_region_requests) m_buffer_cache.SynchronizeRegionRequest(request);
	m_fault_process_pending = true;
	m_bda_used              = true;
}

void GpuResourceManager::EndSubmission() {
	if (m_fault_process_pending &&
	    (kyty_local_fault_scan_frame_mode.load(std::memory_order_relaxed) == 0 || m_gpu_flip_pending ||
	     ++m_fault_process_skips >= 32)) {
		m_fault_process_pending = false;
		m_fault_process_skips   = 0;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	// Only CPU flips (sceVideoOutSubmitFlip) reached AdvanceFrame, so for a game that flips from the
	// GPU the texture and buffer collectors never ran and video memory filled up until an image could
	// not be created (Demon's Souls on an 8 GB card, X91).
	if (std::exchange(m_gpu_flip_pending, false)) {
		AdvanceFrame();
	}
}

void GpuResourceManager::AdvanceFrame() {
	m_texture_cache.AdvanceFrame();
	// Collection runs per frame: per submission (dozens a frame here) its ages of 16 to 160 ticks
	// were a few frames, and it deleted what the next frames used again.
	m_texture_cache.RunGarbageCollector();
	// A shader that reads guest memory through the BDA page table touches no buffer in the LRU: a
	// buffer collected under it faults back at once, and each registration invalidates every BDA
	// region proof. Buffers are collected only after a frame without such shaders.
	m_buffer_cache.RunGarbageCollector(!std::exchange(m_bda_used, false));
}

} // namespace Libs::Graphics
