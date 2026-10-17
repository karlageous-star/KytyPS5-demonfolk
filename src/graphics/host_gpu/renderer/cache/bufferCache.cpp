#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "slow-log.h"

#include "native-buffer-residency.h"
#include "live-census.h"
#include "live-counters.h"
#include "live-trace.h"
#include "native-resource-state.h"
#include "async-upload.h"
#include "readback-queue.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "gpu_tiler_shaders/lod_stats_pack_spv.h"

extern "C" {
// 1: a synchronous download retires only guest readbacks overlapping its pages.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_sync_readback_scoped_drain_mode {0};
// 1: synchronous reads download only their required tracker pages, without 512 KiB prefetch.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_sync_readback_narrow_mode {0};
volatile std::atomic<uint32_t> kyty_local_buffer_residency_mode {0};
volatile std::atomic<uint32_t> kyty_local_copy_feedback_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_detach_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_write_readback_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_slots_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_upload_mode {0};
// 1: RangeSet fast paths (rangeSet.h) for the GPU-modified range set.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_range_set_fast_mode {0};
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_async_reprotect_mode {0};
// Live switch (same-process A/B): ObtainBuffer's memo of repeated shader writes.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_write_memo_mode {1};
// 1: a guest readback whose window meets an image falls back to the request's own pages.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_narrow_mode {0};
// KYTY_READBACK_PAGE_RUNS (default 1; 0 for a same-process A/B): KYTY_READBACK_QUEUE readiness over the page runs a
// guest readback writes back (chenxiao 5f9452f5) instead of its whole window.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_page_runs_mode {1};
// KYTY_READBACK_QUEUE (readback-queue.h): 1: readbacks whose bytes no in-flight GPU work writes
// copy on the transfer queue; 2: also those whose last writer is submitted but still running;
// 3: as 1, and each such copy is repeated on the graphics queue and compared (verification).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_readback_queue_mode {0};
// 1: pack the LOD report on the GPU instead of a CPU wait and copy.
volatile std::atomic<uint32_t> kyty_local_async_lod_stats_mode {0};
// 1: shader constants stream through a host-visible upload ring.
[[gnu::used]] volatile std::atomic_uint32_t kyty_local_stream_upload_mode {0};
// KYTY_BUFFER_RETIRE (performance-switches.h): 1 retires the buffers no binding used for a while
// (RetireStaleBuffers); 0 never retires one.
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_buffer_retire_mode {1};
extern volatile std::atomic_uint32_t kyty_local_frame_pipeline_mode;
}

#include "common/assert.h"
#include "common/guardedCopy.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// kernel32 export, declared here instead of including <windows.h> (KYTY_SYNC_LOG stacks).
extern "C" __declspec(dllimport) unsigned short __stdcall RtlCaptureStackBackTrace(unsigned long, unsigned long,
                                                                                   void**, unsigned long*);
#endif

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

bool SmallUploadRingEnabled() {
	const auto* value = std::getenv("KYTY_RAM_SMALL_UPLOAD_RING");
	return value != nullptr && std::strcmp(value, "1") == 0;
}

uint64_t UploadRingSize() {
	return SmallUploadRingEnabled() ? 128 * MiB : 512 * MiB;
}


uint32_t GpuWriteLogIndexMode() {
	static const uint32_t mode = [] {
		const auto* value = std::getenv("KYTY_GPU_WRITE_LOG_INDEX");
		if (value == nullptr) return 0u;
		if (std::strcmp(value, "1") == 0) return 1u;
		if (std::strcmp(value, "2") == 0) return 2u;
		return 0u;
	}();
	return mode;
}

bool IsLargeGpuWrite(uint64_t begin, uint64_t end) {
	constexpr uint64_t GranuleShift = 16;
	constexpr uint64_t MaxIndexedWriteGranules = 64;
	return begin < end && ((end - 1) >> GranuleShift) - (begin >> GranuleShift) + 1 > MaxIndexedWriteGranules;
}

// Hash 64 KiB guest granules into the fixed-size table. A collision stores a later
// absolute sequence and therefore only makes the query scan farther backward.
size_t GpuWriteIndexSlot(uint64_t address) {
	const uint64_t granule = address >> 16u;
	return static_cast<size_t>(granule * 11400714819323198485ull) & ((size_t {1} << 16u) - 1);
}

// AsyncUpload call (context: the guest address): an image span with unmapped parts (a streamed
// texture pool) into staging, the mapped parts as they are and the rest as zeros.
void ReadMappedOrZeroCall(void* context, uint64_t destination, uint64_t size) {
	auto* target = reinterpret_cast<uint8_t*>(destination);
	if (!Libs::LibKernel::Memory::TryReadMappedOrZero(reinterpret_cast<uint64_t>(context), target, size)) {
		std::memset(target, 0, size);
	}
}

// DownloadBufferMemory can process priority operations while waiting. Its
// exact intervals are retired after the entire copy batch, so a reentrant
// query must use the page tracker until that transaction has finished.
thread_local uint32_t native_buffer_download_depth = 0;
struct NativeBufferDownloadScope {
	NativeBufferDownloadScope() { ++native_buffer_download_depth; }
	~NativeBufferDownloadScope() { --native_buffer_download_depth; }
};

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

struct BufferCache::DownloadCopy {
	Buffer*  buffer        = nullptr;
	uint64_t source_offset = 0;
	uint64_t address       = 0;
	uint64_t size          = 0;
};

struct BufferCache::GuestReadback {
	struct Part { uint64_t address, size, offset; };
	std::vector<Part> parts;
	std::vector<GuestRange> pages;
	uint64_t begin = 0, size = 0, tick = 0, packed_size = 0;
	// KYTY_READBACK_QUEUE: the copy-engine value that completes the copy (0: graphics queue
	// at `tick`), and the graphics tick of the last GPU write it waited for. Mode 3 also copies
	// on the graphics queue at `tick` into `verify`.
	uint64_t queue_value = 0, producer_tick = 0;
	Buffer* verify = nullptr;
	size_t slot = 0;
	Buffer* download = nullptr;
	std::atomic<bool> copying {false};
	std::atomic<bool> copied {false};
	// Detach protocol: the reader claims the copy (Pending -> Copying) after the GPU
	// finished; the GPU thread may detach a request whose copy has not started
	// (Pending -> Detached). A detached copy never writes the backing: its pages stay
	// GPU-owned, so the reader faults again and reads the newer GPU state.
	enum State : uint32_t { Pending, Copying, Detached };
	std::atomic<uint32_t> state {Pending};
};

// With the frame pipeline, guest reads of GPU-written memory are copied out
// asynchronously; read-only GPU bindings may overlap the copy.
static bool GuestReadbacksEnabled() {
	return kyty_local_frame_pipeline_mode.load(std::memory_order_relaxed) != 0;
}

std::shared_ptr<BufferCache::GuestReadback> BufferCache::BeginGuestReadback(
    uint64_t address, uint64_t size, bool* completed) {
	if (completed) *completed = false;
	constexpr uint64_t WindowSize = 512 * 1024;
	constexpr uint64_t Capacity = 2 * WindowSize;
	if (!m_resources || !GuestRange {address, size}.Valid() || size > WindowSize) {
		LiveCounters::Add(LiveCounters::RbRejectSize);
		return {};
	}
	if (!GuestReadbacksEnabled()) return {};
	for (size_t slot = 0; slot < GuestReadbackSlots; ++slot) {
		const auto& pending = m_guest_readbacks[slot];
		if (!pending) continue;
		if (pending->copied.load(std::memory_order_acquire)) {
			FinishGuestReadback(slot);
			continue;
		}
		if (std::ranges::any_of(pending->pages, [&](const GuestRange& page) {
			return address >= page.address && address < page.End() && size <= page.End() - address;
		})) {
			return pending;
		}
	}
	// A request spanning several pending page spans cannot share just one ticket.
	DrainGuestReadback(address, size);
	auto& buffer = m_slot_buffers[FindBuffer(address, size)];
	auto begin = std::max(address & ~(WindowSize - 1), buffer.CpuAddress());
	auto end = std::min(std::max(begin + WindowSize, address + size),
	                          buffer.CpuAddress() + buffer.Size());
	if (!m_resources->IsMapped(begin, end - begin) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(begin, end - begin)) {
		LiveCounters::Add(LiveCounters::RbRejectBacking);
		return {};
	}
	// Only an image the GPU wrote holds bytes the buffer lacks. Other images over the pages
	// (textures the game streams in, images the buffer is newer than) take no part in the
	// copy, as in the synchronous download, and FindImage drains the copy before any use of
	// them. Refusing them sent 1-byte guest reads of texture pages (~5 a frame) to a
	// render-thread download of ~70 us each.
	if (m_texture_cache.HasGpuWrittenImageOverlap(begin, end - begin)) {
		// KYTY_READBACK_NARROW: the widening is speculative; an image elsewhere in the window
		// leaves the request's own pages, if no image covers them, to an asynchronous copy.
		const bool narrow = kyty_local_readback_narrow_mode.load(std::memory_order_relaxed) != 0;
		if (narrow) {
			begin = std::max(address & ~(TRACKER_PAGE_SIZE - 1), buffer.CpuAddress());
			end   = std::min((address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1),
			               buffer.CpuAddress() + buffer.Size());
		}
		if (!narrow || m_texture_cache.HasGpuWrittenImageOverlap(begin, end - begin)) {
			LiveCounters::Add(LiveCounters::RbRejectImage);
			return {};
		}
	}
	RangeSet available_pages;
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin,
	    [&](uint64_t a, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, a, bytes, "guest readback");
	    },
	    [&](uint64_t a, uint64_t bytes) noexcept {
		    available_pages.Add(a, bytes);
	    });
	// Widening is speculative: pages another ticket already covers must not cause
	// this unrelated request to wait or download the same bytes a second time.
	for (const auto& pending: m_guest_readbacks) {
		if (pending) for (const auto& page: pending->pages)
			available_pages.Subtract(page.address, page.size);
	}
	// KYTY_READBACK_QUEUE (chenxiao 5f9452f5): the bytes written back are those of these pages (the window's others are
	// at most copied as envelope fill, never written back): they take the transfer queue while no unfinished GPU write
	// overlaps them, whatever other writes in the window do (at Latria the culling's next stage writing the window sent
	// ~4 readbacks a frame of final bytes behind the whole graphics queue, ~1.2 ms each). When one does but the
	// requested pages' bytes are final, the 64 KiB pieces it overlaps stay GPU-owned for a later request.
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	available_pages.ForEach([&](uint64_t a, uint64_t e) { runs.emplace_back(a, e); });
	if (const auto queue_mode = kyty_local_readback_queue_mode.load(std::memory_order_relaxed);
	    (queue_mode == 1 || queue_mode == 3) && kyty_local_readback_page_runs_mode.load(std::memory_order_relaxed) != 0 &&
	    m_graphics.readback_queue != nullptr && m_gpu_writes_on &&
	    m_gpu_writes_from != UINT64_MAX && !runs.empty()) {
		auto& master = m_scheduler.GetMasterSemaphore();
		master.Refresh();
		const uint64_t completed  = master.KnownGpuTick();
		const uint64_t page_begin = address & ~(TRACKER_PAGE_SIZE - 1);
		const uint64_t page_end   = (address + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
		if (completed >= m_gpu_writes_from && InflightWriteTick(page_begin, page_end, completed, true) == 0) {
			constexpr uint64_t Piece = uint64_t {1} << 16u;
			const auto         drop  = [&](uint64_t from, uint64_t to) {
				if (from < to && InflightWriteTick(from, to, completed, true) != 0) available_pages.Subtract(from, to - from);
			};
			bool narrowed = false;
			for (const auto& [run_begin, run_end]: runs) {
				if (InflightWriteTick(run_begin, run_end, completed, true) == 0) continue;
				narrowed = true;
				// (By piece, the requested pages apart.)
				for (uint64_t at = run_begin; at < run_end;) {
					const uint64_t next = std::min(run_end, (at / Piece + 1) * Piece);
					if (next <= page_begin || at >= page_end) {
						drop(at, next);
					} else {
						drop(at, std::max(at, page_begin));
						drop(std::min(next, page_end), next);
					}
					at = next;
				}
			}
			if (narrowed) {
				LiveCounters::Add(LiveCounters::RbNarrowed);
				runs.clear();
				available_pages.ForEach([&](uint64_t a, uint64_t e) { runs.emplace_back(a, e); });
			}
		}
	}
	std::vector<DownloadCopy> copies;
	std::vector<GuestRange> pages;
	available_pages.ForEach([&](uint64_t a, uint64_t end) {
		pages.push_back({a, end - a});
		m_gpu_modified_ranges.ForEachIntersection(a, end - a, [&](RangeSet::Range range) {
			copies.push_back({&buffer, buffer.Offset(range.address), range.address, range.size});
		});
	});
	if (copies.empty()) {
		if (completed) *completed = true;
		return {};
	}
	// The copies' envelopes in packed regions, one region for envelopes less than 64 KiB apart (the
	// bytes between are copied too and never written back): GPU writes split a window into up to
	// thousands of small ranges, a copy command each otherwise (seconds of driver calls in one frame).
	// The parts place each copy's bytes in the download.
	struct Envelope {
		uint64_t source, size, cursor;
	};
	constexpr uint64_t      Gap = 64 * 1024;
	std::vector<Envelope>   envelopes;
	std::vector<GuestReadback::Part> parts;
	parts.reserve(copies.size());
	uint64_t packed_size = 0;
	for (const auto& copy: copies) {
		const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
		if (!envelopes.empty() && source_begin <= envelopes.back().source + envelopes.back().size + Gap) {
			auto&      last = envelopes.back();
			const auto end  = std::max(last.source + last.size, source_begin + envelope_size);
			packed_size += AlignDownload(end - last.source) - AlignDownload(last.size);
			last.size = end - last.source;
		} else {
			envelopes.push_back({source_begin, envelope_size, packed_size});
			packed_size += AlignDownload(envelope_size);
		}
		parts.push_back({copy.address, copy.size, envelopes.back().cursor + copy.source_offset - envelopes.back().source});
		if (packed_size > Capacity) {
			LiveCounters::Add(LiveCounters::RbRejectCapacity);
			return {};
		}
	}
	LiveCounters::Add(LiveCounters::ReadbackRegions, envelopes.size());
	// KYTY_READBACK_SLOTS: when every slot is pending, the render thread waits for the
	// oldest copy (a GPU tick). Bursts from many guest threads fill the original eight.
	const size_t slots = kyty_local_readback_slots_mode.load(std::memory_order_relaxed) != 0
	                         ? GuestReadbackSlots
	                         : 8;
	size_t slot = 0;
	while (slot < slots && m_guest_readbacks[slot]) ++slot;
	if (slot == slots) {
		slot = 0;
		for (size_t i = 1; i < slots; ++i)
			if (m_guest_readbacks[i]->tick < m_guest_readbacks[slot]->tick) slot = i;
		LiveCounters::Add(LiveCounters::ReadbackEvictions);
		FinishGuestReadback(slot);
	}
	auto& download = m_guest_downloads[slot];
	if (!download)
		download = std::make_unique<Buffer>(m_graphics, m_scheduler,
		    MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, Capacity);
	// A detached copy-engine request may still write this slot.
	if (m_download_queue_values[slot] != 0) {
		m_readback_queue->Wait(m_download_queue_values[slot]);
		m_download_queue_values[slot] = 0;
	}
	auto request = std::make_shared<GuestReadback>();
	request->begin = begin;
	request->size = end - begin;
	request->pages = std::move(pages);
	request->slot = slot;
	request->download = download.get();
	request->packed_size = packed_size;
	// KYTY_READBACK_QUEUE: bytes whose writers the GPU finished (or, mode 2, has been handed)
	// need not wait behind the rest of the graphics queue.
	if (kyty_local_readback_page_runs_mode.load(std::memory_order_relaxed) == 0) runs.assign(1, {begin, end});
	if (ReadbackQueueReady(runs, true, request->producer_tick)) {
		// Also after the slot's last graphics-queue copy (it is submitted: that path flushes).
		request->producer_tick = std::max(request->producer_tick, m_download_ticks[slot]);
		std::vector<ReadbackQueue::Queue::Region> regions;
		regions.reserve(envelopes.size());
		for (const auto& envelope: envelopes)
			regions.push_back({buffer.Handle(), {envelope.source, envelope.cursor, envelope.size}});
		request->parts = std::move(parts);
		request->queue_value = m_readback_queue->Copy(regions, download->Handle(),
		                                              m_scheduler.GetMasterSemaphore().Handle(),
		                                              request->producer_tick);
		LiveTrace::Event(LiveTrace::RbFast, begin, (end - begin) | request->producer_tick << 32u);
		m_download_queue_values[slot] = request->queue_value;
		request->tick                 = m_scheduler.CurrentTick();
		if (kyty_local_readback_queue_mode.load(std::memory_order_relaxed) == 3) {
			auto& verify = m_verify_downloads[slot];
			if (!verify)
				verify = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
				                                  vk::BufferUsageFlagBits::eTransferDst, Capacity);
			for (const auto& region: regions)
				verify->CopyFrom(m_scheduler.Current(), buffer, region.copy.srcOffset, region.copy.dstOffset,
				                 region.copy.size, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
				                 vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
				                 vk::AccessFlagBits::eHostRead);
			request->verify = verify.get();
			m_scheduler.Flush();
		}
		if (LiveTrace::WriteTicks())
			LiveTrace::Event(LiveTrace::ReadbackTicks, address, size | request->tick << 32u);
		m_guest_readbacks[slot] = request;
		m_active_guest_readbacks |= 1u << slot;
		LiveCounters::Add(LiveCounters::AsyncReadbacks);
		return request;
	}
	for (const auto& envelope: envelopes)
		download->CopyFrom(m_scheduler.Current(), buffer, envelope.source, envelope.cursor, envelope.size,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
	request->parts = std::move(parts);
	request->tick = m_scheduler.CurrentTick();
	m_download_ticks[slot] = request->tick;
	if (LiveTrace::WriteTicks()) LiveTrace::Event(LiveTrace::ReadbackTicks, address, size | request->tick << 32u);
	m_scheduler.Flush();
	// Keep GPU ownership and page protection until the caller has copied every byte.
	// Subsequent accesses to this window must retire this request before proceeding.
	m_guest_readbacks[slot] = request;
	m_active_guest_readbacks |= 1u << slot;
	LiveCounters::Add(LiveCounters::AsyncReadbacks);
	return request;
}

// Local diagnostic: every GPU write into guest buffers (tracew), noted or not.
static void TraceGpuWrite(uint64_t vaddr, uint64_t size, uint64_t tick, uint32_t origin) {
	if (LiveTrace::WriteTicks()) {
		LiveTrace::Event(LiveTrace::GpuWrite, vaddr, size | tick << 32u);
		LiveTrace::Event(LiveTrace::WriteContext, LiveTrace::g_mark_tag,
		                 origin | uint64_t {LiveTrace::g_write_kind} << 8u);
	}
}

// KYTY_READBACK_QUEUE. GPU thread, inside a command: the write is recorded at or after the
// current tick, so its tick is only set at the next point between commands. Every GPU write
// into a guest buffer is noted: shader writes, uploads, image copies and buffer joins.
void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size, GpuWriteOrigin origin) {
	TraceGpuWrite(vaddr, size, m_scheduler.CurrentTick(), static_cast<uint32_t>(origin));
	if (kyty_local_readback_queue_mode.load(std::memory_order_relaxed) == 0 || m_graphics.readback_queue == nullptr) {
		if (m_gpu_writes_on) {
			m_gpu_writes_on = false;
			m_gpu_writes.clear();
			m_gpu_writes_head = m_gpu_writes_stamped = 0;
			m_gpu_write_index.reset();
			m_gpu_write_sequence = 0;
			m_gpu_write_index_on = false;
			m_gpu_write_index_safe = true;
			m_gpu_write_large_count = 0;
		}
		return;
	}
	// Starting, or no point between commands for very long: forget what is not stamped and
	// refuse the fast path until everything recorded so far has completed.
	constexpr size_t Limit = size_t {1} << 20;
	if (!m_gpu_writes_on || m_gpu_writes.size() - m_gpu_writes_head >= Limit) {
		m_gpu_writes_on = true;
		m_gpu_writes.clear();
		m_gpu_writes_head = m_gpu_writes_stamped = 0;
		m_gpu_writes_from = UINT64_MAX;
		m_gpu_write_sequence = 0;
		m_gpu_write_index_on = GpuWriteLogIndexMode() != 0;
		m_gpu_write_index_safe = true;
		m_gpu_write_large_count = 0;
		if (m_gpu_write_index_on) {
			m_gpu_write_index = std::make_unique<GpuWriteIndex>();
			m_gpu_write_index->fill(0);
		} else {
			m_gpu_write_index.reset();
		}
	}
	// Writes not stamped yet all get the same tick at the next point between commands: a range
	// noted again before then (a dispatch run writing the same buffers) adds nothing.
	for (size_t i = m_gpu_writes.size(), last = std::max(m_gpu_writes_stamped, i > 4 ? i - 4 : 0); i > last; --i)
		if (m_gpu_writes[i - 1].begin == vaddr && m_gpu_writes[i - 1].end == vaddr + size) return;
	if (m_gpu_write_index_on && m_gpu_write_index_safe) {
		if (m_gpu_write_sequence == UINT64_MAX) {
			// Never wrap sequence arithmetic. The retained log remains authoritative.
			m_gpu_write_index_safe = false;
			m_gpu_write_index.reset();
		} else {
			++m_gpu_write_sequence;
		}
	}
	const uint64_t end = vaddr + size;
	m_gpu_writes.push_back({vaddr, end, 0, m_gpu_write_sequence});
	if (m_gpu_write_index_on && m_gpu_write_index_safe) {
		if (IsLargeGpuWrite(vaddr, end)) {
			++m_gpu_write_large_count;
		} else if (vaddr < end) {
			for (uint64_t granule = vaddr >> 16u; granule <= (end - 1) >> 16u; ++granule)
				(*m_gpu_write_index)[GpuWriteIndexSlot(granule << 16u)] = m_gpu_write_sequence;
		}
	}
}


// KYTY_READBACK_QUEUE. GPU thread: the latest tick of a possibly unfinished GPU write
// overlapping [begin, end); 0 when there is none. Between commands (guest readbacks run there)
// every noted write is recorded by now and gets its tick; inside a command an undated write
// may still be recorded later, so it yields UINT64_MAX.
uint64_t BufferCache::InflightWriteTick(uint64_t begin, uint64_t end, uint64_t completed,
                                        bool between_commands) {
	if (between_commands) {
		const uint64_t current = m_scheduler.CurrentTick();
		for (; m_gpu_writes_stamped < m_gpu_writes.size(); ++m_gpu_writes_stamped)
			m_gpu_writes[m_gpu_writes_stamped].tick = current;
		if (m_gpu_writes_from == UINT64_MAX) m_gpu_writes_from = current;
	}
	// Stamps never decrease: the completed writes are a prefix of the dated ones.
	while (m_gpu_writes_head < m_gpu_writes_stamped && m_gpu_writes[m_gpu_writes_head].tick <= completed) {
		const auto& write = m_gpu_writes[m_gpu_writes_head];
		if (m_gpu_write_index_on && m_gpu_write_index_safe && IsLargeGpuWrite(write.begin, write.end))
			--m_gpu_write_large_count;
		++m_gpu_writes_head;
	}
	if (m_gpu_writes_head >= 4096 && m_gpu_writes_head * 2 >= m_gpu_writes.size()) {
		m_gpu_writes.erase(m_gpu_writes.begin(), m_gpu_writes.begin() + static_cast<std::ptrdiff_t>(m_gpu_writes_head));
		m_gpu_writes_stamped -= m_gpu_writes_head;
		m_gpu_writes_head = 0;
		// Absolute sequence stamps remain valid across prefix compaction.
	}
	auto full_scan = [&] {
		uint64_t result = 0;
		for (size_t i = m_gpu_writes_head; i < m_gpu_writes.size(); ++i) {
			const auto& write = m_gpu_writes[i];
			if (write.begin < end && begin < write.end)
				result = std::max(result, i < m_gpu_writes_stamped ? write.tick : UINT64_MAX);
		}
		return result;
	};
	auto verify_indexed = [&](uint64_t indexed, uint64_t sequence) {
		if (GpuWriteLogIndexMode() != 2) return indexed;
		const uint64_t reference = full_scan();
		if (indexed != reference) {
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1, std::memory_order_relaxed) < 40) {
				std::printf("GPU_WRITE_LOG_INDEX_MISMATCH begin=0x%" PRIx64 " end=0x%" PRIx64
				            " indexed=%" PRIu64 " reference=%" PRIu64 " sequence=%" PRIu64 "\n",
				            begin, end, indexed, reference, sequence);
				std::fflush(stdout);
			}
			m_gpu_write_index_safe = false;
			m_gpu_write_index.reset();
			return reference;
		}
		static uint64_t verified = 0; // GPU thread owns the write log.
		if (++verified == 1 || (verified & 65535u) == 0) {
			std::printf("GPU_WRITE_LOG_INDEX verified=%" PRIu64 " mismatches=0\n", verified);
			std::fflush(stdout);
		}
		return indexed;
	};
	if (!m_gpu_write_index_on || !m_gpu_write_index_safe || m_gpu_write_large_count != 0 || begin >= end)
		return full_scan();
	constexpr uint64_t MaxQueryGranules = 256;
	constexpr size_t MaxBackwardCandidates = 4096;
	const uint64_t first = begin >> 16u;
	const uint64_t last  = (end - 1) >> 16u;
	if (last - first + 1 > MaxQueryGranules) return full_scan();
	uint64_t candidate_sequence = 0;
	for (uint64_t granule = first; granule <= last; ++granule)
		candidate_sequence = std::max(candidate_sequence, (*m_gpu_write_index)[GpuWriteIndexSlot(granule << 16u)]);
	if (candidate_sequence == 0 || m_gpu_writes.empty() || candidate_sequence < m_gpu_writes.front().sequence)
		return verify_indexed(0, candidate_sequence);
	const uint64_t offset = candidate_sequence - m_gpu_writes.front().sequence;
	if (offset >= m_gpu_writes.size()) return full_scan();
	size_t i = static_cast<size_t>(offset) + 1;
	size_t walked = 0;
	while (i > m_gpu_writes_head && walked < MaxBackwardCandidates) {
		const auto& write = m_gpu_writes[--i];
		++walked;
		if (write.begin < end && begin < write.end) {
			const uint64_t indexed = i < m_gpu_writes_stamped ? write.tick : UINT64_MAX;
			return verify_indexed(indexed, candidate_sequence);
		}
	}
	if (i > m_gpu_writes_head) return full_scan();
	return verify_indexed(0, candidate_sequence);
}

bool BufferCache::ReadbackQueueReady(std::span<const std::pair<uint64_t, uint64_t>> ranges,
                                     bool between_commands, uint64_t& wait_tick) {
	const auto mode = kyty_local_readback_queue_mode.load(std::memory_order_relaxed);
	if (mode == 0 || m_graphics.readback_queue == nullptr || !m_gpu_writes_on) return false;
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const uint64_t completed = master.KnownGpuTick();
	uint64_t       producer  = 0;
	for (const auto& [begin, end]: ranges)
		producer = std::max(producer, InflightWriteTick(begin, end, completed, between_commands));
	// Writes before the log started are dated only once everything recorded then completed.
	if (m_gpu_writes_from == UINT64_MAX || completed < m_gpu_writes_from) return false;
	// Mode 2 waits for an unfinished last writer on the transfer queue; mode 3 does the same and
	// compares every copy with a graphics-queue copy (the write log must know every writer).
	if (producer != 0 && (mode < 2 || producer >= m_scheduler.CurrentTick())) return false;
	if (!m_readback_queue) m_readback_queue = std::make_unique<ReadbackQueue::Queue>(m_graphics);
	LiveCounters::Add(producer == 0 ? LiveCounters::RbQueueDone : LiveCounters::RbQueueInflight);
	wait_tick = std::max(producer, completed);
	return true;
}

// KYTY_READBACK_QUEUE=3. Any thread.
void BufferCache::VerifyReadback(uint64_t address, const uint8_t* fast, const uint8_t* reference, uint64_t size,
                                 const char* path, uint64_t waited) {
	LiveCounters::Add(LiveCounters::RbQueueVerified);
	if (std::memcmp(fast, reference, size) == 0) return;
	LiveCounters::Add(LiveCounters::RbQueueMismatch);
	LiveTrace::Event(LiveTrace::RbMismatch, address, size | waited << 32u);
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 40) {
		uint64_t first = 0, differing = 0;
		while (fast[first] == reference[first]) ++first;
		for (uint64_t i = 0; i < size; ++i) differing += fast[i] != reference[i];
		// Every logged write overlapping the range, oldest first (the log is GPU-thread state;
		// a guest thread's report may race, it is diagnostic only).
		std::string writes;
		for (size_t i = m_gpu_writes_head; i < m_gpu_writes.size() && writes.size() < 300; ++i) {
			const auto& write = m_gpu_writes[i];
			if (write.begin < address + size && address < write.end)
				writes += fmt::format(" [{:#x}+{:#x} t{}{}]", write.begin, write.end - write.begin, write.tick,
				                      i < m_gpu_writes_stamped ? "" : "?");
		}
		std::string bytes;
		for (uint64_t i = first & ~uint64_t {3}; i < std::min(size, (first & ~uint64_t {3}) + 16); ++i)
			bytes += fmt::format("{:02x}/{:02x} ", fast[i], reference[i]);
		std::printf("READBACK_QUEUE_MISMATCH %s address=0x%" PRIx64 " size=0x%" PRIx64 " first=0x%" PRIx64
		            " differing=%" PRIu64 " waited=%" PRIu64 " current=%" PRIu64 " known=%" PRIu64
		            " bytes(fast/ref)=%s writes:%s\n",
		            path, address, size, first, differing, waited, m_scheduler.CurrentTick(),
		            m_scheduler.GetMasterSemaphore().KnownGpuTick(), bytes.c_str(), writes.c_str());
		std::fflush(stdout);
	}
}

void BufferCache::CopyGuestReadback(const std::shared_ptr<GuestReadback>& request) {
	if (request->copying.exchange(true, std::memory_order_acq_rel)) {
		while (!request->copied.load(std::memory_order_acquire)) request->copied.wait(false);
		return;
	}
	if (request->queue_value != 0) {
		m_readback_queue->Wait(request->queue_value);
		m_scheduler.WaitPriorityOperations(request->producer_tick);
	} else {
		m_scheduler.GetMasterSemaphore().Wait(request->tick);
		m_scheduler.WaitPriorityOperations(request->tick);
	}
	if (request->verify != nullptr) {
		m_scheduler.GetMasterSemaphore().Wait(request->tick);
		m_scheduler.WaitPriorityOperations(request->tick);
	}
	uint32_t pending = GuestReadback::Pending;
	if (request->state.compare_exchange_strong(pending, GuestReadback::Copying, std::memory_order_acq_rel)) {
		request->download->Invalidate(0, request->packed_size);
		if (request->verify != nullptr) {
			request->verify->Invalidate(0, request->packed_size);
			for (const auto& part: request->parts)
				VerifyReadback(part.address, request->download->Mapped().data() + part.offset,
				               request->verify->Mapped().data() + part.offset, part.size, "guest",
				               request->producer_tick);
		}
		const auto* source = (request->verify != nullptr ? request->verify : request->download)->Mapped().data();
		for (const auto& part: request->parts) WriteBackGpuOwned(part.address, source + part.offset, part.size, "readback");
	}
	request->copied.store(true, std::memory_order_release);
	request->copied.notify_all();
}

void BufferCache::DrainGuestReadback(uint64_t address, uint64_t size, bool gpu_read_only, bool gpu_write) {
	// Read-only bindings use device bytes. Pending copies retain GPU ownership
	// and CPU-clean pages, so synchronization cannot upload over those bytes.
	if (!m_active_guest_readbacks || gpu_read_only) return;
	for (size_t slot = 0; slot < GuestReadbackSlots; ++slot) {
		const auto& request = m_guest_readbacks[slot];
		if (!request) continue;
		if (address != 0 && (address >= request->begin + request->size ||
		                    (address < request->begin && size <= request->begin - address))) continue;
		if (address != 0 && std::ranges::none_of(request->pages, [&](const GuestRange& page) {
			return address < page.End() && (address >= page.address || size > page.address - address);
		})) continue;
		FinishGuestReadback(slot, gpu_write && kyty_local_readback_detach_mode.load(std::memory_order_relaxed) != 0);
	}
}

void BufferCache::FinishGuestReadback(size_t slot, bool detach) {
	auto request = m_guest_readbacks[slot];
	if (!request) return;
	uint32_t pending = GuestReadback::Pending;
	if (detach && !request->copied.load(std::memory_order_acquire) &&
	    request->state.compare_exchange_strong(pending, GuestReadback::Detached, std::memory_order_acq_rel)) {
		// The pages stay GPU-owned (ranges and page marks untouched): the next GPU
		// write keeps them so, and a reader faults again for the newer bytes.
		LiveCounters::Add(LiveCounters::ReadbackDetaches);
		m_guest_readbacks[slot].reset();
		m_active_guest_readbacks &= ~(1u << slot);
		return;
	}
	if (!request->copied.load(std::memory_order_acquire)) {
		LiveCensus::Scope census(LiveCensus::ReadbackWait, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
		                         reinterpret_cast<uint64_t>(__builtin_return_address(1)));
		LiveCensus::WaitScope waiting(LiveCensus::WaitReadback);
		while (!request->copied.load(std::memory_order_acquire)) request->copied.wait(false);
	}
	LiveCounters::Add(LiveCounters::ReadbackParts, request->parts.size());
	LiveCounters::Add(LiveCounters::GpuRangeEntries, m_gpu_modified_ranges.Count());
	for (const auto& part: request->parts) m_gpu_modified_ranges.Subtract(part.address, part.size);
	++m_gpu_coverage_epoch;
	// The renderer may have dirtied other pages in the widened window since handoff.
	for (const auto& page: request->pages)
		m_memory_tracker.UnmarkRegionAsGpuModified(page.address, page.size);
	m_guest_readbacks[slot].reset();
	m_active_guest_readbacks &= ~(1u << slot);
}

// All metadata and mapped reads belong to the GPU thread. A slot is reused only
// after its copy tick completes; speculative copies never publish CPU ownership.
struct BufferCache::CopyFeedback {
	static constexpr uint64_t SlotSize = 64 * 1024;
	static constexpr size_t SlotCount = 512;
	struct Slot {
		uint64_t address = 0, size = 0, tick = 0, mapping_epoch = 0;
		vk::Buffer owner = nullptr;
	};
	Buffer download;
	std::array<Slot, SlotCount> slots {};
	// Slot of each snapshot by its guest address, sorted by address (at most SlotCount entries:
	// a search of one array, where a map walked cold tree nodes on every GPU write).
	std::vector<std::pair<uint64_t, size_t>> index;
	size_t cursor = 0;
	[[nodiscard]] auto UpperBound(uint64_t address) {
		return std::upper_bound(index.begin(), index.end(), address,
		                        [](uint64_t value, const auto& entry) { return value < entry.first; });
	}
	[[nodiscard]] auto LowerBound(uint64_t address) {
		return std::lower_bound(index.begin(), index.end(), address,
		                        [](const auto& entry, uint64_t value) { return entry.first < value; });
	}
	bool Erase(uint64_t address) {
		const auto it = LowerBound(address);
		if (it == index.end() || it->first != address) return false;
		index.erase(it);
		return true;
	}
	bool Insert(uint64_t address, size_t slot) {
		const auto it = LowerBound(address);
		if (it != index.end() && it->first == address) return false;
		index.insert(it, {address, slot});
		return true;
	}
	// Index entries per 16 MiB granule, hashed into 4096 counters: a range whose granules count
	// none overlaps no entry, and the walk of the index (a cold tree) is skipped. Counters two
	// granules share only cost that walk.
	std::array<uint16_t, 4096> granules {};
	void Count(const Slot& slot, int delta) {
		for (uint64_t g = slot.address >> 24u; g <= (slot.address + slot.size - 1) >> 24u; ++g)
			granules[g & 4095u] = static_cast<uint16_t>(granules[g & 4095u] + delta);
	}
	[[nodiscard]] bool MayOverlap(uint64_t vaddr, uint64_t size) const {
		const uint64_t first = vaddr >> 24u, last = (vaddr + size - 1) >> 24u;
		if (size == 0 || last - first >= granules.size()) return true;
		for (uint64_t g = first; g <= last; ++g)
			if (granules[g & 4095u] != 0) return true;
		return false;
	}
	CopyFeedback(GraphicContext& graphics, CommandScheduler& scheduler)
	    : download(graphics, scheduler, MemoryUsage::Download, 0,
	               vk::BufferUsageFlagBits::eTransferDst, SlotCount * SlotSize) {
		// Each slot has a separate non-coherent atom, even when adjacent slots are in flight.
		const auto atom = graphics.physical_device_properties.limits.nonCoherentAtomSize;
		EXIT_IF(atom == 0 || SlotSize % atom != 0);
	}
};

void BufferCache::InvalidateCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) return;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size)) return;
	auto it = feedback.LowerBound(vaddr);
	if (it != feedback.index.begin()) {
		const auto prior = std::prev(it);
		const auto& slot = feedback.slots[prior->second];
		if (slot.address + slot.size > vaddr) it = prior;
	}
	auto last = it;
	for (; last != feedback.index.end() && last->first < vaddr + size; ++last) {
		feedback.Count(feedback.slots[last->second], -1);
		feedback.slots[last->second].address = 0;
	}
	feedback.index.erase(it, last);
}

void BufferCache::ScheduleCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (kyty_local_copy_feedback_mode.load(std::memory_order_relaxed) == 0) return;
	const auto mapping_epoch = m_resources ? m_resources->MappingEpoch() : 0;
	if (!m_resources || !GuestRange {vaddr, size}.Valid() ||
	    ((vaddr | size) & 3u) != 0 || size > CopyFeedback::SlotSize ||
	    !m_gpu_modified_ranges.Contains(vaddr, size) ||
	    m_texture_cache.HasTrackedDataOverlap(vaddr, size) ||
	    !m_resources->IsMapped(vaddr, size) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(vaddr, size)) {
		return;
	}
	const auto* owner_id = m_page_table.Find(vaddr >> PageTable::kPageBits);
	auto* owner = owner_id && *owner_id ? m_slot_buffers.try_get(*owner_id) : nullptr;
	if (!owner || owner->is_deleted || !owner->IsInBounds(vaddr, size)) {
		return;
	}
	if (!m_copy_feedback) m_copy_feedback = std::make_unique<CopyFeedback>(m_graphics, m_scheduler);
	auto& feedback = *m_copy_feedback;
	auto& master = m_scheduler.GetMasterSemaphore();
	size_t selected = CopyFeedback::SlotCount;
	for (unsigned refresh = 0; refresh < 2 && selected == CopyFeedback::SlotCount; ++refresh) {
		if (refresh) master.Refresh();
		const auto completed = master.KnownGpuTick();
		// Prefer an invalidated slot, then evict an old completed snapshot if necessary.
		for (unsigned evict = 0; evict < 2 && selected == CopyFeedback::SlotCount; ++evict) {
			for (size_t n = 0; n < CopyFeedback::SlotCount; ++n) {
				const auto i = (feedback.cursor + n) % CopyFeedback::SlotCount;
				const auto& slot = feedback.slots[i];
				if (slot.tick <= completed && (evict || slot.address == 0)) {
					selected = i;
					break;
				}
			}
		}
	}
	if (selected == CopyFeedback::SlotCount) {
		return; // Never submit or wait merely to obtain speculative storage.
	}
	if (m_resources->MappingEpoch() != mapping_epoch) return;
	auto& slot = feedback.slots[selected];
	if (slot.address && feedback.Erase(slot.address)) {
		feedback.Count(slot, -1);
	}
	InvalidateCopyFeedback(vaddr, size);
	feedback.download.CopyFrom(m_scheduler.Current(), *owner, owner->Offset(vaddr),
	    selected * CopyFeedback::SlotSize, size, vk::AccessFlagBits::eMemoryWrite,
	    vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostRead,
	    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	    vk::AccessFlagBits::eHostRead);
	slot = {vaddr, size, m_scheduler.CurrentTick(), mapping_epoch, owner->Handle()};
	if (feedback.Insert(vaddr, selected)) feedback.Count(slot, 1);
	feedback.cursor = (selected + 1) % CopyFeedback::SlotCount;
}

bool BufferCache::TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (kyty_local_copy_feedback_mode.load(std::memory_order_relaxed) == 0 || !m_copy_feedback || !m_resources)
		return false;
	const auto begin = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto end = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	if (!buffer.IsInBounds(begin, end - begin) || end - begin > CopyFeedback::SlotSize ||
	    m_texture_cache.HasTrackedDataOverlap(begin, end - begin)) {
		return false;
	}
	std::vector<DownloadCopy> copies;
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "copy feedback");
	    },
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachIntersection(address, bytes, [&](RangeSet::Range range) {
			    copies.push_back({&buffer, buffer.Offset(range.address), range.address, range.size});
		    });
	    });
	if (copies.empty()) return false;
	struct Part { uint64_t address, size, offset; };
	std::vector<Part> parts;
	auto& feedback = *m_copy_feedback;
	const auto mapping_epoch = m_resources->MappingEpoch();
	uint64_t latest_tick = 0;
	for (const auto& copy: copies) {
		auto cursor = copy.address;
		while (cursor < copy.address + copy.size) {
			auto it = feedback.UpperBound(cursor);
			if (it == feedback.index.begin()) return false;
			--it;
			const auto& slot = feedback.slots[it->second];
			if (cursor >= slot.address + slot.size) return false;
			if (slot.owner != buffer.Handle() || slot.mapping_epoch != mapping_epoch ||
			    !m_resources->IsMapped(slot.address, slot.size) ||
			    !LibKernel::Memory::IsUniqueGuestBackingRange(slot.address, slot.size)) {
				return false;
			}
			const auto bytes = std::min(copy.address + copy.size, slot.address + slot.size) - cursor;
			parts.push_back({cursor, bytes, it->second * CopyFeedback::SlotSize + cursor - slot.address});
			latest_tick = std::max(latest_tick, slot.tick);
			cursor += bytes;
		}
	}
	if (!m_scheduler.IsFree(latest_tick)) return false;
	m_scheduler.WaitPriorityOperations(latest_tick);
	if (m_resources->MappingEpoch() != mapping_epoch) return false;
	for (const auto& part: parts) feedback.download.Invalidate(part.offset, part.size);
	for (const auto& part: parts) {
		WriteBackGpuOwned(part.address, feedback.download.Mapped().data() + part.offset, part.size, "feedback");
	}
	for (const auto& copy: copies) m_gpu_modified_ranges.Subtract(copy.address, copy.size);
	++m_gpu_coverage_epoch;
	// Only complete dirty-page coverage allows dropping protection. CPU-clean bytes
	// are never published from the snapshot. Any later upload/GPU write invalidates it.
	m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
	return true;
}

// GPU thread: a guest access to bytes a copy-feedback snapshot holds (copied after their last GPU
// write, the copy complete) is served from it at once, instead of a transfer and a second command
// for the reader; the whole snapshot then, whose other pages would fault one by one. Not while a
// pending readback covers the pages: its copy would land later.
bool BufferCache::TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) return false;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size) || !IsRegionRegistered(vaddr, size)) return false;
	auto first = vaddr, last = vaddr + size;
	if (auto it = feedback.UpperBound(vaddr); it != feedback.index.begin()) {
		const auto& slot = feedback.slots[std::prev(it)->second];
		if (slot.address <= vaddr && vaddr < slot.address + slot.size) {
			first = std::min(first, slot.address);
			last  = std::max(last, slot.address + slot.size);
		}
	}
	const auto begin = first & ~(TRACKER_PAGE_SIZE - 1);
	const auto end   = (last + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (const auto& pending: m_guest_readbacks)
		if (pending && pending->begin < end && begin < pending->begin + pending->size) return false;
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (last - first == size || !buffer.IsInBounds(first, last - first) || !TryReadCopyFeedback(buffer, first, last - first)) {
		if (!TryReadCopyFeedback(buffer, vaddr, size)) return false;
	}
	if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	return true;
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	DrainGuestReadback(m_slot_buffers[id].CpuAddress(), m_slot_buffers[id].Size());
	m_sync_buffers_valid = false;
	++m_registration_epoch;
	// Region requests prove their state against the registration epoch of their own region: only the
	// requests of these regions need to be synchronized again.
	if (m_region_registrations.empty()) m_region_registrations.assign(TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE, 0);
	for (auto region = m_slot_buffers[id].CpuAddress() / TRACKER_REGION_SIZE,
	          last   = (m_slot_buffers[id].CpuAddress() + m_slot_buffers[id].Size() - 1) / TRACKER_REGION_SIZE;
	     region <= last && region < m_region_registrations.size(); ++region) {
		m_region_registrations[region] = m_registration_epoch;
		BdaDirtyRegions::Mark(region);
	}
	LiveCounters::Add(LiveCounters::BufferRegistrations);
	auto& buffer = m_slot_buffers[id];
	if constexpr (!insert) InvalidateCopyFeedback(buffer.CpuAddress(), buffer.Size());
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

bool BufferCache::IsGpuMapped(uint64_t vaddr, uint64_t size) const noexcept {
	return m_resources == nullptr || m_resources->IsMapped(vaddr, size);
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted) {
		return;
	}
	Unregister(id);
	// KYTY_READBACK_QUEUE: a copy on the transfer queue may still read it after the graphics timeline
	// passes (it waits on that timeline only for the bytes' last writer). Freed then, its memory was
	// in use again on GPUs short of VRAM (frequent GC): VK_ERROR_DEVICE_LOST.
	const uint64_t readback = m_readback_queue ? m_readback_queue->Submitted() : 0;
	const auto     erase    = [this, id, readback] {
		if (readback != 0 && m_readback_queue) m_readback_queue->Wait(readback);
		m_slot_buffers.erase(id);
	};
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation(erase);
	} else {
		erase();
	}
}

std::pair<uint64_t, uint64_t> BufferCache::DownloadEnvelope(const DownloadCopy& copy) {
	if (copy.buffer == nullptr || copy.size == 0 || copy.source_offset > copy.buffer->Size() ||
	    copy.size > copy.buffer->Size() - copy.source_offset) {
		EXIT("BufferCache: invalid download copy\n");
	}
	const auto begin = copy.source_offset & ~uint64_t {3};
	if (copy.source_offset > UINT64_MAX - copy.size ||
	    copy.source_offset + copy.size > UINT64_MAX - 3) {
		EXIT("BufferCache: download copy alignment overflow\n");
	}
	const auto end = (copy.source_offset + copy.size + 3) & ~uint64_t {3};
	if (end > copy.buffer->Size()) {
		EXIT("BufferCache: aligned download copy exceeds its owner\n");
	}
	return {begin, end - begin};
}

// GPU data into guest memory, only on pages the GPU still owns. Every write-back path gives pages back to
// the CPU only after writing them, so a CPU-owned page here holds what the CPU wrote since the copy was
// taken (or nothing the GPU ever wrote): the stale GPU bytes would overwrite newer game data, which shows
// up much later as a broken game structure (a null pointer in a job worker while streaming). Logged.
void BufferCache::WriteBackGpuOwned(uint64_t address, const uint8_t* data, uint64_t size, const char* source) {
	if (m_memory_tracker.IsRegionFullyGpuModified(address, size)) {
		LibKernel::Memory::WriteBacking(address, data, size, source);
		return;
	}
	uint64_t skipped = 0;
	for (uint64_t at = address, end = address + size; at < end;) {
		const auto next = std::min((at & ~(TRACKER_PAGE_SIZE - 1)) + TRACKER_PAGE_SIZE, end);
		if (m_memory_tracker.IsRegionFullyGpuModified(at, next - at)) {
			LibKernel::Memory::WriteBacking(at, data + (at - address), next - at, source);
		} else {
			skipped += next - at;
		}
		at = next;
	}
	LiveCounters::Add(LiveCounters::WriteBackSkips, skipped);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("BufferCache: %s write-back skipped 0x%" PRIx64 " of 0x%" PRIx64 " bytes at 0x%016" PRIx64
		            ": CPU-owned pages\n",
		            source, skipped, size, address);
		std::fflush(stdout);
	}
}

void BufferCache::DownloadBufferMemory(std::span<const DownloadCopy> copies) {
	NativeBufferDownloadScope native_download_scope;
	std::vector<DownloadCopy> batch;
	batch.reserve(copies.size());
	uint64_t                  packed_size = 0;
	auto&                     download    = m_download_buffer;
	const auto flush = [&] {
		const auto [mapped, base_offset] = download.Map(packed_size, DOWNLOAD_ALIGNMENT);
		EXIT_IF(mapped == nullptr);
		uint64_t cursor = 0;
		// KYTY_READBACK_QUEUE: bytes no unfinished GPU write touches copy on the transfer queue
		// instead of draining the graphics queue. This may run inside a command.
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		ranges.reserve(batch.size());
		for (const auto& copy: batch) ranges.emplace_back(copy.address, copy.address + copy.size);
		if (uint64_t wait_tick = 0; ReadbackQueueReady(ranges, false, wait_tick)) {
			std::vector<ReadbackQueue::Queue::Region> regions;
			regions.reserve(batch.size());
			for (const auto& copy: batch) {
				const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
				regions.push_back({copy.buffer->Handle(), {source_begin, base_offset + cursor, envelope_size}});
				cursor += AlignDownload(envelope_size);
			}
			download.Commit();
			if (LiveTrace::WriteTicks())
				for (const auto& copy: batch)
					LiveTrace::Event(LiveTrace::SyncReadback, copy.address, copy.size | m_scheduler.CurrentTick() << 32u);
			m_readback_queue->Wait(m_readback_queue->Copy(regions, download.Handle(),
			                                              m_scheduler.GetMasterSemaphore().Handle(), wait_tick));
			m_scheduler.WaitPriorityOperations(wait_tick);
			if (kyty_local_readback_queue_mode.load(std::memory_order_relaxed) == 3) {
				// The reference copy lands in a second reservation; the backing gets its bytes.
				const auto [reference, reference_offset] = download.Map(packed_size, DOWNLOAD_ALIGNMENT);
				EXIT_IF(reference == nullptr);
				for (size_t i = 0; i < batch.size(); ++i)
					download.CopyFrom(m_scheduler.Current(), *batch[i].buffer, regions[i].copy.srcOffset,
					                  reference_offset + regions[i].copy.dstOffset - base_offset, regions[i].copy.size,
					                  vk::AccessFlagBits::eMemoryWrite, vk::AccessFlags {},
					                  vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
					                  vk::AccessFlagBits::eHostRead);
				download.Commit();
				const auto completion_tick = m_scheduler.CurrentTick();
				m_scheduler.Finish();
				m_scheduler.WaitPriorityOperations(completion_tick);
				uint64_t at = 0;
				for (const auto& copy: batch) {
					const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
					const auto offset = at + copy.source_offset - source_begin;
					download.Invalidate(base_offset + offset, copy.size);
					download.Invalidate(reference_offset + offset, copy.size);
					VerifyReadback(copy.address, mapped + offset, reference + offset, copy.size, "sync", wait_tick);
					std::memcpy(mapped + offset, reference + offset, copy.size);
					at += AlignDownload(envelope_size);
				}
			}
		} else {
			for (const auto& copy: batch) {
				const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
				download.CopyFrom(m_scheduler.Current(), *copy.buffer, source_begin, base_offset + cursor,
				                  envelope_size, vk::AccessFlagBits::eMemoryWrite,
				                  vk::AccessFlagBits::eTransferWrite,
				                  vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
				                  vk::AccessFlagBits::eHostRead);
				cursor += AlignDownload(envelope_size);
			}
			download.Commit();
			const auto completion_tick = m_scheduler.CurrentTick();
			if (LiveTrace::WriteTicks())
				for (const auto& copy: batch)
					LiveTrace::Event(LiveTrace::SyncReadback, copy.address, copy.size | completion_tick << 32u);
			m_scheduler.Finish();
			m_scheduler.WaitPriorityOperations(completion_tick);
		}
		cursor = 0;
		for (const auto& copy: batch) {
			const auto [source_begin, envelope_size] = DownloadEnvelope(copy);
			const auto offset = cursor + copy.source_offset - source_begin;
			download.Invalidate(base_offset + offset, copy.size);
			WriteBackGpuOwned(copy.address, mapped + offset, copy.size, "download");
			cursor += AlignDownload(envelope_size);
		}
		batch.clear();
		packed_size = 0;
	};
	for (auto copy: copies) {
		while (copy.size != 0) {
			const auto available = download.Size() - packed_size;
			const auto prefix    = copy.source_offset & 3u;
			const auto bytes     = std::min(copy.size, available - prefix);
			DownloadCopy part {copy.buffer, copy.source_offset, copy.address, bytes};
			const auto [source_begin, envelope_size] = DownloadEnvelope(part);
			(void)source_begin;
			packed_size += AlignDownload(envelope_size);
			batch.push_back(part);
			copy.source_offset += bytes;
			copy.address += bytes;
			copy.size -= bytes;
			if (packed_size == download.Size()) {
				flush();
			}
		}
	}
	if (!batch.empty()) {
		flush();
	}
	for (const auto& copy: copies) {
		m_gpu_modified_ranges.Subtract(copy.address, copy.size);
	}
	++m_gpu_coverage_epoch;
}

bool BufferCache::TryReportLodStatsOnGpu(uint64_t address, bool reset) {
    constexpr uint64_t ReportSize = 0x840;
    if (address % 4 || !m_resources || !m_resources->IsMapped(address, ReportSize) ||
        !LibKernel::Memory::IsUniqueGuestBackingRange(address, ReportSize) ||
        m_texture_cache.HasTrackedDataOverlap(address, ReportSize)) return false;
    if (!m_lod_pack_pipeline) {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings {};
        for (uint32_t i = 0; i < bindings.size(); ++i)
            bindings[i] = {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
        vk::DescriptorSetLayoutCreateInfo descriptors {};
        descriptors.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
        descriptors.bindingCount = bindings.size(); descriptors.pBindings = bindings.data();
        RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptors, nullptr, &m_lod_pack_descriptors),
                             "create LOD report descriptors");
        vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 8};
        vk::PipelineLayoutCreateInfo layout {};
        layout.setLayoutCount = 1; layout.pSetLayouts = &m_lod_pack_descriptors;
        layout.pushConstantRangeCount = 1; layout.pPushConstantRanges = &push;
        RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&layout, nullptr, &m_lod_pack_layout),
                             "create LOD report pipeline layout");
        vk::ShaderModuleCreateInfo shader {};
        shader.codeSize = sizeof(LOD_STATS_PACK_SPV); shader.pCode = LOD_STATS_PACK_SPV;
        vk::ShaderModule module;
        RequireVulkanSuccess(m_graphics.device.createShaderModule(&shader, nullptr, &module), "create LOD report shader");
        vk::ComputePipelineCreateInfo pipeline {};
        pipeline.stage.stage = vk::ShaderStageFlagBits::eCompute;
        pipeline.stage.module = module; pipeline.stage.pName = "main";
        pipeline.layout = m_lod_pack_layout;
        RequireVulkanSuccess(m_graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &m_lod_pack_pipeline),
                             "create LOD report pipeline");
        m_graphics.device.destroyShaderModule(module, nullptr);
    }
    // The output remains GPU-owned until the normal checked readback path
    // publishes all bytes. A CPU polling the ready word faults and waits for
    // this dispatch; never publish readiness on the CPU before GPU completion.
    const auto [output, offset] = ObtainBuffer(address, ReportSize, true);
    const auto alignment = m_graphics.GetPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    const auto base = offset & ~(alignment - 1);
    const uint32_t constants[] {uint32_t((offset - base) / 4), reset ? 1u : 0u};
    const vk::DescriptorBufferInfo buffers[] {{m_lod_stats_buffer.Handle(), 0, 256 * 16},
                                             {output->Handle(), base, offset - base + ReportSize}};
    std::array<vk::WriteDescriptorSet, 2> writes {};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = vk::DescriptorType::eStorageBuffer; writes[i].pBufferInfo = &buffers[i];
    }
    m_scheduler.EndRendering();
    const auto command = m_scheduler.Current().Handle();
    vk::MemoryBarrier barrier {};
    barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
                            {}, 1, &barrier, 0, nullptr, 0, nullptr);
    command.bindPipeline(vk::PipelineBindPoint::eCompute, m_lod_pack_pipeline);
    command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_lod_pack_layout, 0, writes);
    command.pushConstants(m_lod_pack_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), constants);
    command.dispatch(4, 1, 1);
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands,
                            {}, 1, &barrier, 0, nullptr, 0, nullptr);
    return true;
}

// GET_LOD_STATS of another layout than the packed 0x840 bytes: zeros with a ready label (1) in the first
// word. When the range holds GPU-written bytes the CPU write faulted and drained the GPU for its whole
// readback window first (Demon's Souls at the title: once a frame, ~10 MB/s of downloads); the GPU
// writes the same bytes in stream order instead.
bool BufferCache::TryFillLodStatsOnGpu(uint64_t vaddr, uint32_t size) {
	static const bool enabled = [] { // KYTY_LOD_FILL_GPU=0: the CPU write as before (A/B)
		const char* value = std::getenv("KYTY_LOD_FILL_GPU");
		return value == nullptr || value[0] != '0';
	}();
	if (!enabled) return false;
	// GPU-written pages, not bytes: the CPU write faults on a page another GPU write dirtied too, and that fault
	// drained the GPU (S2: 14% of the GPU thread's samples, with the stats' own bytes CPU-clean).
	if (size < 4 || (vaddr | size) % 4 != 0 || !GuestRange {vaddr, size}.Valid() ||
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) || m_texture_cache.HasTrackedDataOverlap(vaddr, size) ||
	    !m_resources || !m_resources->IsMapped(vaddr, size)) {
		return false;
	}
	m_scheduler.EndRendering();
	auto [buffer, offset] = ObtainBuffer(vaddr, size, true, false, BufferId {});
	if (buffer == nullptr) return false;
	const auto command = m_scheduler.Current().Handle();
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
	                        &barrier, 0, nullptr, 0, nullptr);
	command.fillBuffer(buffer->Handle(), offset, size, 0u);
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 1, &barrier,
	                        0, nullptr, 0, nullptr);
	const uint32_t ready = 1;
	command.updateBuffer(buffer->Handle(), offset, sizeof(ready), &ready);
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 1,
	                        &barrier, 0, nullptr, 0, nullptr);
	return true;
}

void BufferCache::ReportLodStats(void* dst, uint32_t size, bool reset) {
	// Pack the 64-byte completion header and 256 eight-byte LOD counters.
	// The command processor keeps other packet layouts on the existing path.
	EXIT_IF(dst == nullptr || size != 0x840);
	if (kyty_local_async_lod_stats_mode.load(std::memory_order_relaxed) &&
	    TryReportLodStatsOnGpu(reinterpret_cast<uint64_t>(dst), reset))
		return;
	auto& command = m_scheduler.Current();
	command.EndRendering();
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = m_lod_stats_buffer.Handle();
	barrier.size = 256 * 16;
	command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	    vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &barrier, 0, nullptr);
	m_scheduler.Finish();
	m_lod_stats_buffer.Invalidate(0, 256 * 16);
	auto* words = reinterpret_cast<uint32_t*>(m_lod_stats_buffer.Mapped().data());
	std::memset(dst, 0, size);
	const uint32_t ready = 1;
	std::memcpy(dst, &ready, sizeof(ready));
	for (uint32_t i = 0; i < 256; ++i) {
		const uint64_t entry = (uint64_t(words[i * 4] & 15u) << 56u) |
		    (uint64_t(std::min(words[i * 4 + 1], 0xffffffu)) << 32u) | words[i * 4 + 1];
		std::memcpy(static_cast<uint8_t*>(dst) + 64 + i * 8, &entry, sizeof(entry));
		if (reset) {
			words[i * 4] = 15;
			words[i * 4 + 1] = words[i * 4 + 2] = words[i * 4 + 3] = 0;
		}
	}
	if (reset) m_lod_stats_buffer.Flush(0, 256 * 16);
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache,
                         GpuResourceManager* resources)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_lod_stats_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, 256 * 16),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, UploadRingSize()),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_host_shader_upload(graphics, scheduler, MemoryUsage::Upload, 64 * MiB),
      m_table_upload(graphics, scheduler, MemoryUsage::Upload, 32 * MiB,
                     AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 32 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache), m_resources(resources) {
	m_gpu_modified_ranges.AllowFastPath();
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	std::memset(m_lod_stats_buffer.Mapped().data(), 0, 256 * 16);
	for (uint32_t i = 0; i < 256; ++i) {
		reinterpret_cast<uint32_t*>(m_lod_stats_buffer.Mapped().data())[i * 4] = 15;
	}
	m_lod_stats_buffer.Flush(0, 256 * 16);
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	DrainGuestReadback();
	m_readback_queue.reset();
    m_graphics.device.destroyPipeline(m_lod_pack_pipeline, nullptr);
    m_graphics.device.destroyPipelineLayout(m_lod_pack_layout, nullptr);
    m_graphics.device.destroyDescriptorSetLayout(m_lod_pack_descriptors, nullptr);
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// A write takes the same asynchronous copy (KYTY_ASYNC_WRITE_READBACK): the
	// GPU thread records the download instead of draining the GPU; the writer
	// waits and copies, then the written range becomes CPU-owned.
	const bool async_write = is_write && kyty_local_async_write_readback_mode.load(std::memory_order_relaxed) != 0;
	const bool guest       = !GuestGpu::IsGpuThread();
	if (guest) LiveTrace::Event(LiveTrace::GuestReadback, 1, vaddr);
	struct TraceEnd {
		bool     on;
		uint64_t vaddr;
		~TraceEnd() {
			if (on) LiveTrace::Event(LiveTrace::GuestReadback, 0, vaddr);
		}
	} trace_end {guest, vaddr};
	if ((!is_write || async_write) && !GuestGpu::IsGpuThread() && GuestReadbacksEnabled()) {
		std::shared_ptr<GuestReadback> request;
		auto& gpu = m_scheduler.Context().GetGpu();
		gpu.SendCommandSync([&] {
			if (TryGuestReadFromFeedback(vaddr, size, is_write)) return;
			bool completed = false;
			request = BeginGuestReadback(vaddr, size, &completed);
			if (!request && (is_write || !completed)) {
				LiveCounters::Add(LiveCounters::SyncReadsGuest);
				ReadMemoryOnGpu(vaddr, size, is_write);
			}
		});
		if (request) {
			CopyGuestReadback(request);
			gpu.SendCommandSync([this, request, vaddr, size, is_write] {
				// An overlapping access may already have retired this request.
				if (m_guest_readbacks[request->slot] == request) FinishGuestReadback(request->slot);
				if (is_write) FinishWriteReadback(vaddr, size);
			});
		}
		return;
	}
	LiveCounters::Add(GuestGpu::IsGpuThread() ? LiveCounters::SyncReadsRender : LiveCounters::SyncReadsGuest);
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		ReadMemoryOnGpu(vaddr, size, is_write);
	});
}

// GPU thread, after an asynchronous readback for a guest write: the tail of
// ReadMemoryOnGpu(is_write). A range the readback left GPU-owned (detached, or
// outside its window) is downloaded as before.
void BufferCache::FinishWriteReadback(uint64_t vaddr, uint64_t size) {
	if (!IsRegionRegistered(vaddr, size)) return;
	if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		ReadMemoryOnGpu(vaddr, size, true);
		return;
	}
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

void BufferCache::ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write) {
	LiveCensus::Scope census(LiveCensus::SyncDownload, vaddr >> 20u << 20u, is_write);
	LiveCensus::WaitScope waiting(LiveCensus::WaitDownload);
	if (LiveTrace::WriteTicks()) {
		LiveTrace::Event(LiveTrace::SyncRequest, vaddr, size | (uint64_t {is_write} << 63u));
		LiveTrace::Event(LiveTrace::SyncConsumer, Libs::LibKernel::Memory::g_srt_shader_hash,
		                 m_scheduler.CurrentTick());
	}
	const bool scoped_drain = kyty_local_sync_readback_scoped_drain_mode.load(std::memory_order_relaxed) != 0;
	// A synchronous SRT read need not retire tickets for unrelated guest pages.
	// Retire the requested pages before owner discovery, then the entire widened
	// window below before enumerating dirty ranges or publishing CPU ownership.
	// Owner replacement still drains that owner's tickets in DeleteBuffer.
	if (scoped_drain) DrainGuestReadback(vaddr, size);
	else DrainGuestReadback();
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	if (TryReadCopyFeedback(buffer, vaddr, size)) {
		if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		return;
	}

	// Widen nearby CPU reads so they share one GPU drain.
	constexpr uint64_t WindowSize   = 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const bool narrow = kyty_local_sync_readback_narrow_mode.load(std::memory_order_relaxed) != 0;
	const auto window_begin = std::max(vaddr & ~(narrow ? TRACKER_PAGE_SIZE - 1 : WindowSize - 1), buffer_begin);
	const auto window_end = narrow
	    ? std::min((vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1), buffer_end)
	    : std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	if (LiveTrace::WriteTicks()) LiveTrace::Event(LiveTrace::SyncWindow, window_begin, window_end - window_begin);
	if (scoped_drain) DrainGuestReadback(window_begin, window_end - window_begin);

	std::vector<DownloadCopy> copies;
	m_memory_tracker.ForEachDownloadRange<false>(
	    window_begin, window_end - window_begin,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "memory invalidation");
	    },
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachIntersection(address, bytes, [&](RangeSet::Range range) {
			    copies.push_back(
			        {&buffer, buffer.Offset(range.address), range.address, range.size});
		    });
	    });
	if (!copies.empty()) {
		LiveCounters::Add(LiveCounters::SyncDownloads);
		// KYTY_SYNC_LOG: every 10 s, the guest ranges whose CPU reads drained the GPU most often (and
		// the shader whose resource evaluation read them): what a frame stalls on.
		static const bool sync_log = std::getenv("KYTY_SYNC_LOG") != nullptr;
		if (sync_log) {
			struct Entry {
				uint64_t count = 0, bytes = 0, size = 0, shader = 0;
			};
			static std::unordered_map<uint64_t, Entry> entries;
			static auto                                last = std::chrono::steady_clock::now();
			auto&                                      entry = entries[vaddr];
			entry.count++;
			entry.size   = size;
			entry.shader = Libs::LibKernel::Memory::g_srt_shader_hash;
			for (const auto& copy: copies) entry.bytes += copy.size;
			if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(10)) {
				last = now;
				std::vector<std::pair<uint64_t, Entry>> top(entries.begin(), entries.end());
				std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
				std::printf("Sync downloads in 10 s:");
				for (size_t i = 0; i < top.size() && i < 10; i++) {
					std::printf(" [0x%llx size 0x%llx x%llu %llu KiB shader %016llx]",
					            static_cast<unsigned long long>(top[i].first),
					            static_cast<unsigned long long>(top[i].second.size),
					            static_cast<unsigned long long>(top[i].second.count),
					            static_cast<unsigned long long>(top[i].second.bytes >> 10u),
					            static_cast<unsigned long long>(top[i].second.shader));
				}
				std::printf("\n");
				std::fflush(stdout);
				entries.clear();
			}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			// The code paths that drained: return-address stacks (through the fault handler into the faulting
			// code), the most frequent printed every 10 s for offline symbolization (tools/sync_stacks.py).
			struct Stack {
				uint64_t count = 0, vaddr = 0;
				void*    frames[40] {};
				uint16_t depth = 0;
			};
			static std::unordered_map<uint64_t, Stack> stacks;
			static auto                                stacks_last = std::chrono::steady_clock::now();
			void*                                      frames[40] {};
			const auto depth = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
			uint64_t   hash  = 1469598103934665603ull;
			for (uint16_t i = 0; i < depth; ++i) hash = (hash ^ reinterpret_cast<uint64_t>(frames[i])) * 1099511628211ull;
			auto& stack = stacks[hash];
			if (stack.count++ == 0) {
				stack.vaddr = vaddr;
				stack.depth = depth;
				std::memcpy(stack.frames, frames, sizeof(frames));
			}
			if (const auto now = std::chrono::steady_clock::now(); now - stacks_last >= std::chrono::seconds(10)) {
				stacks_last = now;
				std::vector<const Stack*> top;
				for (const auto& [key, value]: stacks) top.push_back(&value);
				std::sort(top.begin(), top.end(), [](const Stack* a, const Stack* b) { return a->count > b->count; });
				for (size_t i = 0; i < top.size() && i < 8; ++i) {
					std::printf("SYNC_STACK count=%llu vaddr=0x%llx frames=", static_cast<unsigned long long>(top[i]->count),
					            static_cast<unsigned long long>(top[i]->vaddr));
					for (uint16_t f = 0; f < top[i]->depth; ++f)
						std::printf("%s%llx", f == 0 ? "" : ",",
						            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(top[i]->frames[f])));
					std::printf("\n");
				}
				std::fflush(stdout);
				stacks.clear();
			}
#endif
		}
		DownloadBufferMemory(copies);
		// The enumeration covered whole dirty pages and every exact interval on them.
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	// Registration changes still drain separately before retiring an owner.
	DrainGuestReadback(vaddr, size, true);
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Fix the shadPS4 bug that reserves space opposite to the incoming stream's growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end);
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.bda_read |= overlap.bda_read;
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	// The new owner holds these bytes only once the copy ran.
	NoteGpuWrite(overlap.CpuAddress(), overlap.Size(), GpuWriteOrigin::BufferJoin);
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = (vaddr + size + CACHING_PAGESIZE - 1) & ~(CACHING_PAGESIZE - 1);
	vaddr &= ~(CACHING_PAGESIZE - 1);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (!is_written && !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// CPU cleanliness does not prove that an aliased image is current.
		return is_texel_buffer && SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	bool fully_gpu_modified = false;
	if (is_written && kyty_local_buffer_residency_mode.load(std::memory_order_relaxed) != 0 &&
	    native_buffer_download_depth == 0 && m_gpu_modified_ranges.Contains(vaddr, size)) {
		// Exact ranges are added only after SynchronizeBuffer marks their
		// pages GPU-owned. Readback subtracts them before clearing page
		// ownership; CPU writes and unmapping take that readback path.
		// A complete byte cover is sufficient, while a hole or partial
		// cover still takes the original page query.
		fully_gpu_modified = true;
	}
	if (is_written && (fully_gpu_modified || m_memory_tracker.IsRegionFullyGpuModified(vaddr, size))) {
		// ObtainBuffer still records the exact write range and invalidates its epoch.
		return false;
	}
	// Only mapped guest memory is read. Pages the guest unmapped stay CPU-dirty (UnmapMemory
	// invalidates them) until memory is mapped there again, and an image's range can reach into
	// them: a texture of a streamed pool whose layers the game released while respawning. The
	// guest mappings tell (a PRT aperture is mapped for the GPU as a whole, its pages one by one);
	// not GpuResourceManager's ranges, whose lock PrepareBdaReadRanges holds around this.
	if (LibKernel::Memory::IsFullyMapped(vaddr, size)) {
		UploadDirtyRanges(buffer, vaddr, size, is_written);
	} else {
		std::vector<std::pair<uint64_t, uint64_t>> parts;
		LibKernel::Memory::MappedParts(vaddr, size, &parts);
		for (const auto& [address, bytes]: parts) UploadDirtyRanges(buffer, address, bytes, is_written);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

// The CPU-dirty pages of a mapped range copied into the buffer (the written ones then GPU-owned).
void BufferCache::UploadDirtyRanges(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written) {
	// Reused per thread: the memory tracker refuses a nested upload (CheckNotInUploadCallback).
	thread_local std::vector<vk::BufferCopy> copies;
	copies.clear();
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    // A run over consecutive mappings (the game maps its heaps a few hundred KiB at a time) goes out as a copy
		    // per mapping: each has a backing pointer of its own, so the upload worker reads it and takes its write
		    // protections (UploadCopies). One copy over a boundary had none, and the upload's bytes were read and its
		    // pages protected on this thread (Latria standing: ReapplyProtection 5.9% of the render thread).
		    for (uint64_t at = 0; at < bytes;) {
			    const uint64_t end   = LibKernel::Memory::MappingEnd(address + at);
			    const uint64_t piece = end > address + at ? std::min(bytes - at, end - (address + at)) : bytes - at;
			    copies.emplace_back(total_size, buffer.Offset(address + at), piece);
			    total_size += piece;
			    at += piece;
		    }
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		// The pieces of one run (split at mapping boundaries for the upload worker's reads, which UploadCopies pushed)
		// are adjacent in the staging ring and in the buffer: one copy region each run from here on. (The game's
		// mappings get finer at every level load: 434 -> 586 copies a frame at Stonefang over four rounds.)
		size_t merged = 0;
		for (const auto& copy: copies) {
			if (merged != 0 && copies[merged - 1].srcOffset + copies[merged - 1].size == copy.srcOffset &&
			    copies[merged - 1].dstOffset + copies[merged - 1].size == copy.dstOffset) {
				copies[merged - 1].size += copy.size;
			} else {
				copies[merged++] = copy;
			}
		}
		copies.resize(merged);
		for (const auto& copy: copies) {
			InvalidateCopyFeedback(buffer.CpuAddress() + copy.dstOffset, copy.size);
			NoteGpuWrite(buffer.CpuAddress() + copy.dstOffset, copy.size, GpuWriteOrigin::CpuUpload);
		}
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		// KYTY_UPLOAD_BARRIER=transfer (opt-in): the copy waits for earlier transfers only, not all GPU work.
		// Guest reads of these bytes recorded earlier in this command buffer were submitted before the guest
		// CPU wrote them (labels are written at GPU completion, so the guest cannot have waited for them):
		// on the console they may run after the write too. GPU-written bytes are never uploaded (a CPU write
		// to them reads them back first, in an earlier command buffer). What stays ordered is the
		// emulator's own transfers over the range (uploads, readback and forward copies, image uploads).
		// S8: ~500 uploads a frame, each a full drain (2,157 barriers, ~5.5 ms of GPU time a frame).
		static const bool transfer_only = [] {
			const char* value = std::getenv("KYTY_UPLOAD_BARRIER");
			return value != nullptr && std::string_view(value) == "transfer";
		}();
		if (transfer_only) {
			before.srcAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		}
		native.pipelineBarrier(transfer_only ? vk::PipelineStageFlagBits::eTransfer
		                                     : vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
}

// An upload's source bytes (a CPU-dirty range) into host memory, from inside the memory tracker's upload callback: through
// the backing view, whatever the guest range's protection, else through the guest view without faulting. A read of the
// guest view that faulted re-entered the tracker from the callback (another tracker's page protection: the fault
// handler read GPU data back) and ended the emulator (10-12, Tower of Latria, third visit). Bytes not readable at all (no
// mapping) are uploaded as zeros, reported.
static void ReadUploadSource(uint8_t* destination, uint64_t address, uint64_t size) {
	if (Libs::LibKernel::Memory::TryReadBackingToHost(address, destination, size)) return;
#if defined(_WIN32)
	if (const auto missed = Common::GuardedCopy(destination, reinterpret_cast<const void*>(address), size); missed != 0) {
		std::memset(destination + (size - missed), 0, missed);
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 8)
			std::printf("BufferCache: an upload's source 0x%llx+0x%llx is unreadable from +0x%llx: zeros uploaded\n",
			            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
			            static_cast<unsigned long long>(size - missed));
	}
#else
	std::memcpy(destination, reinterpret_cast<const void*>(address), size);
#endif
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	LiveCounters::Add(LiveCounters::UploadCopies, copies.size());
	LiveCounters::Add(LiveCounters::UploadBytes, total_size);
	if (LiveCounters::g_granules_on.load(std::memory_order_relaxed)) {
		for (const auto& copy: copies) {
			LiveCounters::AddGranule(buffer.CpuAddress() + copy.dstOffset, LiveCounters::GUploadBytes, copy.size);
			LiveCounters::AddGranule(buffer.CpuAddress() + copy.dstOffset, LiveCounters::GUploadCalls, 1);
		}
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	// KYTY_ASYNC_REPROTECT: the pages being copied were marked clean with their write
	// protection deferred. It precedes every copy: queued ahead of them when all go to the
	// worker, applied here otherwise.
	if (auto deferred = MemoryTracker::TakeDeferredProtects(); !deferred.empty()) {
		bool queued = mapped != nullptr && AsyncUpload::Enabled() && m_staging_buffer.IsCoherent();
		for (const auto& copy: copies)
			queued = queued && LibKernel::Memory::TryGetBackingPointer(buffer.CpuAddress() + copy.dstOffset, copy.size);
		for (const auto& item: deferred) {
			if (queued) {
				AsyncUpload::Get().PushCall(
				    [](void* manager, uint64_t address, uint64_t size) {
					    static_cast<RegionManager*>(manager)->ApplyDeferredProtection(address, size);
				    },
				    item.manager, item.address, item.size, item.address, item.size);
			} else {
				item.manager->ApplyDeferredProtection(item.address, item.size);
			}
		}
	}
	if (mapped != nullptr) {
		// KYTY_ASYNC_UPLOAD: the worker fills coherent staging memory from the backing view
		// before the submission that carries these copies; ranges without one copy here.
		const bool async = AsyncUpload::Enabled() && m_staging_buffer.IsCoherent();
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			const auto* source = async ? LibKernel::Memory::TryGetBackingPointer(address, copy.size) : nullptr;
			if (source != nullptr) {
				AsyncUpload::Get().Push(mapped + copy.srcOffset, source, copy.size, address);
			} else {
				ReadUploadSource(mapped + copy.srcOffset, address, copy.size);
			}
			copy.srcOffset += base_offset;
		}
		if (async) AsyncUpload::Get().Kick();
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		ReadUploadSource(temporary->Mapped().data() + copy.srcOffset, buffer.CpuAddress() + copy.dstOffset, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

BufferId BufferCache::EnsureBufferContents(uint64_t vaddr, uint64_t size) {
	const auto id     = FindBuffer(vaddr, size);
	auto&      buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
	return id;
}

StreamBuffer& BufferCache::GetShaderUploadBuffer() noexcept {
	return kyty_local_stream_upload_mode.load(std::memory_order_relaxed) != 0 ? m_host_shader_upload
	                                                                        : m_stream_buffer;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	DrainGuestReadback(vaddr, size, !is_written && !is_texel_buffer, is_written);
	if (LiveTrace::WriteTicks())
		LiveTrace::Event(LiveTrace::BufferUse, vaddr, size | (is_written ? uint64_t {1} << 63u : 0));
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	// CPU-modified first: a clean range (the usual case) stops here and skips the GPU query.
	if (!is_written && size <= CACHING_PAGESIZE &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		auto& upload = GetShaderUploadBuffer();
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 4);
		auto [mapped, offset] = upload.Map(size, alignment, false);
		if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBackingToHost(vaddr, mapped, size)) {
			upload.Commit();
			return {&upload, offset};
		}
	}

	// A range written again with nothing changed since (m_write_memo): no lookup, synchronization or range Add.
	// (Light-loop chains write the same few shared blocks in every dispatch: about 2100 of 3200 shader-written
	// ranges a frame at Latria were repeats, each costing ObtainBuffer a page-table lookup, a GPU-written coverage
	// query and a RangeSet Add; ~57% of written requests hit.)
	auto& memo = m_write_memo[((vaddr >> 4u) ^ (size * 0x9e3779b97f4a7c15ull >> 40u)) % m_write_memo.size()];
	if (is_written && memo.vaddr == vaddr && memo.size == size && memo.texel == is_texel_buffer &&
	    memo.registration == RegistrationEpoch() && memo.coverage == m_gpu_coverage_epoch &&
	    kyty_local_write_memo_mode.load(std::memory_order_relaxed) != 0) {
		auto* hit = m_slot_buffers.try_get(memo.id);
		if (hit != nullptr && !hit->is_deleted && hit->IsInBounds(vaddr, size)) {
			TouchBuffer(*hit);
			InvalidateCopyFeedback(vaddr, size);
			NoteGpuWrite(vaddr, size, GpuWriteOrigin::WritableBinding);
			LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWrites, 1);
			LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWriteBytes, size);
			return {hit, hit->Offset(vaddr)};
		}
	}

	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(vaddr, size)) {
		id     = FindBuffer(vaddr, size);
		buffer = &m_slot_buffers[id];
	}
	TouchBuffer(*buffer);
	(void)SynchronizeBuffer(*buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		InvalidateCopyFeedback(vaddr, size);
		m_gpu_modified_ranges.Add(vaddr, size);
		NoteGpuWrite(vaddr, size, GpuWriteOrigin::WritableBinding);
		LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWrites, 1);
		LiveCounters::AddGranule(vaddr, LiveCounters::GGpuWriteBytes, size);
		memo = {vaddr, size, RegistrationEpoch(), m_gpu_coverage_epoch, id, is_texel_buffer};
	}
	return {buffer, buffer->Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	using Clock = std::chrono::steady_clock;
	Clock::time_point marks[4] {Clock::now()};
	const char*       path = "staging";
	SlowLog::Scope    slow([&](double ms) {
        const auto at = [&](int i) {
            return marks[i] == Clock::time_point {}
		               ? 0.0
		               : std::chrono::duration<double, std::milli>(marks[i] - marks[0]).count();
        };
        std::printf("SLOW ObtainBufferForImage %.1f ms addr=0x%llx size=0x%llx path=%s drain=%.1f gpu_check=%.1f "
		               "map=%.1f\n",
		               ms, static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size), path,
		               at(1), at(2), at(3));
    }, SlowLog::HitchThreshold());
	DrainGuestReadback(vaddr, size);
	marks[1] = Clock::now();
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			path = "buffer";
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		path = "gpu";
		return ObtainBuffer(vaddr, size, false, false);
	}
	// More than the whole staging ring holds (a texture over a streamed pool expanded to 1.35 GB while
	// playing in Boletaria, and the upload exited): a buffer over the range, which SynchronizeBuffer
	// fills a staging-sized piece at a time (a temporary buffer past that) and only where memory is mapped.
	if (size > m_staging_buffer.Size()) {
		path = "large";
		return ObtainBuffer(vaddr, size, false, false);
	}
	marks[2] = Clock::now();

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	marks[3] = Clock::now();
	// KYTY_ASYNC_UPLOAD=2: the upload worker fills the staging copy too (the image upload only
	// records GPU work on it, and every submission waits for the copies pushed before it).
	if (staging != nullptr && kyty_local_async_upload_mode.load(std::memory_order_relaxed) >= 2 &&
	    m_staging_buffer.IsCoherent()) {
		bool queued = false;
		if (const auto* source = Libs::LibKernel::Memory::TryGetBackingPointer(vaddr, size)) {
			AsyncUpload::Get().Push(staging, source, size, vaddr);
			queued = true;
		} else if (Libs::LibKernel::Memory::TryGetBackingPieces(vaddr, size, m_backing_pieces)) {
			// Large images usually span several guest mappings.
			uint64_t offset = 0;
			for (const auto& [piece, bytes]: m_backing_pieces) {
				AsyncUpload::Get().Push(staging + offset, piece, bytes, vaddr + offset);
				offset += bytes;
			}
			queued = true;
		} else {
			// Partly unmapped (a texture pool with released layers): the worker reads the mapped
			// parts; the render thread copied hundreds of MiB here synchronously before.
			path = "sparse";
			AsyncUpload::Get().PushCall(ReadMappedOrZeroCall, reinterpret_cast<void*>(vaddr),
			                            reinterpret_cast<uint64_t>(staging), size, vaddr, size);
			queued = true;
		}
		if (queued) {
			AsyncUpload::Get().Kick();
			LiveCounters::Add(LiveCounters::AsyncImageBytes, size);
			m_staging_buffer.Commit();
			return {&m_staging_buffer, stage_offset};
		}
	}
	const char* prt_failure = "not-attempted";
	if (staging == nullptr || (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	                           !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size, &prt_failure))) {
		EXIT("BufferCache: failed to read mapped guest image backing: "
		     "address=0x%016" PRIx64 " size=0x%016" PRIx64
		     " staging=%p staging_capacity=0x%016" PRIx64 " mapped=%u prt_failure=%s\n",
		     vaddr, size, static_cast<void*>(staging), m_staging_buffer.Size(),
		     m_resources != nullptr && m_resources->IsMapped(vaddr, size), prt_failure);
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

std::pair<Buffer*, uint64_t> BufferCache::StageImagePieces(const std::vector<StagingPiece>& pieces,
                                                           uint64_t total, ImageStageFailure* failure,
                                                           bool sources_prevalidated) {
	if (failure != nullptr) *failure = ImageStageFailure::None;
	if (sources_prevalidated) {
		if (total > m_staging_buffer.Size()) {
			if (failure != nullptr) *failure = ImageStageFailure::Capacity;
			return {nullptr, 0};
		}
		for (const auto& piece: pieces) {
			if (!GuestRange {piece.vaddr, piece.size}.Valid() || piece.offset > total ||
			    piece.size > total - piece.offset) {
				EXIT("BufferCache: invalid prevalidated image staging piece\n");
			}
		}
	} else {
		const auto source_failure = ValidateImageStageSources(pieces, total);
		if (source_failure != ImageStageFailure::None) {
			if (failure != nullptr) *failure = source_failure;
			return {nullptr, 0};
		}
	}
	auto [staging, stage_offset] = m_staging_buffer.Map(total, 16);
	if (staging == nullptr) {
		if (failure != nullptr) *failure = ImageStageFailure::RingUnavailable;
		return {nullptr, 0};
	}
	const bool async = kyty_local_async_upload_mode.load(std::memory_order_relaxed) >= 2 &&
	                   m_staging_buffer.IsCoherent();
	bool queued = false;
	for (const auto& piece: pieces) {
		auto* target = staging + piece.offset;
		if (async) {
			if (const auto* source = Libs::LibKernel::Memory::TryGetBackingPointer(piece.vaddr, piece.size)) {
				AsyncUpload::Get().Push(target, source, piece.size, piece.vaddr);
				queued = true;
				continue;
			}
			if (Libs::LibKernel::Memory::TryGetBackingPieces(piece.vaddr, piece.size, m_backing_pieces)) {
				uint64_t offset = 0;
				for (const auto& [source, bytes]: m_backing_pieces) {
					AsyncUpload::Get().Push(target + offset, source, bytes, piece.vaddr + offset);
					offset += bytes;
				}
				queued = true;
				continue;
			}
		}
		// Part of the span is unmapped (a streamed texture pool layer the game released): those
		// pages read as zeros, the rest as they are. The unmapped pages are marked dirty again
		// when memory is mapped there (TextureCache::MapMemory).
		if (async) {
			AsyncUpload::Get().PushCall(ReadMappedOrZeroCall, reinterpret_cast<void*>(piece.vaddr),
			                            reinterpret_cast<uint64_t>(target), piece.size, piece.vaddr, piece.size);
			queued = true;
			continue;
		}
		ReadMappedOrZeroCall(reinterpret_cast<void*>(piece.vaddr), reinterpret_cast<uint64_t>(target),
		                     piece.size);
	}
	if (queued) {
		AsyncUpload::Get().Kick();
		LiveCounters::Add(LiveCounters::AsyncImageBytes, total);
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

BufferCache::ImageStageFailure BufferCache::ValidateImageStageSources(
	const std::vector<StagingPiece>& pieces, uint64_t total, uint64_t* rejected_address) {
	if (total > m_staging_buffer.Size()) return ImageStageFailure::Capacity;
	for (const auto& piece: pieces) {
		if (!GuestRange {piece.vaddr, piece.size}.Valid() || piece.offset > total ||
		    piece.size > total - piece.offset) {
			EXIT("BufferCache: invalid image staging piece\n");
		}
		DrainGuestReadback(piece.vaddr, piece.size);
		const auto* owner = m_page_table.Find(piece.vaddr >> PageTable::kPageBits);
		if ((owner != nullptr && *owner) || IsRegionGpuModified(piece.vaddr, piece.size)) {
			if (rejected_address != nullptr) *rejected_address = piece.vaddr;
			return ImageStageFailure::SourceOwned;
		}
	}
	return ImageStageFailure::None;
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr, value);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, "fill");
	const auto id          = FindBuffer(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true, id);
	EXIT_IF(dst == nullptr);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		// The game's linear copies mostly rewrite what the destination already holds (95% of the
		// bytes in the fixed scene): only the destination pages whose bytes differ are written, so
		// the others stay clean (no write fault, no upload of their bytes to the GPU again).
		for (uint64_t at = 0; at < size;) {
			const auto bytes = std::min(TRACKER_PAGE_SIZE - (dst_vaddr + at) % TRACKER_PAGE_SIZE, size - at);
			auto*       to   = reinterpret_cast<void*>(dst_vaddr + at);
			const auto* from = reinterpret_cast<const void*>(src_vaddr + at);
			if (std::memcmp(to, from, bytes) != 0) std::memcpy(to, from, bytes);
			at += bytes;
		}
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, "copy");
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	EXIT_IF(src == nullptr || dst == nullptr);
	if (src_memory && src->is_deleted) {
		// Obtaining the destination created a buffer that joined (and deleted) the source's. The
		// deleted buffer is freed once the current tick completes, and that synchronization may
		// already have submitted the tick the join was recorded in: copy from the joined buffer.
		const auto joined = ObtainBuffer(src_vaddr, size, false, true, BufferId {});
		src        = joined.first;
		src_offset = joined.second;
		EXIT_IF(src == nullptr || src->is_deleted);
	}
	if (src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size) {
		EXIT("BufferCache: resolved Vulkan copy ranges overlap\n");
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	if (src_memory && dst_memory) ForwardCopy(dst_vaddr, src_vaddr, size);
}

// KYTY_COPY_FORWARD=1: a copy whose source the CPU holds current (no GPU write pending on it) also lands
// in the destination's guest memory when it is recorded, and those bytes stop being GPU-written: a later
// CPU read (an SRT evaluation reading constants the game copies on the GPU every frame, Demon's Souls'
// copy kernels) is served from memory instead of draining the GPU. The GPU copy stays, so the GPU buffer
// holds the same bytes once it runs. Not over images, not under a pending guest readback (its write-back
// could land older bytes), and only bytes: pages keep GPU ownership while other GPU-written bytes remain.
void BufferCache::ForwardCopy(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size) {
	static const bool forward = [] {
		const char* value = std::getenv("KYTY_COPY_FORWARD");
		return value != nullptr && value[0] == '1';
	}();
	constexpr uint64_t ForwardMax = 16ull << 20u;
	if (!forward || size > ForwardMax || IsRegionGpuModified(src_vaddr, size) ||
	    m_texture_cache.FindImageFromRange(src_vaddr, size) || m_texture_cache.HasTrackedDataOverlap(dst_vaddr, size) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(src_vaddr, size) ||
	    !LibKernel::Memory::IsUniqueGuestBackingRange(dst_vaddr, size)) {
		return;
	}
	const auto begin = dst_vaddr & ~(TRACKER_PAGE_SIZE - 1);
	const auto end   = (dst_vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	for (const auto& pending: m_guest_readbacks)
		if (pending && pending->begin < end && begin < pending->begin + pending->size) return;
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(size);
	if (!LibKernel::Memory::TryReadBackingToHost(src_vaddr, bytes.data(), size)) return;
	WriteBackGpuOwned(dst_vaddr, bytes.data(), size, "forward");
	m_gpu_modified_ranges.Subtract(dst_vaddr, size);
	++m_gpu_coverage_epoch; // (the write memo: these bytes are no longer GPU-written)
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE)
		if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE))
			m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
	LiveCounters::Add(LiveCounters::CopyForwards);
	LiveCounters::Add(LiveCounters::CopyForwardBytes, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

// KYTY_BUFFER_RETIRE (live; 0 never retires). Per flip, up to 64 buffers no binding used for 600 flips go, whatever
// the memory use. The collector below waits for a frame without BDA shaders, which Demon's Souls never has, so every
// buffer it registered stayed; those inside the BDA shaders' read windows had their CPU-written pages uploaded and
// write-protected again at each BDA preparation. Passed over (moved to the back of the list): buffers with GPU-owned
// bytes (their download stays the collector's) and buffers BDA shaders read with no buffer registered (bda_read: such
// reads touch no LRU entry, and one after the buffer went would read zeros once).
void BufferCache::RetireStaleBuffers(uint64_t tick) {
	constexpr uint64_t StaleTicks     = 600;
	constexpr size_t   MaxRetirements = 64;
	if (kyty_local_buffer_retire_mode.load(std::memory_order_relaxed) == 0 || tick <= StaleTicks) return;
	thread_local std::vector<BufferId> stale, kept;
	stale.clear();
	kept.clear();
	m_lru_cache.ForEachItemBelow(tick - StaleTicks, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		if (buffer.is_deleted) return false;
		// The guest readbacks over the buffer complete first (the collector drains each candidate the same way):
		// their copies may still read it, and the GPU-owned bytes they hold decide below.
		DrainGuestReadback(buffer.CpuAddress(), buffer.Size());
		if (buffer.bda_read || m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
			m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size()))
			kept.push_back(id);
		else
			stale.push_back(id);
		return stale.size() == MaxRetirements;
	});
	// (The ones passed over go to the back of the list: the next flips' walks do not cross them again.)
	for (const auto id: kept) TouchBuffer(m_slot_buffers[id]);
	if (stale.empty()) return;
	for (const auto id: stale) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		// DeleteBuffer unregisters it (its BDA page table entries go) and frees it once the GPU has passed its tick.
		DeleteBuffer(id);
	}
	LiveCounters::Add(LiveCounters::GcBufferDeletes, stale.size());
}

void BufferCache::RunGarbageCollector(bool collect) {
	const auto tick = m_gc_tick++;
	RetireStaleBuffers(tick);
	// After a frame with BDA shaders a collected buffer faults straight back, so buffers are kept,
	// unless video memory is about to run out: on an 8 GB card Demon's Souls filled it by the
	// title screen (nearly every frame has BDA shaders) and the next allocation failed.
	if (!collect && (!m_graphics.CanReportMemoryUsage() ||
	                 m_graphics.GetDeviceMemoryUsage() < m_critical_gc_memory)) {
		return;
	}
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	// Only a readback over a buffer the scan looks at can change its dirty state or still copy from it, so
	// each candidate drains its own range below. Draining every readback here made the GPU thread wait for
	// the GPU once a frame whenever memory sat above the trigger (Demon's Souls in Boletaria: 14% of its
	// samples in S2, with old buffers almost never under a readback). KYTY_GC_DRAIN_ALL=1: drain all (A/B).
	static const bool drain_all = [] {
		const char* value = std::getenv("KYTY_GC_DRAIN_ALL");
		return value != nullptr && value[0] == '1';
	}();
	if (drain_all) DrainGuestReadback();

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	std::vector<DownloadCopy> copies;
	size_t                    retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (!drain_all) DrainGuestReadback(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			m_memory_tracker.ForEachDownloadRange<false>(
			    buffer.CpuAddress(), buffer.Size(),
			    [&](uint64_t dirty_address, uint64_t dirty_size) noexcept {
				    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, dirty_address,
				                                           dirty_size, "garbage collection");
			    },
			    [&](uint64_t dirty_address, uint64_t dirty_size) noexcept {
				    m_gpu_modified_ranges.ForEachIntersection(
				        dirty_address, dirty_size, [&](RangeSet::Range range) {
					    copies.push_back({&buffer, range.address - buffer.CpuAddress(),
					                      range.address, range.size});
				        });
				});
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	EXIT_IF(copies.empty());
	DownloadBufferMemory(copies);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		// Freed once the GPU is done with it: a download on the readback queue (KYTY_READBACK_QUEUE)
		// waits only for the last write, not for the reads in flight or the commands recorded so far.
		DeleteBuffer(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	// This records into FaultManager's own buffer. Its later registration callbacks
	// pass through ChangeRegister, which drains before changing a guest owner.
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::CollectMappedRegisteredRanges(const RangeSet& mapped,
                                                std::vector<RangeSet::Range>& ranges) const {
	ranges.clear();
	for (const auto& [address, id] : m_buffers) {
		mapped.ForEachIntersection(address, m_slot_buffers[id].Size(), [&](RangeSet::Range range) {
			if (!ranges.empty() && ranges.back().address + ranges.back().size == range.address)
				ranges.back().size += range.size;
			else ranges.push_back(range);
		});
	}
}

void BufferCache::SynchronizeRegionRequest(SyncRegionRequest& request) {
	SlowLog::Scope slow([&](double ms) {
		std::printf("SLOW SynchronizeRegionRequest %.1f ms addr=0x%llx size=0x%llx\n", ms,
		            static_cast<unsigned long long>(request.address), static_cast<unsigned long long>(request.size));
	});
	DrainGuestReadback(request.address, request.size, true);
	const auto epoch = m_memory_tracker.CpuModificationEpoch(request.address, request.size);
	// A request lies in one tracker region: registrations elsewhere do not touch its buffers.
	const auto registered = RegionRegistrationEpoch(request.address);
	if (epoch != 0 && request.cpu_epoch == epoch && request.registration_epoch == registered) {
		LiveCounters::Add(LiveCounters::RegionSkips);
		return;
	}
	LiveCounters::Add(LiveCounters::RegionSyncs);
	SynchronizeBuffersInRange(request.address, request.size);
	request.cpu_epoch = 0;
	if (epoch != 0 && RegionRegistrationEpoch(request.address) == registered &&
	    m_memory_tracker.CpuModificationEpoch(request.address, request.size) == epoch) {
		request.cpu_epoch = epoch;
		request.registration_epoch = registered;
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	DrainGuestReadback(vaddr, size, true);
	if (!m_sync_buffers_valid) {
		// The pieces of buffers that stayed registered keep their stamps: a stamp is about its own buffer's
		// contents against the CPU epoch of its region, which another buffer's registration does not move.
		std::swap(m_sync_buffers, m_old_sync_buffers);
		std::swap(m_sync_stamps, m_old_sync_stamps);
		m_sync_buffers.clear();
		m_sync_stamps.clear();
		m_sync_buffers.reserve(m_buffers.size());
		size_t old = 0;
		for (const auto& [address, id]: m_buffers) {
			auto& buffer = m_slot_buffers[id];
			const auto end = address + buffer.Size();
			for (auto start = address; start < end;) {
				const auto finish = std::min(end, (start / TRACKER_REGION_SIZE + 1) * TRACKER_REGION_SIZE);
				while (old < m_old_sync_buffers.size() && m_old_sync_buffers[old].start < start) ++old;
				const bool kept = old < m_old_sync_buffers.size() && m_old_sync_buffers[old].start == start &&
				                  m_old_sync_buffers[old].end == finish && m_old_sync_buffers[old].id == id;
				m_sync_buffers.push_back({start, finish, &buffer, id});
				m_sync_stamps.push_back(kept ? m_old_sync_stamps[old] : SyncStamp {});
				start = finish;
			}
		}
		m_sync_buffers_valid = true;
	}
	const auto end = vaddr + size;

	auto it = std::lower_bound(m_sync_buffers.begin(), m_sync_buffers.end(), vaddr,
	                           [](const SyncBuffer& buffer, uint64_t address) {
		                           return buffer.end <= address;
	                           });
	for (; it != m_sync_buffers.end() && it->start < end; ++it) {
		const auto start  = std::max(it->start, vaddr);
		const auto finish = std::min(it->end, end);
		if (start < finish) {
			SyncStamp* stamp = nullptr;
			uint64_t epoch = 0;
			{
				stamp = &m_sync_stamps[static_cast<size_t>(it - m_sync_buffers.begin())];
				epoch = m_memory_tracker.CpuModificationEpoch(start, finish - start);
				if (epoch != 0 && stamp->epoch == epoch && start >= stamp->begin && finish <= stamp->end) continue;
			}
			(void)SynchronizeBuffer(*it->buffer, start, finish - start, false, false);
			if (stamp != nullptr && epoch != 0 && m_sync_buffers_valid &&
			    m_memory_tracker.CpuModificationEpoch(start, finish - start) == epoch) {
				*stamp = {start, finish, epoch};
			}
		}
	}

}

} // namespace Libs::Graphics
