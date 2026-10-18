#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <algorithm>
#include <compare>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

extern "C" {
extern volatile std::atomic<uint32_t> kyty_local_partial_mixed_upload_mode;
}

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
	vk::ImageAspectFlags pixel_sampled_aspects;
	vk::ImageAspectFlags other_sampled_aspects;
};

// Scopes a group of transitions that all precede the same recorded command. No
// command executes between transitions in one group, so repeating an identical
// transition of the same image inside a group cannot be covering a hazard.
// Transitions recorded outside any group are never treated as repeats, which is
// what keeps the copy and upload paths correct.
void BeginTransitGroup() noexcept;
void EndTransitGroup() noexcept;

class TransitGroup final {
public:
	TransitGroup() noexcept { BeginTransitGroup(); }
	~TransitGroup() { EndTransitGroup(); }
	KYTY_CLASS_NO_COPY(TransitGroup);
};

class Image final {
public:
	// `may_fail`: an image the GPU cannot make (its memory full, more layers or bytes than the format allows) is left
	// without one (CreateFailed) instead of ending the emulator: a sampled texture's lookup binds a null image instead.
	// One that must exist is tried past those limits, and left without one only when its memory could not be had
	// (TextureCache::InsertImage makes room and tries again).
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info, bool may_fail = false);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] bool CreateFailed() const noexcept { return m_create_failed; }

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	// Levels of the host image (fewer than the guest's with reduced texture quality).
	[[nodiscard]] uint32_t HostLevels() const noexcept { return info.resources.levels - info.host_mip_skip; }
	// Transit(layout, eShaderRead, range) would record nothing: the whole image is already in `layout` for shader
	// reads (GetBarriers' no-op case: no per-subresource states and a range of every guest level and layer, which
	// GetBarriers maps to every host level whatever the host mip skip). Repeated shader reads need no barrier and
	// Transit then changes no state.
	[[nodiscard]] bool ReadyForShaderRead(vk::ImageLayout layout, const ImageSubresourceRange& range) const noexcept {
		return backing.state.layout == layout && backing.state.access_mask == vk::AccessFlagBits2::eShaderRead &&
		       backing.subresource_states.empty() && !info.IsVolume() && info.host_mip_skip < info.resources.levels &&
		       range.base_level == 0 && range.level_count == info.resources.levels && range.base_layer == 0 &&
		       range.layer_count == info.resources.layers;
	}
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] const Barriers& GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	// Moves whenever an image becomes (maybe) CPU-dirty: a proof that bound images were clean
	// holds while it stays (native XPR records used again within a frame).
	[[nodiscard]] static uint64_t CpuDirtyEpoch() noexcept { return s_cpu_dirty_epoch.load(std::memory_order_acquire); }

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
			DropPartialDirty();
			NoteCpuDirty();
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
			NoteCpuDirty();
		}
	}

	// Partial CPU writes (KYTY_PARTIAL_IMAGE_DIRTY): while the image is dirty only in the
	// recorded guest ranges, it keeps watching its other pages, so a refresh uploads just the
	// subresources over those ranges. A clean image or one already dirty this way can take one.
	[[nodiscard]] bool CanTakePartialDirty() const noexcept {
		return !m_maybe_cpu_dirty && (!m_cpu_dirty || m_partial_dirty);
	}
	void InvalidateCpuWritePartial(uint64_t begin, uint64_t end) {
		if (!CanTakePartialDirty() || begin >= end) {
			EXIT("image cannot take a partial CPU write\n");
		}
		if (!m_cpu_dirty) {
			m_cpu_dirty        = true;
			m_partial_dirty    = true;
			m_maybe_hash_valid = false;
			m_dirty_ranges.clear();
		}
		AddRange(m_dirty_ranges, begin, end);
		NoteCpuDirty();
	}
	[[nodiscard]] bool IsPartiallyCpuDirty() const noexcept { return m_cpu_dirty && m_partial_dirty; }
	[[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& CpuDirtyRanges() const noexcept {
		return m_dirty_ranges;
	}
	// The whole image must be uploaded again (the image stopped watching all of its pages).
	void DropPartialDirty() noexcept {
		m_partial_dirty = false;
		m_dirty_ranges.clear();
	}
	// Sorted, disjoint [begin, end) ranges; merges adjacent and overlapping ones.
	static void AddRange(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t begin,
	                     uint64_t end) {
		auto at = ranges.begin();
		while (at != ranges.end() && at->second < begin) ++at;
		auto last = at;
		while (last != ranges.end() && last->first <= end) {
			begin = std::min(begin, last->first);
			end   = std::max(end, last->second);
			++last;
		}
		at = ranges.erase(at, last);
		ranges.insert(at, {begin, end});
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
			NoteCpuDirty();
		}
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		DropPartialDirty();
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept { m_gpu_modified = false; }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	// GPU buffer writes changed guest bytes of the image, which ones unknown.
	void MarkBufferModified() noexcept {
		m_buffer_modified = true;
		m_buffer_whole    = true;
		m_buffer_ranges.clear();
	}
	// GPU buffer writes changed the image's guest bytes [begin, end): kept as ranges until the image is
	// refreshed, so a write over part of it (or over an aliased smaller image) refreshes only that part
	// (TextureCache::UploadImagePartial from the buffer). Too many ranges count as the whole image.
	void MarkBufferModified(uint64_t begin, uint64_t end) noexcept {
		if (!m_buffer_modified) {
			m_buffer_modified = true;
			m_buffer_whole    = false;
			m_buffer_ranges.clear();
		}
		if (m_buffer_whole || begin >= end) {
			m_buffer_whole = true;
			return;
		}
		AddRange(m_buffer_ranges, begin, end);
		// (More in a large image with KYTY_PARTIAL_MIXED_UPLOAD, an image uploaded a piece at a time anyway (256 MiB and up,
		// TextureCache's ChunkedUploadMin): where the whole is GBs, a texture pool's range over memory the GPU writes in
		// many places, issue #49. Otherwise a few.)
		const bool large_ranges = kyty_local_partial_mixed_upload_mode.load(std::memory_order_relaxed) != 0 &&
		                          info.data.size >= (uint64_t {256} << 20u);
		if (m_buffer_ranges.size() > (large_ranges ? 256u : 32u)) {
			m_buffer_whole = true;
			m_buffer_ranges.clear();
		}
	}
	void ClearBufferModified() noexcept {
		m_buffer_modified = false;
		m_buffer_whole    = false;
		m_buffer_ranges.clear();
	}
	// The buffer-written ranges, when known (empty: the whole image or none).
	[[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& BufferDirtyRanges() const noexcept {
		return m_buffer_whole ? s_no_ranges : m_buffer_ranges;
	}

	[[nodiscard]] bool IsStencilModified() const noexcept { return m_stencil_modified; }
	void               MarkStencilModified() noexcept { m_stencil_modified = true; }
	void               ClearStencilModified() noexcept { m_stencil_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : (info.data.size + 1023) & ~uint64_t {1023};
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	// Page-aligned ranges inside the tracked pages the image stopped watching after partial
	// CPU writes; always inside CpuDirtyRanges while the image is partially dirty.
	std::vector<std::pair<uint64_t, uint64_t>> untracked_holes;
	// KYTY_PARTIAL_IMAGE_DIRTY=2: guest data hashes per partial-dirty granule at the last upload.
	std::vector<uint64_t> partial_hashes;
	ImageId          depth_id {};
	uint64_t         frame_accessed_last = 0; // TextureCache::AdvanceFrame count at the last use
	size_t           lru_id             = 0;
	// Transit group that last set the whole-image state; see BeginTransitGroup.
	uint64_t         transit_group      = 0;
	// Unique per image object: a deleted image's slot id goes to later images.
	uint64_t         serial             = 0;
	// TextureCache frame of the last full upload (KYTY_TEXTURE_RAM_TRIM cooldown; UINT64_MAX: never).
	uint64_t         frame_uploaded_last = UINT64_MAX;
	// RegisterImage calls on this object: a proof names one registration.
	uint32_t         registrations      = 0;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	bool              m_create_failed    = false;
	uint64_t          m_maybe_cpu_hash   = 0;
	static void NoteCpuDirty() noexcept { s_cpu_dirty_epoch.fetch_add(1, std::memory_order_release); }
	inline static std::atomic<uint64_t> s_cpu_dirty_epoch {1};
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
	bool              m_stencil_modified  = false;
	bool              m_partial_dirty     = false;
	bool              m_buffer_whole      = false;
	std::vector<std::pair<uint64_t, uint64_t>> m_dirty_ranges;
	std::vector<std::pair<uint64_t, uint64_t>> m_buffer_ranges;
	inline static const std::vector<std::pair<uint64_t, uint64_t>> s_no_ranges {};
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
