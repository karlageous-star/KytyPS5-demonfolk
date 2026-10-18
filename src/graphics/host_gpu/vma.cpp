#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "live-counters.h"

#include <cstdio>

#include <algorithm>
#include <cinttypes>

namespace Libs::Graphics {

void FlushBufferReclaimer(); // streamBuffer.cpp (KYTY_BUFFER_RECLAIM)

} // namespace Libs::Graphics

extern "C" {
// KYTY_VRAM_FIRST (default 1; 0: as before, for an A/B): a game buffer or image asks for video memory within the
// budget, then without the freed images kept for reuse, then past the budget (Windows moves out what it uses
// least), and gets system memory only when all of that fails. VMA's budgeted choice among all memory types put
// game buffers in system memory while 1.3 GB of video memory sat free inside its blocks (N3: four buffers,
// 1.05 GB, read over the bus every frame). Images of 16 MiB or more get their own memory: freed, they no longer
// leave holes in the shared blocks (N3: 17 holes of 16 MiB or more, 769 MiB).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_vram_first_mode {1};
}

#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned short __stdcall RtlCaptureStackBackTrace(unsigned long frames_to_skip,
                                                                                  unsigned long frames_to_capture,
                                                                                  void** back_trace,
                                                                                  unsigned long* back_trace_hash);
#endif

namespace Libs::Graphics {

namespace {
// The GPU memory blocks in system memory (heaps without DEVICE_LOCAL) the allocator holds, and which made them: a block
// of 64 MiB or more, and every 256 MiB more of them, is logged with the return addresses that asked for it. A heavy
// scene's walk put +1.2 GB of private memory on within a second outside the working set (S39, 10-19: the RAM watchdog
// ended the run at 40 MiB free) with video memory not full.
std::atomic<int64_t>                    g_system_block_bytes {0};
std::atomic<int64_t>                    g_system_block_report {int64_t {256} << 20};
VkPhysicalDeviceMemoryProperties        g_block_memory_properties {};
bool                                    BlockInSystemMemory(uint32_t type) {
    const auto heap = g_block_memory_properties.memoryTypes[type].heapIndex;
    return (g_block_memory_properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0;
}
void VKAPI_PTR OnBlockAllocate(VmaAllocator, uint32_t type, VkDeviceMemory, VkDeviceSize size, void*) {
	const bool    system = BlockInSystemMemory(type);
	const int64_t total  = system ? (g_system_block_bytes += static_cast<int64_t>(size)) : g_system_block_bytes.load();
	int64_t       due    = g_system_block_report.load();
	const bool    large  = size >= (uint64_t {64} << 20u);
	const bool    step   = system && total >= due && g_system_block_report.compare_exchange_strong(due, total + (int64_t {256} << 20));
	if (!large && !step) return;
	std::printf("VMA block: %llu MiB of type %u (%s); system-memory blocks %lld MiB; asked from",
	            static_cast<unsigned long long>(size >> 20u), type, system ? "system memory" : "video memory",
	            static_cast<long long>(total >> 20));
#if defined(_WIN32)
	void*      frames[12] {};
	const auto count = RtlCaptureStackBackTrace(1, 12, frames, nullptr);
	for (unsigned short i = 0; i < count; ++i) std::printf(" %p", frames[i]);
#endif
	std::printf("\n");
	std::fflush(stdout);
}
void VKAPI_PTR OnBlockFree(VmaAllocator, uint32_t type, VkDeviceMemory, VkDeviceSize size, void*) {
	if (BlockInSystemMemory(type)) g_system_block_bytes -= static_cast<int64_t>(size);
}
const VmaDeviceMemoryCallbacks g_block_callbacks {OnBlockAllocate, OnBlockFree, nullptr};
std::atomic<int64_t>  g_system_fallback_bytes {0};
std::atomic<uint64_t> g_over_budget_bytes {0};
} // namespace

void NoteSystemMemoryFallback(int64_t bytes) {
	g_system_fallback_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

uint64_t SystemMemoryFallbackBytes() {
	return static_cast<uint64_t>(std::max<int64_t>(g_system_fallback_bytes.load(std::memory_order_relaxed), 0));
}

uint64_t VideoMemoryOverBudgetBytes() {
	return g_over_budget_bytes.load(std::memory_order_relaxed);
}

// Local diagnostic (live `vma <path>`): VMA is thread-safe, so the live thread writes it.
static VmaAllocator g_report_allocator = nullptr;
// Freed images kept for reuse (KYTY_IMAGE_POOL): at most 1 GiB, and a 16th of the GPU's memory budget.
static uint64_t g_image_pool_limit = 1024ull << 20;
static void WriteVmaReport(const char* path) {
	if (g_report_allocator == nullptr) return;
	char* json = nullptr;
	vmaBuildStatsString(g_report_allocator, &json, VK_TRUE);
	if (FILE* file = std::fopen(path, "wb"); file != nullptr) {
		std::fputs(json, file);
		std::fclose(file);
	}
	vmaFreeStatsString(g_report_allocator, json);
}

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	g_block_memory_properties = static_cast<VkPhysicalDeviceMemoryProperties>(physical_device_memory_properties);
	info.pDeviceMemoryCallbacks = &g_block_callbacks;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	// KYTY_VRAM_LIMIT_MB=<n>: video memory as on a GPU with n MiB (VMA's heap size limit): its budget,
	// and allocations past it failing, for tests of memory pressure on smaller GPUs.
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_limits {};
	if (const char* text = std::getenv("KYTY_VRAM_LIMIT_MB"); text != nullptr && std::strtoull(text, nullptr, 10) > 0) {
		heap_limits.fill(VK_WHOLE_SIZE);
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				heap_limits[heap] = std::strtoull(text, nullptr, 10) << 20u;
			}
		}
		info.pHeapSizeLimit = heap_limits.data();
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	g_report_allocator          = allocator;
	LiveCounters::g_vma_report = WriteVmaReport;
	g_image_pool_limit         = std::min<uint64_t>(1024ull << 20, GetTotalMemoryBudget() / 16);
	return true;
}

static void DestroyImagePool(VmaAllocator allocator);

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	LiveCounters::g_vma_report = nullptr;
	g_report_allocator         = nullptr;
	FlushBufferReclaimer();
	DestroyImagePool(allocator);
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0, over = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
		if (device_local && budgets[heap].usage > budgets[heap].budget) over += budgets[heap].usage - budgets[heap].budget;
	}
	// (The window title's SHARED figure: past the budget Windows moves allocations out of video memory.)
	g_over_budget_bytes.store(over, std::memory_order_relaxed);
	return usage;
}

uint64_t GraphicContext::GetVideoMemoryRoom() const {
	// (Without VK_EXT_memory_budget VMA counts only its own blocks against a guessed budget: no figure to go by.)
	if (allocator == nullptr || !CanReportMemoryUsage()) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto& heaps = physical_device_memory_properties.memoryHeaps;
	uint32_t    video = UINT32_MAX;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local = static_cast<bool>(heaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		const bool larger       = video == UINT32_MAX || heaps[heap].size > heaps[video].size;
		if (device_local && larger) video = heap;
	}
	if (video == UINT32_MAX) {
		return 0;
	}
	const auto& budget = budgets[video];
	return budget.budget > budget.usage ? budget.budget - budget.usage : 0;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

// Recycling of freed images (KYTY_IMAGE_POOL). The game aliases transient render
// targets in one heap, so the texture cache deletes and re-creates the same few
// images every frame. An image reaches DeleteImage only after the GPU is done
// with it; a new image always starts in eUndefined layout, so reusing the VkImage
// and its memory for identical creation parameters is equivalent to a fresh
// allocation.
extern "C" {
volatile std::atomic<uint32_t> kyty_local_image_pool_mode {0};
}
namespace {
struct PooledImage {
	std::array<uint32_t, 10> key {};
	VkImage                  image      = VK_NULL_HANDLE;
	VmaAllocation            allocation = nullptr;
	uint64_t                 bytes      = 0;
};
std::mutex               g_image_pool_mutex;
std::vector<PooledImage> g_image_pool;
uint64_t                 g_image_pool_bytes = 0;

std::array<uint32_t, 10> ImagePoolKey(const vk::ImageCreateInfo& info) {
	return {static_cast<uint32_t>(static_cast<VkImageCreateFlags>(info.flags)), static_cast<uint32_t>(info.imageType),
	        static_cast<uint32_t>(info.format), info.extent.width, info.extent.height, info.extent.depth,
	        info.mipLevels, info.arrayLayers,
	        static_cast<uint32_t>(info.samples) | (static_cast<uint32_t>(info.tiling) << 16u),
	        static_cast<uint32_t>(static_cast<VkImageUsageFlags>(info.usage))};
}
std::array<uint32_t, 10> ImagePoolKey(const VulkanImage& image) {
	vk::ImageCreateInfo info {};
	info.flags       = image.flags;
	info.imageType   = image.image_type;
	info.format      = image.format;
	info.extent      = image.extent;
	info.mipLevels   = image.mip_levels;
	info.arrayLayers = image.layers;
	info.samples     = static_cast<vk::SampleCountFlagBits>(image.samples);
	info.tiling      = vk::ImageTiling::eOptimal;
	info.usage       = image.usage;
	return ImagePoolKey(info);
}
} // namespace

static void DestroyImagePool(VmaAllocator allocator) {
	std::scoped_lock lock(g_image_pool_mutex);
	for (const auto& pooled: g_image_pool) {
		vmaDestroyImage(allocator, pooled.image, pooled.allocation);
	}
	g_image_pool.clear();
	g_image_pool_bytes = 0;
}

void GraphicContext::TrimImagePool() {
	if (allocator != nullptr) DestroyImagePool(allocator);
}

// Out of video memory: the first few times say so in the log, with the heaps' budgets.
void GraphicContext::PrintMemoryStatistics() const {
	if (allocator == nullptr) return;
	VmaTotalStatistics stats {};
	vmaCalculateStatistics(allocator, &stats);
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	std::printf("Memory: blocks %u (%" PRIu64 " MiB), allocations %u (%" PRIu64 " MiB), image pool %" PRIu64
	            " MiB",
	            stats.total.statistics.blockCount, stats.total.statistics.blockBytes >> 20u,
	            stats.total.statistics.allocationCount, stats.total.statistics.allocationBytes >> 20u,
	            g_image_pool_bytes >> 20u);
	for (uint32_t i = 0; i < GetPhysicalDeviceMemoryProperties().memoryHeapCount; i++) {
		std::printf("; heap %u usage %" PRIu64 " of %" PRIu64 " MiB", i, budgets[i].usage >> 20u,
		            budgets[i].budget >> 20u);
	}
	std::printf("\n");
	std::fflush(stdout);
	static bool reported = false;
	const char* report   = std::getenv("KYTY_VMA_REPORT");
	const auto  heaps    = GetPhysicalDeviceMemoryProperties().memoryHeapCount;
	for (uint32_t i = 0; !reported && report != nullptr && i < heaps; i++) {
		if (budgets[i].budget != 0 && budgets[i].usage * 100 >= budgets[i].budget * 95 &&
		    (GetPhysicalDeviceMemoryProperties().memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
			reported = true;
			WriteVmaReport(report);
			std::printf("Memory: VMA report written to %s\n", report);
		}
	}
}

void GraphicContext::ReportMemoryFallback(const char* what, uint64_t bytes) const {
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) >= 8) return;
	std::printf("Vulkan: video memory full, %s (%" PRIu64 " bytes)\n", what, bytes);
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < GetPhysicalDeviceMemoryProperties().memoryHeapCount; i++) {
		std::printf("  heap %u: usage %" PRIu64 " MiB of budget %" PRIu64 " MiB\n", i, budgets[i].usage >> 20u,
		            budgets[i].budget >> 20u);
	}
	std::fflush(stdout);
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	if (kyty_local_image_pool_mode.load(std::memory_order_relaxed) != 0 && image_info.pNext == nullptr &&
	    image_info.tiling == vk::ImageTiling::eOptimal &&
	    image_info.initialLayout == vk::ImageLayout::eUndefined) {
		const auto key = ImagePoolKey(image_info);
		std::scoped_lock lock(g_image_pool_mutex);
		for (auto it = g_image_pool.begin(); it != g_image_pool.end(); ++it) {
			if (it->key != key) {
				continue;
			}
			image.image      = it->image;
			image.allocation = it->allocation;
			g_image_pool_bytes -= it->bytes;
			g_image_pool.erase(it);
			image.format     = image_info.format;
			image.image_type = image_info.imageType;
			image.extent     = image_info.extent;
			image.layers     = image_info.arrayLayers;
			image.mip_levels = image_info.mipLevels;
			image.samples    = static_cast<uint32_t>(image_info.samples);
			image.usage      = image_info.usage;
			image.flags      = image_info.flags;
			image.state      = {.layout = image_info.initialLayout};
			image.subresource_states.clear();
			return true;
		}
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	// KYTY_VRAM_FIRST: within the budget first (then without the pool, then past it); a large image in memory of
	// its own, so that freeing it leaves no hole in a shared block.
	const bool vram_first = kyty_local_vram_first_mode.load(std::memory_order_relaxed) != 0 && CanReportMemoryUsage();
	if (vram_first) {
		constexpr vk::DeviceSize DedicatedMin = vk::DeviceSize {16} << 20u;
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements != nullptr) {
			vk::DeviceImageMemoryRequirements requirements_info {};
			requirements_info.pCreateInfo = &image_info;
			const auto requirements      = device.getImageMemoryRequirements(requirements_info);
			if (requirements.memoryRequirements.size >= DedicatedMin)
				alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		}
		alloc_info.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	}
	const auto create = [&] {
		vk::Image::CType native_image = VK_NULL_HANDLE;
		const auto       result       = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
		image.image = native_image;
		return result == vk::Result::eSuccess;
	};
	// Out of video memory: without the freed images kept for reuse, then in system memory (slower
	// to sample, but the game goes on).
	if (!create()) {
		TrimImagePool();
		bool created = create();
		if (!created && (alloc_info.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0) {
			alloc_info.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
			created = create();
		}
		if (!created) {
			alloc_info.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
			alloc_info.requiredFlags  = 0;
			alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			created                   = create();
			VmaAllocationInfo allocated {};
			if (created) vmaGetAllocationInfo(allocator, image.allocation, &allocated);
			ReportMemoryFallback(created ? "an image is in system memory" : "an image could not be created",
			                     allocated.size);
			if (!created) return false;
			VkMemoryPropertyFlags properties = 0;
			vmaGetAllocationMemoryProperties(allocator, image.allocation, &properties);
			if ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
				image.in_system_memory = true;
				NoteSystemMemoryFallback(static_cast<int64_t>(allocated.size));
			}
		}
	}

	static const bool allocation_names = [] {
		const char* value = std::getenv("KYTY_VMA_ALLOCATION_NAMES");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	if (allocation_names) {
		char name[160];
		std::snprintf(name, sizeof(name), "Image %ux%ux%u layers=%u mips=%u format=%u usage=0x%x",
		              image_info.extent.width, image_info.extent.height, image_info.extent.depth,
		              image_info.arrayLayers, image_info.mipLevels, static_cast<unsigned>(image_info.format),
		              static_cast<unsigned>(image_info.usage));
		vmaSetAllocationName(allocator, image.allocation, name);
	}
	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (image.in_system_memory) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		NoteSystemMemoryFallback(-static_cast<int64_t>(allocation_info.size));
		image.in_system_memory = false;
		vmaDestroyImage(allocator, image.image, image.allocation);
		image.image      = nullptr;
		image.allocation = nullptr;
		return;
	}
	// KYTY_VRAM_FIRST: nothing is kept for reuse while video memory is past its budget or something is in system
	// memory (the pool held up to a 16th of the budget).
	const bool short_of_vram = kyty_local_vram_first_mode.load(std::memory_order_relaxed) != 0 &&
	                           (VideoMemoryOverBudgetBytes() != 0 || SystemMemoryFallbackBytes() != 0);
	if (kyty_local_image_pool_mode.load(std::memory_order_relaxed) != 0 && !short_of_vram) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		std::scoped_lock lock(g_image_pool_mutex);
		g_image_pool.push_back({ImagePoolKey(image), image.image, image.allocation, allocation_info.size});
		g_image_pool_bytes += allocation_info.size;
		while (g_image_pool_bytes > g_image_pool_limit && !g_image_pool.empty()) {
			auto& oldest = g_image_pool.front();
			vmaDestroyImage(allocator, oldest.image, oldest.allocation);
			g_image_pool_bytes -= oldest.bytes;
			g_image_pool.erase(g_image_pool.begin());
		}
		image.image      = nullptr;
		image.allocation = nullptr;
		return;
	}
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
