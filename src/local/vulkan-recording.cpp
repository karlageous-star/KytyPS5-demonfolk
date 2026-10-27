#include "vulkan-recording.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "live-census.h"
#include "live-counters.h"
#include "live-trace-gpu.h"
#include "local-platform.h"
#include "spin-gate.h"
#include "time-census.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

extern "C" {
// 1: record Vulkan commands for the worker thread, draws and descriptor
// encoding as whole packets (KYTY_VULKAN_RECORDING).
volatile std::atomic<uint32_t> kyty_local_vulkan_recording_mode {1};
// 1: queue command-buffer begin/end/submit instead of draining (DeferredSubmitEnabled).
volatile std::atomic<uint32_t> kyty_local_deferred_submit_mode {0};
// 1: back-to-back buffer-only pipeline barriers of the recording thread become one call (HoldBarrier).
volatile std::atomic<uint32_t> kyty_local_barrier_merge_mode {0};
// Pauses the idle worker spins before it blocks (a blocked worker costs the render thread a
// wake-up system call per publish on Windows; Linux the same unless KYTY_LINUX_FAST_LOCKS=0).
#if defined(_WIN32)
volatile std::atomic<uint32_t> kyty_local_recording_spin {4000};
#elif defined(__linux__)
volatile std::atomic<uint32_t> kyty_local_recording_spin {[] {
    const char* value = std::getenv("KYTY_LINUX_FAST_LOCKS");
    return value == nullptr || value[0] != '0' ? 4000u : 0u;
}()};
#else
volatile std::atomic<uint32_t> kyty_local_recording_spin {0};
#endif
}

namespace LocalVulkanRecording {
namespace {
constexpr size_t ChunkBytes = 64 * 1024;
// Deep enough that a burst of recorded calls never blocks the producer in WaitForRoom. 64 (4 MiB) was not:
// Demon's Souls at the Gates spent 5% of the render thread's samples there (Kyty.Record at 41-50%: bursts, not
// a slower consumer). 16 MiB for the one stream.
constexpr size_t ChunkCount = 256;
constexpr uint32_t End = std::numeric_limits<uint32_t>::max();
vk::detail::DispatchLoaderDynamic original;

struct Header {
    void (*execute)(void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    uint32_t payload = 0;
    uint32_t next = End;
};
struct Chunk {
    alignas(64) std::array<std::byte, ChunkBytes> data;
    size_t used = 0;
    uint32_t first = End, last = End, count = 0;
};

// A transaction owns one command plus its parameter graph in a single chunk.
// Failure never publishes any of its partially copied data.
class Writer {
public:
    explicit Writer(Chunk& chunk) : data(chunk.data.data()), used(chunk.used) {}
    bool valid = true, overflow = false;
    std::byte* data;
    size_t used;
    uint32_t command = End;

    void* Allocate(size_t bytes, size_t alignment) {
        if (!valid) return nullptr;
        const size_t offset = (used + alignment - 1) & ~(alignment - 1);
        if (bytes > ChunkBytes || offset > ChunkBytes - bytes) {
            valid = false;
            overflow = true;
            return nullptr;
        }
        used = offset + bytes;
        return data + offset;
    }
    const void* Copy(const void* source, size_t bytes) {
        if (!source || !bytes) return nullptr;
        auto* target = Allocate(bytes, alignof(std::max_align_t));
        if (target) std::memcpy(target, source, bytes);
        return target;
    }
    template<typename T>
    const T* Copy(const T* source, size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!source || !count) return nullptr;
        if (count > ChunkBytes / sizeof(T)) { valid = false; overflow = true; return nullptr; }
        auto* target = static_cast<T*>(Allocate(count * sizeof(T), alignof(T)));
        if (!target) return nullptr;
        std::memcpy(target, source, count * sizeof(T));
        for (size_t i = 0; i < count && valid; ++i) {
            auto& value = target[i];
            if constexpr (requires { value.pNext; }) {
                // Unknown extension chains retain the exact synchronous call.
                if (value.pNext) { valid = false; break; }
            }
            if constexpr (std::is_same_v<T, VkDependencyInfo>) {
                value.pMemoryBarriers = Copy(value.pMemoryBarriers, value.memoryBarrierCount);
                value.pBufferMemoryBarriers = Copy(value.pBufferMemoryBarriers, value.bufferMemoryBarrierCount);
                value.pImageMemoryBarriers = Copy(value.pImageMemoryBarriers, value.imageMemoryBarrierCount);
            } else if constexpr (std::is_same_v<T, VkRenderingInfo>) {
                value.pColorAttachments = Copy(value.pColorAttachments, value.colorAttachmentCount);
                value.pDepthAttachment = Copy(value.pDepthAttachment, 1);
                value.pStencilAttachment = Copy(value.pStencilAttachment, 1);
            } else if constexpr (std::is_same_v<T, VkWriteDescriptorSet>) {
                // Ignored fields need not point to readable memory. Copy only
                // the payload selected by descriptorType, never all pointers.
                value.pImageInfo = nullptr;
                value.pBufferInfo = nullptr;
                value.pTexelBufferView = nullptr;
                switch (value.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                        value.pImageInfo = Copy(source[i].pImageInfo, value.descriptorCount);
                        break;
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        value.pBufferInfo = Copy(source[i].pBufferInfo, value.descriptorCount);
                        break;
                    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                        value.pTexelBufferView = Copy(source[i].pTexelBufferView, value.descriptorCount);
                        break;
                    default: valid = false; break;
                }
            }
        }
        return target;
    }
    template<typename Function>
    void Command(Function function) {
        static_assert(std::is_nothrow_move_constructible_v<Function>);
        static_assert(std::is_nothrow_destructible_v<Function>);
        auto* header = static_cast<Header*>(Allocate(sizeof(Header), alignof(Header)));
        void* payload = Allocate(sizeof(Function), alignof(Function));
        if (!valid) return;
        new (payload) Function(std::move(function));
        new (header) Header {
            .execute = [](void* pointer) { (*static_cast<Function*>(pointer))(); },
            .destroy = nullptr,
            .payload = static_cast<uint32_t>(static_cast<std::byte*>(payload) - data),
        };
        if constexpr (!std::is_trivially_destructible_v<Function>)
            header->destroy = [](void* pointer) { static_cast<Function*>(pointer)->~Function(); };
        command = static_cast<uint32_t>(reinterpret_cast<std::byte*>(header) - data);
    }
};

// A producer waiting for the worker (a full stream, a drain): the slow-frame lines' record_wait (render thread) and
// the live counter record_wait_us.
struct RecordWait {
    LiveCensus::WaitScope                       scope {LiveCensus::WaitRecord};
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    RecordWait() = default;
    RecordWait(const RecordWait&) = delete;
    RecordWait& operator=(const RecordWait&) = delete;
    ~RecordWait() {
        LiveCounters::Add(LiveCounters::RecordWaitUs,
                          static_cast<uint64_t>((std::chrono::steady_clock::now() - start).count()) / 1000u);
    }
};

class Stream;
thread_local Stream* executing = nullptr;
class Stream {
public:
    Stream() : worker([this] { Run(); }) {}
    ~Stream() {
        Drain();
        stopping.store(true, std::memory_order_release);
        // atomic::wait requires a changed value, not just notify, to terminate.
        published.fetch_add(1, std::memory_order_release);
        published.notify_one();
        worker.join();
    }
    alignas(64) std::atomic<uint64_t> state_epoch {0};
    alignas(64) bool was_recording = false;
    void BeforeDirect() {
        Drain();
        const bool active = kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0;
        if (active || was_recording) state_epoch.fetch_add(1, std::memory_order_relaxed);
        was_recording = active;
    }
    template<typename Capture>
    bool Enqueue(Capture capture) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            WaitForRoom();
            auto& chunk = chunks[sequence % ChunkCount];
            Writer writer(chunk);
            capture(writer);
            if (writer.valid && writer.command != End) {
                if (chunk.last != End)
                    reinterpret_cast<Header*>(chunk.data.data() + chunk.last)->next = writer.command;
                else chunk.first = writer.command;
                chunk.last = writer.command;
                chunk.used = writer.used;
                ++chunk.count;
                if (chunk.count >= 128 || chunk.used >= ChunkBytes * 3 / 4) Publish();
                return true;
            }
            if (!writer.overflow || chunk.count == 0) break;
            Publish();
        }
        return false;
    }
    void Flush() { Publish(); }
    // Packets recorded or published and not yet replayed (a direct call would wait for them).
    [[nodiscard]] bool Pending() const {
        return chunks[sequence % ChunkCount].count != 0 || completed.load(std::memory_order_acquire) != sequence;
    }
    void Drain() {
        Publish();
        if (completed.load(std::memory_order_acquire) == sequence) return;
        const RecordWait waiting;
#if defined(__linux__)
        static const bool fast_locks = [] {
            const char* value = std::getenv("KYTY_LINUX_FAST_LOCKS");
            return value == nullptr || value[0] != '0';
        }();
        // Linux: a short spin first, the same as the worker's (KYTY_LINUX_FAST_LOCKS=0: waits at once).
        if (fast_locks) {
            for (uint32_t spin = 0; spin < 4000; ++spin) {
                if (completed.load(std::memory_order_acquire) == sequence) return;
                _mm_pause();
            }
        }
#endif
        for (auto done = completed.load(std::memory_order_acquire); done != sequence;
             done = completed.load(std::memory_order_acquire)) completed.wait(done, std::memory_order_acquire);
    }
private:
    // Producer and consumer only touch a slot while they own it. Release /
    // acquire counters also publish resets before the producer reuses a slot.
    std::array<Chunk, ChunkCount> chunks;
    uint64_t sequence = 0;
    alignas(64) std::atomic<uint64_t> published {0};
    alignas(64) std::atomic<uint64_t> completed {0};
    alignas(64) std::atomic<bool> sleeping {false};
    std::atomic<bool> stopping {false};
    std::thread worker;

    void WaitForRoom() {
        auto done = completed.load(std::memory_order_acquire);
        if (sequence - done < ChunkCount) return;
        const RecordWait waiting;
        for (; sequence - done >= ChunkCount; done = completed.load(std::memory_order_acquire))
            completed.wait(done, std::memory_order_acquire);
    }
    void Publish() {
        WaitForRoom();
        auto& chunk = chunks[sequence % ChunkCount];
        if (chunk.count == 0) return;
        published.store(++sequence, std::memory_order_seq_cst);
        // Wake the worker only when it said it blocks (both sides seq_cst): a wake-up is a
        // system call on Windows, and the worker is usually busy or still spinning.
        if (sleeping.load(std::memory_order_seq_cst)) published.notify_one();
    }
    // The replay thread must not land on an efficiency core: every GPU wait includes its
    // latency.  KYTY_RECORDING_CPUS (e.g. "1,2,3,6,7") restricts it; re-applied now and
    // then because the launcher assigns one mask to every game thread at startup.
    static void KeepAffinity() {
        static const char* const list = std::getenv("KYTY_RECORDING_CPUS");
        if (list) LocalPlatform::PinThreadToCpuList(list);
    }
    void Run() {
        executing = this;
        LocalPlatform::SetThreadName("Kyty.Record");
        KeepAffinity();
        uint64_t current = 0;
        for (;;) {
            if ((current & 4095u) == 0) KeepAffinity();
            for (auto ready = published.load(std::memory_order_acquire); current == ready;
                 ready = published.load(std::memory_order_acquire)) {
                const uint32_t spins = SpinGate::Open(true) ? kyty_local_recording_spin.load(std::memory_order_relaxed) : 0;
                for (uint32_t spin = 0; spin < spins && ready == current; ++spin) {
                    _mm_pause();
                    ready = published.load(std::memory_order_acquire);
                }
                if (ready != current) break;
                LiveCounters::Add(LiveCounters::RecordSleeps);
                sleeping.store(true, std::memory_order_seq_cst);
                if (published.load(std::memory_order_seq_cst) == current)
                    published.wait(current, std::memory_order_seq_cst);
                sleeping.store(false, std::memory_order_relaxed);
            }
            if (stopping.load(std::memory_order_acquire)) return;
            auto& chunk = chunks[current % ChunkCount];
            for (auto offset = chunk.first; offset != End;) {
                const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + offset);
                header->execute(chunk.data.data() + header->payload);
                if (header->destroy) header->destroy(chunk.data.data() + header->payload);
                offset = header->next;
            }
            chunk.used = 0;
            chunk.count = 0;
            chunk.first = chunk.last = End;
            completed.store(++current, std::memory_order_release);
            completed.notify_one();
        }
    }
};

thread_local std::unique_ptr<Stream> producer;
Stream* RecordingStream() {
    if (producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0) {
        producer->was_recording = true;
        return producer.get();
    }
    return nullptr;
}
void InvalidateRawState() {
    if (executing) executing->state_epoch.fetch_add(1, std::memory_order_relaxed);
}
[[gnu::noinline]] void BeforeDirect() {
    if (producer) {
        // Diagnostic (live timecensus, source DirectDrain): direct calls made while packets are pending.
        if (producer->Pending()) KYTY_TIME_CENSUS(DirectDrain, 0);
        producer->BeforeDirect();
    }
}

// Work calls recorded by this thread (read by the render thread's barrier dedupe, which orders
// only its own command stream): a plain increment, not a locked add on every command.
thread_local uint64_t g_work_calls = 0;
thread_local uint64_t g_packets    = 0;

// KYTY_BARRIER_MERGE: consecutive vkCmdPipelineBarrier calls of the recording thread on one command buffer, with
// nothing recorded between them and only memory and buffer barriers, become one call: the stage masks ORed, the
// dependency flags ANDed, the barriers together. It orders and makes visible all that each of them did (never
// less), and the GPU waits and flushes its caches once instead of once a call (Demon's Souls at the Gates: ~2,000
// back-to-back pairs a frame, a dispatch's access barrier then the next one's hazard barrier, and the barrier pairs
// of uploads and copies). Image barriers (layout transitions), queue-family transfers and extension chains are
// recorded as they come. A held call is recorded before any other call this thread makes through the dispatcher
// (every generated wrapper flushes it first) or the packet paths below.
struct HeldBarrier {
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkPipelineStageFlags source = 0, destination = 0;
    VkDependencyFlags flags = 0;
    std::vector<VkMemoryBarrier> memory;
    std::vector<VkBufferMemoryBarrier> buffers;
    bool held = false;
};
// KYTY_BARRIER_MERGE bit 1 (=3 with the above): the same for vkCmdPipelineBarrier2, whose barriers carry their own
// stages. Calls merged become one with every barrier's stages and accesses widened to the union of all of them: each
// then orders and makes visible what the calls did, a dependency chained through the first call included. Image
// barriers keep their layouts; a call with an image the held calls transition is recorded after them, not merged.
struct HeldBarrier2 {
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkDependencyFlags flags = 0;
    std::vector<VkMemoryBarrier2> memory;
    std::vector<VkBufferMemoryBarrier2> buffers;
    std::vector<VkImageMemoryBarrier2> images;
    uint32_t calls = 0;
};
thread_local HeldBarrier g_held;
thread_local HeldBarrier2 g_held2;
// A merged call stays well inside a recording chunk.
constexpr size_t HeldBarrierLimit = 256;
void FlushHeldBarrier();
bool HoldBarrier(VkCommandBuffer command, VkPipelineStageFlags source, VkPipelineStageFlags destination,
                 VkDependencyFlags flags, uint32_t memory_count, const VkMemoryBarrier* memory,
                 uint32_t buffer_count, const VkBufferMemoryBarrier* buffers, uint32_t image_count,
                 const VkImageMemoryBarrier* /*images*/) {
    if ((kyty_local_barrier_merge_mode.load(std::memory_order_relaxed) & 1u) == 0 || image_count != 0 ||
        (flags & ~static_cast<VkDependencyFlags>(VK_DEPENDENCY_BY_REGION_BIT)) != 0 || (memory_count != 0 && !memory) ||
        (buffer_count != 0 && !buffers) || RecordingStream() == nullptr)
        return false;
    for (uint32_t i = 0; i < memory_count; ++i)
        if (memory[i].pNext) return false;
    for (uint32_t i = 0; i < buffer_count; ++i)
        if (buffers[i].pNext || buffers[i].srcQueueFamilyIndex != buffers[i].dstQueueFamilyIndex) return false;
    if (g_held2.calls != 0) FlushHeldBarrier();
    auto& held = g_held;
    if (held.held && (held.command != command || held.memory.size() + memory_count > HeldBarrierLimit ||
                      held.buffers.size() + buffer_count > HeldBarrierLimit))
        FlushHeldBarrier();
    if (!held.held) {
        held.held        = true;
        held.command     = command;
        held.source      = 0;
        held.destination = 0;
        held.flags       = flags;
        held.memory.clear();
        held.buffers.clear();
    }
    held.source |= source;
    held.destination |= destination;
    held.flags &= flags;
    held.memory.insert(held.memory.end(), memory, memory + memory_count);
    held.buffers.insert(held.buffers.end(), buffers, buffers + buffer_count);
    return true;
}
bool HoldBarrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
    if ((kyty_local_barrier_merge_mode.load(std::memory_order_relaxed) & 2u) == 0 || info == nullptr || info->pNext ||
        (info->dependencyFlags & ~static_cast<VkDependencyFlags>(VK_DEPENDENCY_BY_REGION_BIT)) != 0 ||
        (info->memoryBarrierCount != 0 && !info->pMemoryBarriers) ||
        (info->bufferMemoryBarrierCount != 0 && !info->pBufferMemoryBarriers) ||
        (info->imageMemoryBarrierCount != 0 && !info->pImageMemoryBarriers) || RecordingStream() == nullptr)
        return false;
    for (uint32_t i = 0; i < info->memoryBarrierCount; ++i)
        if (info->pMemoryBarriers[i].pNext) return false;
    for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
        const auto& barrier = info->pBufferMemoryBarriers[i];
        if (barrier.pNext || barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex) return false;
    }
    for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
        const auto& barrier = info->pImageMemoryBarriers[i];
        if (barrier.pNext || barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex) return false;
    }
    if (g_held.held) FlushHeldBarrier();
    auto& held     = g_held2;
    bool  separate = held.calls != 0 &&
                    (held.command != command || held.memory.size() + info->memoryBarrierCount > HeldBarrierLimit ||
                     held.buffers.size() + info->bufferMemoryBarrierCount > HeldBarrierLimit ||
                     held.images.size() + info->imageMemoryBarrierCount > HeldBarrierLimit);
    for (uint32_t i = 0; i < info->imageMemoryBarrierCount && !separate; ++i)
        for (const auto& image: held.images)
            if (image.image == info->pImageMemoryBarriers[i].image) {
                separate = true;
                break;
            }
    if (separate) FlushHeldBarrier();
    if (held.calls == 0) {
        held.command = command;
        held.flags   = info->dependencyFlags;
        held.memory.clear();
        held.buffers.clear();
        held.images.clear();
    }
    held.flags &= info->dependencyFlags;
    held.memory.insert(held.memory.end(), info->pMemoryBarriers, info->pMemoryBarriers + info->memoryBarrierCount);
    held.buffers.insert(held.buffers.end(), info->pBufferMemoryBarriers,
                        info->pBufferMemoryBarriers + info->bufferMemoryBarrierCount);
    held.images.insert(held.images.end(), info->pImageMemoryBarriers,
                       info->pImageMemoryBarriers + info->imageMemoryBarrierCount);
    ++held.calls;
    return true;
}
#include "local-vulkan-recording.inc"
void FlushHeldBarrier() {
    if (auto& held = g_held; held.held) {
        held.held = false;
        Record_vkCmdPipelineBarrier(held.command, held.source, held.destination, held.flags,
                                    static_cast<uint32_t>(held.memory.size()), held.memory.data(),
                                    static_cast<uint32_t>(held.buffers.size()), held.buffers.data(), 0, nullptr);
    }
    if (auto& held = g_held2; held.calls != 0) {
        if (held.calls > 1) {
            VkPipelineStageFlags2 source = 0, destination = 0;
            VkAccessFlags2        source_access = 0, destination_access = 0;
            const auto gather = [&](const auto& list) {
                for (const auto& barrier: list) {
                    source |= barrier.srcStageMask;
                    destination |= barrier.dstStageMask;
                    source_access |= barrier.srcAccessMask;
                    destination_access |= barrier.dstAccessMask;
                }
            };
            const auto widen = [&](auto& list) {
                for (auto& barrier: list) {
                    barrier.srcStageMask  = source;
                    barrier.dstStageMask  = destination;
                    barrier.srcAccessMask = source_access;
                    barrier.dstAccessMask = destination_access;
                }
            };
            gather(held.memory);
            gather(held.buffers);
            gather(held.images);
            widen(held.memory);
            widen(held.buffers);
            widen(held.images);
        }
        held.calls = 0;
        VkDependencyInfo info {};
        info.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        info.dependencyFlags          = held.flags;
        info.memoryBarrierCount       = static_cast<uint32_t>(held.memory.size());
        info.pMemoryBarriers          = held.memory.data();
        info.bufferMemoryBarrierCount = static_cast<uint32_t>(held.buffers.size());
        info.pBufferMemoryBarriers    = held.buffers.data();
        info.imageMemoryBarrierCount  = static_cast<uint32_t>(held.images.size());
        info.pImageMemoryBarriers     = held.images.data();
        Record_vkCmdPipelineBarrier2(held.command, &info);
    }
}
} // namespace

void Install() { InstallDispatch(); }
const vk::detail::DispatchLoaderDynamic& DirectDispatch() { return original; }
void* ReplaceBindPipeline(void* entry) {
    auto* replaced = reinterpret_cast<void*>(original.vkCmdBindPipeline);
    original.vkCmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(entry);
    return replaced;
}
uint64_t WorkCalls() { return g_work_calls; }
uint64_t RecordedWork() { return g_work_calls + g_packets; }
void Drain() {
    FlushHeldBarrier();
    if (producer) producer->Drain();
}
bool PacketsEnabled() {
    return producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0;
}
uint64_t StateEpoch() {
    return executing ? executing->state_epoch.load(std::memory_order_relaxed) : UINT64_MAX;
}
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner) {
    FlushHeldBarrier();
    if (!PacketsEnabled() || !replay || segments.size() > 8) return false;
    for (const auto& segment : segments)
        if (segment.size && !segment.data) return false;
    ++g_packets;
    auto* stream = RecordingStream();
    return stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        writer.Command([owner, copied, count, replay] { replay({copied.data(), count}, original); });
    });
}
bool DeferredSubmitEnabled() {
    return producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0 &&
        kyty_local_deferred_submit_mode.load(std::memory_order_relaxed) != 0;
}
bool EnqueueDeferred(ReplayPacket replay, std::span<const Segment> segments, bool publish) {
    FlushHeldBarrier();
    if (!DeferredSubmitEnabled() || !replay || segments.size() > 8) return false;
    auto* stream = RecordingStream();
    if (!stream) return false;
    const bool queued = stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        // A new command buffer may reuse a handle: drop raw-state reuse as a direct call would.
        writer.Command([copied, count, replay] { InvalidateRawState(); replay({copied.data(), count}, original); });
    });
    if (queued && publish) stream->Flush();
    return queued;
}
namespace {
std::atomic<uint64_t> g_deferred_queued {0};
std::atomic<uint64_t> g_deferred_done {0};
} // namespace
void NoteDeferredSubmitQueued() { g_deferred_queued.fetch_add(1, std::memory_order_release); }
uint64_t DeferredSubmitsQueued() { return g_deferred_queued.load(std::memory_order_acquire); }
uint64_t DeferredSubmitsDone() { return g_deferred_done.load(std::memory_order_acquire); }
void NoteDeferredSubmitDone() {
    g_deferred_done.fetch_add(1, std::memory_order_release);
    g_deferred_done.notify_all();
}
void WaitDeferredSubmits() {
    const auto target = g_deferred_queued.load(std::memory_order_acquire);
    for (auto done = g_deferred_done.load(std::memory_order_acquire); done < target;
         done = g_deferred_done.load(std::memory_order_acquire)) g_deferred_done.wait(done, std::memory_order_acquire);
}
void ReplayInline(ReplayPacket replay, std::span<const Segment> segments) {
    FlushHeldBarrier();
    if (producer) producer->BeforeDirect();
    replay(segments, original);
}
void InvalidateReplayState() { InvalidateRawState(); }
ProducerScope::ProducerScope() {
    if (producer) std::abort();
    producer = std::make_unique<Stream>();
}
ProducerScope::~ProducerScope() {
    FlushHeldBarrier();
    producer.reset();
}
} // namespace LocalVulkanRecording
