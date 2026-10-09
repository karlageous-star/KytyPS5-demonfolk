# Synchronous download-ring write ordering

## Change

In `BufferCache::DownloadBufferMemory`, the synchronous graphics-queue fallback now passes
`vk::AccessFlagBits::eTransferWrite` as `Buffer::CopyFrom`'s `destination_before` access mask.
`Buffer::CopyFrom` uses that mask for the destination buffer's pre-copy barrier over the copied
destination range, with the transfer stage as the consumer. This expresses the required
transfer-write → transfer-write dependency when a download copy reuses a ring region that an
earlier download copy wrote.

`StreamBuffer::Map` waits for prior tick use before a region is reused. That host-side wait may
mean the earlier copy has completed in practice, but it does not state the device-side memory
dependency that Vulkan synchronization validation expects. The destination pre-barrier makes
that dependency explicit. The source pre-mask, both post-copy masks, ring reservation/commit,
and completion handling are unchanged.

The change is limited to the synchronous `download.CopyFrom` fallback. The readback-queue fast
path and the separate verification copy retain their existing behavior. Historical commit
`df14c37e7c3b1767e91873cd8996f9d339289dbe` added the equivalent range barrier; reference
`f9464f82f27b03d6c5bd8cb894eaf1ca3a43d35` contains it. The current implementation uses the
existing `Buffer::CopyFrom` API to express the destination dependency.

## Source-focused regression check

Inspect the synchronous fallback call in `bufferCache.cpp` and verify:

1. `destination_before` is `vk::AccessFlagBits::eTransferWrite`.
2. The call still targets `download` at `base_offset + cursor` for `envelope_size`.
3. The fast path and verification copy are unchanged.

No build or runtime test was run for this change. This restores a concrete synchronization
dependency; it does not establish that this path caused, or fixes, any observed crash or device
loss.
