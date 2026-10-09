# Native misaligned buffer access

Adapts reference `fea89bb6` to the current runtime byte-offset specialization. `KYTY_BUFFER_BYTE_EXACT=1` (default off) handles native descriptor buffers marked `byte_base_offset`: dword loads splice adjacent host dwords; a short crossing a dword boundary reads both; corresponding stores merge masked bytes atomically into each touched host dword. Neighboring bytes retain their prior values. Each touched dword preserves the existing bounds behavior. Aligned buffer specializations retain their existing emission.

The current `ByteAddress` already adds the binding's low byte remainder and `EmitMemoryElementIndex` adds its whole-dword displacement. The port does not add the remainder a second time. Shift counts remain below 32, including effective aligned addresses within a misaligned buffer.

This switch does not change table-mode physical addresses, runtime format decoding, scalar-buffer access, LDS, scratch or atomics. Those paths require separate tests and are not claimed fixed. It is a bounded port for the native specialized descriptor path used in the comparison. No FPS or device-loss fix is claimed before game tests. Use a separate run cache for each shader-emission treatment.

Revert this commit or leave the switch unset to restore prior emission. Qualification needs all base remainders 1/2/3, dword and crossing short loads/stores, signed short loads, aligned effective addresses and neighboring-byte preservation, followed by ASTRO and DS runs.
