# Wide color-clear register decoding

For DCC key `0x20`, the color-buffer clear is held in `CB_COLORn_CLEAR_WORD0` and, for wide
texels, `CB_COLORn_CLEAR_WORD1`. The render-target description now carries both words, and the
known-fill `PrepareDccClear` path can decode RGBA16F register clears such as Astro Bot's galaxy-map
target.

Wide decoding is disabled by default. Set `KYTY_CLEAR_REGISTER_WIDE=1` before starting Kyty to
enable it during validation. With the setting absent or zero, the existing 32-bit decoder remains
in use. The standalone CPU tests cover both opt-in states, both register words, wide float/integer
formats, metadata equality, and unchanged 32-bit channel swizzles.

The port keeps DCC clear preparation limited to known, uniform fill metadata already recorded by the
texture cache. An unknown or pending metadata address is not interpreted as a clear, and this change
does not add CPU metadata readback or a materialization path. `ImageMetadataInfo` uses defaulted
equality, so the appended word participates in metadata comparisons. No warmup or native-cache key
serializes this runtime metadata structure; the Vulkan pipeline cache is separate.

Astro Bot visual and array A/B validation remain pending.
