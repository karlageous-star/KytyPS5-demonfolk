# Native byte-exact buffer regression tests

Run the focused native descriptor regressions with `KYTY_BUFFER_BYTE_EXACT=1` set in the test process:

```powershell
$env:KYTY_BUFFER_BYTE_EXACT = "1"
& .\path\to\ShaderRecompilerComputeTests.exe --byte-exact-only
```

The selector refuses to start unless the variable is exactly `1`. Set it before launching the process because the emitter reads the switch once. The existing `--shader-data-storage-only` selector remains unchanged.

The focused cases use native descriptors whose base addresses have byte remainders 1, 2, and 3. Each case checks a raw dword load and store at offset zero, dword access at an effectively aligned address within the misaligned base, signed and unsigned 16-bit loads crossing effective byte address 3, and a crossing 16-bit store. Read values are written to separate aligned locations for comparison. The expected backing-buffer contents also verify that bytes adjacent to the short store retain their original values.

These cases cover the raw native buffer path exercised by the fixtures. They make no claim about table-mode addressing, runtime format decoding, or unrelated memory paths.
