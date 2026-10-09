# Scheduler test device setup

The Vulkan test harness requires its normal compute and rasterization features.
It reports the selected physical device and lists any missing required Vulkan
extensions or features before failing initialization.

Attachment feedback loop layout and dynamic state are optional for the harness.
When the selected device does not expose both extensions and both feature bits,
the context reports feedback as unavailable and only the depth attachment
feedback rasterization case is printed as `SKIP`. Other tests still run against
the device, and missing required features remain failures.

When `KYTY_LOCAL_VULKAN_RECORDING` is enabled, the harness defaults
`kyty_local_vulkan_recording_mode` to `4`, preserving the serialized recording
path used by existing tests. Set `KYTY_TEST_VULKAN_RECORDING=0` to leave the
recording mode untouched. This lets scheduler tests exercise the direct async
submit worker with `KYTY_ASYNC_SUBMIT=1`:

```powershell
$env:KYTY_ASYNC_SUBMIT = '1'
$env:KYTY_TEST_VULKAN_RECORDING = '0'
& '<path-to-test-binary>' --scheduler-only
```

Remove either environment variable after the run if the same shell is used for
other tests. Omitting `KYTY_TEST_VULKAN_RECORDING` keeps the default recording
mode `4`.
