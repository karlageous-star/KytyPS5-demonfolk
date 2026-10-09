# Precompile shaders in the local Demon's Souls player

Close the game in this player, then double-click `precompile.cmd` beside `launcher.exe`.
The remembered game folder is selected automatically; otherwise a folder picker asks for
the folder containing `eboot.bin` and `sce_sys/param.json`.

The tool extracts shader bundles and embedded shaders from your own game files, compiles
their derived pipeline variants on your GPU, includes recorded play specializations, and
merges the result into this player's `_PipelineCache/static`. No Python installation or
downloaded shader list is required. The current runtime still uses title-only static cache
names; the wrapper preserves the title's previous static cache and warmup recordings if its game version changes.
The seed files are named by title and version in `_Precompile`.

First compilation can take hours. Progress appears in the console and detailed output is
in `Logs/<timestamp>-precompile`. Closing the console stops only its precompile workers.
Rerunning extracts fresh GPU-specific inputs and resumes pipelines already saved by completed
shards or periodic checkpoints. Work since a worker's last checkpoint can be repeated.
Launch `launcher.exe` in the same folder afterward. Recompile after changing the graphics
driver, GPU, game version or shader compiler.

Default: two worker processes, three translation threads per worker. For another setting:

```powershell
.\precompile.cmd -Jobs 3 -Threads 3
```

## Implementation and limits

The standalone seed reader is adapted from chenxiao's `static-seeds.cpp` at `a3f5141c`.
The shader compiler and pipeline emitter remain those of our `2f947473` player. The tool
detects compute wave64 and fragment subgroup reduction from its Vulkan device instead of
hardcoding the reference generator's NVIDIA inputs. It keeps our existing cache file layout;
the newer release's versioned-cache backend and table/speculation architecture are not imported.
The wrapper applies the same title environment as `launcher.exe`, including its driver-cache
key, and converts local warmup3/warmup4 recordings to compatible seeds without changing
their checksum or specialization payload. Static input freshness accepts both supported
warmup schemas; the previous status check accepted only warmup3 although the writer uses 4.

This is an offline coverage tool, not a guarantee that all runtime specializations are covered.
The reference render-pass palette and extractor were developed for EU versions; deriving
inputs from US v1.04 does not establish complete coverage. Parsing failures and compiler
rejections are logged. This packaging work is built only: no tests, precompile run or game
launch were performed at the user's request. No FPS improvement is inferred from compilation.

Rollback: remove the added precompile executable, scripts, `game-path.txt`, `tools/local/static-precompile`,
and `_Precompile`. If a completed precompile causes trouble, move `_PipelineCache/static` aside.
For the current game version, recorded `_PipelineCache/warmup-v2`, regular driver caches and saves
remain independent. Recordings moved after a game-version change are retained in `_Precompile/Previous-cache-*`.
