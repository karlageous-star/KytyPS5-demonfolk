# Portable precompile for the local Demon's Souls player. Uses its own backend and cache layout.
param(
    [string]$Game = '',
    [ValidateRange(1, 64)][int]$Jobs = 2,
    [ValidateRange(1, 64)][int]$Threads = 3,
    [ValidateRange(1, 65536)][int]$Shards = 512
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($PSScriptRoot)
$exe = Join-Path $root 'kyty_shader_precompile.exe'
$work = Join-Path $root '_Precompile'
$running = @{}
$runLock = $null
$job = $null
$savedEnvironment = @{}
Set-Location -LiteralPath $root

# A Windows job owns only this script's children. Closing this console closes the job handle
# and stops those workers, including when PowerShell cannot execute its finally block.
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public sealed class KytyPrecompileJob : IDisposable {
    [StructLayout(LayoutKind.Sequential)] struct BasicLimit {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinWorkingSet, MaxWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct IoCounters {
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimit {
        public BasicLimit Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    IntPtr handle;
    public KytyPrecompileJob() {
        handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero) throw new Win32Exception();
        var limits = new ExtendedLimit();
        limits.Basic.Flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        int size = Marshal.SizeOf(typeof(ExtendedLimit));
        IntPtr data = Marshal.AllocHGlobal(size);
        try {
            Marshal.StructureToPtr(limits, data, false);
            if (!SetInformationJobObject(handle, 9, data, (uint)size)) {
                int error = Marshal.GetLastWin32Error();
                CloseHandle(handle); handle = IntPtr.Zero;
                throw new Win32Exception(error);
            }
        } finally { Marshal.FreeHGlobal(data); }
    }
    public void Add(IntPtr process) {
        if (!AssignProcessToJobObject(handle, process)) throw new Win32Exception();
    }
    public void Dispose() {
        if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; }
        GC.SuppressFinalize(this);
    }
    ~KytyPrecompileJob() { Dispose(); }
}
'@

function Write-Utf8([string]$path, [string]$value) {
    [IO.File]::WriteAllText($path, $value, (New-Object Text.UTF8Encoding($false)))
}
function Start-Worker([string]$name, [string[]]$arguments) {
    # Start-Process joins arguments into a command line; quote each path and each whole argument.
    # Windows filenames cannot contain quotes. No argument is interpreted as shell code.
    $quoted = foreach ($argument in (@('--game', $Game) + $arguments)) {
        if ($argument.Contains('"') -or $argument.EndsWith('\')) { throw 'Unsupported quoted/trailing-slash argument' }
        '"' + $argument + '"'
    }
    $process = Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory $root `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs "$name.out.log") `
        -RedirectStandardError (Join-Path $logs "$name.err.log")
    try {
        $null = $process.Handle
        if (!$process.HasExited) {
            $job.Add($process.Handle)
            if (!$process.HasExited) { $process.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal }
        }
    } catch {
        if ($process.HasExited) { return $process }
        if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
        throw
    }
    return $process
}
function Wait-Worker($process, [string]$name) {
    $nextMessage = (Get-Date).AddSeconds(30)
    while (!$process.WaitForExit(500)) {
        if ((Get-Date) -ge $nextMessage) {
            Write-Host "$name still working; details: $logs"
            $nextMessage = (Get-Date).AddSeconds(30)
        }
    }
    if ($process.ExitCode -ne 0) { throw "$name failed ($($process.ExitCode)); see $logs" }
    $process.Dispose()
}
function Export-RecordedSeeds([string]$recording, [string]$output) {
    $bytes = [IO.File]::ReadAllBytes($recording)
    $newline = [Array]::IndexOf($bytes, [byte]10)
    if ($newline -lt 0 -or $newline -gt 1024) { throw 'Invalid recorded shader-cache header' }
    $identity = [Text.Encoding]::UTF8.GetString($bytes, 0, $newline + 1)
    $schema = if ($identity.StartsWith("KytyShaderWarmup4:${title}")) { 3 } `
        elseif ($identity.StartsWith("KytyShaderWarmup3:${title}")) { 2 } `
        else { throw 'Unexpected recorded shader-cache identity' }
    $header = [Text.Encoding]::UTF8.GetBytes("KytyShaderSeeds${schema}:${title}`n")
    $stream = [IO.File]::Create($output)
    try {
        $stream.Write($header, 0, $header.Length)
        # Preserve the checksum, records, pipeline recipes and their native-binding flags.
        $stream.Write($bytes, $newline + 1, $bytes.Length - $newline - 1)
    } finally { $stream.Dispose() }
}

try {
    if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing $exe" }
    if (!$Game -and (Test-Path -LiteralPath (Join-Path $root 'game-path.txt'))) {
        $Game = (Get-Content -LiteralPath (Join-Path $root 'game-path.txt') -Raw).Trim()
    }
    if (!$Game -or !(Test-Path -LiteralPath (Join-Path $Game 'sce_sys\param.json'))) {
        Add-Type -AssemblyName System.Windows.Forms
        $picker = New-Object Windows.Forms.FolderBrowserDialog
        $picker.Description = "Select Demon's Souls folder containing eboot.bin and sce_sys"
        $picker.ShowNewFolderButton = $false
        try {
            if ($picker.ShowDialog() -ne [Windows.Forms.DialogResult]::OK) { throw 'No game selected' }
            $Game = $picker.SelectedPath
        } finally { $picker.Dispose() }
    }
    $Game = [IO.Path]::GetFullPath($Game).TrimEnd('\', '/')
    if (!(Test-Path -LiteralPath (Join-Path $Game 'eboot.bin') -PathType Leaf)) { throw 'Selected folder has no eboot.bin' }
    $info = Get-Content -LiteralPath (Join-Path $Game 'sce_sys\param.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $title = [string]$info.titleId
    $version = [string]$info.contentVersion
    if ($title -notin @('PPSA01341', 'PPSA01342') -or $version -notmatch '^\d{2}\.\d{3}\.\d{3}$') {
        throw 'This precompile package supports Demons Souls PPSA01341/PPSA01342 game folders only'
    }
    foreach ($emulator in @(Get-Process -Name kyty_emulator -ErrorAction SilentlyContinue)) {
        if ($emulator.Path -and [IO.Path]::GetFullPath($emulator.Path) -eq (Join-Path $root 'kyty_emulator.exe')) {
            throw 'Close this player before precompiling so the same cache is not written by both processes'
        }
    }
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $runLock = [IO.File]::Open((Join-Path $work 'run.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $logs = Join-Path $root "Logs\$stamp-precompile"
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $job = New-Object KytyPrecompileJob
    $profilePath = Join-Path $root 'launcher-environment.json'
    if (Test-Path -LiteralPath $profilePath) {
        $profiles = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
        $profile = $profiles.titles.PSObject.Properties[$title]
        if ($profile) {
            foreach ($setting in $profile.Value.PSObject.Properties) {
                if ($setting.Name -notmatch '^KYTY_[A-Z0-9_]+$' -or $setting.Value -isnot [string]) {
                    throw 'Invalid launcher environment setting'
                }
                $savedEnvironment[$setting.Name] = [Environment]::GetEnvironmentVariable($setting.Name, 'Process')
                [Environment]::SetEnvironmentVariable($setting.Name, $setting.Value, 'Process')
            }
        }
    }
    Write-Utf8 (Join-Path $root 'game-path.txt') $Game
    $cache = Join-Path $root '_PipelineCache\static'
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    # This emulator names static caches by title. Preserve that title's previous version when
    # the remembered game changes, rather than silently mixing pipeline inputs from two versions.
    $cacheOwnerPath = Join-Path $work "cache-owner-$title.json"
    $owner = $null
    if (Test-Path -LiteralPath $cacheOwnerPath) {
        $owner = Get-Content -LiteralPath $cacheOwnerPath -Raw | ConvertFrom-Json
        if ($owner.version -ne $version) {
            $backup = Join-Path $work "Previous-cache-$title-$stamp"
            New-Item -ItemType Directory -Force -Path $backup | Out-Null
            foreach ($file in @(Get-ChildItem -LiteralPath $cache -File | Where-Object { $_.Name.StartsWith("$title.") })) {
                Move-Item -LiteralPath $file.FullName -Destination (Join-Path $backup $file.Name)
            }
            $recordingRoot = Join-Path $root '_PipelineCache\warmup-v2'
            foreach ($file in @(Get-ChildItem -LiteralPath $recordingRoot -Filter "$title.shaders" -Recurse -File -ErrorAction SilentlyContinue)) {
                $savedRecording = Join-Path $backup $file.Directory.Name
                New-Item -ItemType Directory -Force -Path $savedRecording | Out-Null
                Move-Item -LiteralPath $file.FullName -Destination (Join-Path $savedRecording $file.Name)
            }
        }
    }
    Write-Utf8 $cacheOwnerPath (@{title=$title; version=$version; game=$Game} | ConvertTo-Json)
    $seeds = Join-Path $work "seeds-$title-$version.seeds"
    $states = Join-Path $root 'tools\local\static-precompile\pass-states.json'
    $begin = Get-Date
    Write-Host "Precompiling $title v$version on this PC; $Jobs workers, $Threads threads each."
    Write-Host 'First run can take hours. Close this window to stop; run it again to resume saved work.'
    Write-Host "Logs: $logs"
    # Regenerate from this game and this GPU on every invocation. Compiled shards/cache still resume.
    # This avoids a seed made for a 32-lane host being reused after changing to a wave64 host.
    Wait-Worker (Start-Worker 'make-seeds' @('--make-seeds', $seeds, '--states', $states)) 'Extracting shader inputs'
    Wait-Worker (Start-Worker 'merge-before' @('--merge')) 'Merging previous work'
    $pending = New-Object 'System.Collections.Generic.Queue[object]'
    for ($i = 0; $i -lt $Shards; $i++) { $pending.Enqueue(@($seeds, "$i/$Shards")) }
    $recordingRoot = Join-Path $root '_PipelineCache\warmup-v2'
    $recording = Get-ChildItem -LiteralPath $recordingRoot -Filter "$title.shaders" -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($recording) {
        $recorded = Join-Path $work "recorded-$title-$version.seeds"
        Export-RecordedSeeds $recording.FullName $recorded
        $recordedShards = [Math]::Max(1, [Math]::Floor($Shards / 16))
        for ($i = 0; $i -lt $recordedShards; $i++) { $pending.Enqueue(@($recorded, "$i/$recordedShards")) }
        Write-Host 'Including specializations learned during previous play.'
    }
    $finished = 0
    $shown = Get-Date
    while ($pending.Count -gt 0 -or $running.Count -gt 0) {
        while ($pending.Count -gt 0 -and $running.Count -lt $Jobs) {
            $seedFile, $shard = $pending.Dequeue()
            $name = 'shard-' + ($shard -replace '/', 'of') + '-' + [IO.Path]::GetFileNameWithoutExtension($seedFile)
            $running[$name] = @($seedFile, $shard, (Start-Worker $name @('--seeds', $seedFile, '--shard', $shard, '--threads', "$Threads")))
        }
        Start-Sleep -Milliseconds 250
        foreach ($name in @($running.Keys)) {
            $seedFile, $shard, $process = $running[$name]
            if (!$process.HasExited) { continue }
            $running.Remove($name)
            $exitCode = $process.ExitCode
            $process.Dispose()
            if ($exitCode -eq 3) {
                $index, $count = [int[]]($shard -split '/')
                if ($count -ge 65536) { throw "$name cannot be split further; see $logs" }
                $pending.Enqueue(@($seedFile, "$index/$(2 * $count)"))
                $pending.Enqueue(@($seedFile, "$($index + $count)/$(2 * $count)"))
            } elseif ($exitCode -ne 0) {
                throw "$name failed ($exitCode); see $logs"
            } else { $finished++ }
        }
        if (((Get-Date) - $shown).TotalSeconds -ge 30) {
            $shown = Get-Date
            Write-Host ("{0:hh\:mm\:ss}: {1} shards complete, {2} running, {3} waiting" -f ((Get-Date) - $begin), $finished, $running.Count, $pending.Count)
        }
    }
    Wait-Worker (Start-Worker 'merge-final' @('--merge')) 'Merging compiled pipelines'
    Wait-Worker (Start-Worker 'static-inputs' @('--seeds', $seeds, '--no-pipelines', '--static-inputs', '--threads', "$Threads")) 'Saving prefetch inputs'
    Write-Host "Finished. Cache: $cache"
    Write-Host 'Launch launcher.exe from this folder. Unseen specializations can still compile during play.'
} catch {
    Write-Host "Precompile error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($job) { $job.Dispose() }
    foreach ($entry in $running.Values) { $entry[2].Dispose() }
    foreach ($setting in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($setting, $savedEnvironment[$setting], 'Process')
    }
    if ($runLock) { $runLock.Dispose() }
}
