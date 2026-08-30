<#
.SYNOPSIS
    Run KeySync over every video/audio pair in a folder and report timings.

.DESCRIPTION
    Pairs the .mp4 and .wav files in -InputDir by file name, runs the lip-sync
    pipeline on each pair, and writes the generated videos plus
    benchmark_results.csv / benchmark_summary.json into -OutputFolder.

    Reported per case: audio duration, generated video duration, wall-clock
    runtime, runtime per second of audio, and peak VRAM.

.EXAMPLE
    .\scripts\benchmark.ps1 -InputDir data\samples -OutputFolder output

.EXAMPLE
    # separate folders, cap each case at 20 seconds, one warm-up run
    .\scripts\benchmark.ps1 -VideoDir data\videos -AudioDir data\audios `
        -OutputFolder output -ComputeUntil 20 -Warmup 1

.EXAMPLE
    # 16 GB card: offload idle models to CPU, run the UNet in bf16
    .\scripts\benchmark.ps1 -InputDir data\samples -CpuOffload -Precision bf16
#>
[CmdletBinding()]
param(
    [string]$InputDir,
    [string]$VideoDir,
    [string]$AudioDir,
    [string]$OutputFolder = "output",
    [ValidateSet("name", "index", "cross")][string]$PairMode = "name",
    [string]$ComputeUntil = "end",
    [int]$Warmup = 0,
    [int]$Limit = 0,
    [switch]$SkipExisting,
    [switch]$FailFast,
    [switch]$CpuOffload,
    [ValidateSet("fp32", "bf16", "fp16")][string]$Precision = "fp32",
    [string]$KeyframesCkpt = "pretrained_models\checkpoints\keyframe_dub.pt",
    [string]$InterpolationCkpt = "pretrained_models\checkpoints\interpolation_dub.pt"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not $InputDir -and (-not $VideoDir -or -not $AudioDir)) {
        throw "Pass -InputDir, or both -VideoDir and -AudioDir."
    }

    $arguments = @(
        "scripts\benchmark_folder.py",
        "--output_folder", $OutputFolder,
        "--pair_mode", $PairMode,
        "--compute_until", $ComputeUntil,
        "--warmup", $Warmup,
        "--precision", $Precision,
        "--keyframes_ckpt", $KeyframesCkpt,
        "--interpolation_ckpt", $InterpolationCkpt
    )
    if ($InputDir) { $arguments += @("--input_dir", $InputDir) }
    if ($VideoDir) { $arguments += @("--video_dir", $VideoDir) }
    if ($AudioDir) { $arguments += @("--audio_dir", $AudioDir) }
    if ($Limit -gt 0) { $arguments += @("--limit", $Limit) }
    if ($SkipExisting) { $arguments += "--skip_existing" }
    if ($FailFast) { $arguments += "--fail_fast" }
    if ($CpuOffload) { $arguments += "--cpu_offload" }

    python @arguments
    if ($LASTEXITCODE -ne 0) { throw "benchmark_folder.py exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
