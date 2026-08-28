<#
.SYNOPSIS
    Run the KeySync dubbing pipeline on precomputed video/audio embeddings.

.EXAMPLE
    .\scripts\inference.ps1 -OutputFolder my_animations `
        -FileList filelist_inference.txt `
        -FileListAudio filelist_inference_audio.txt `
        -KeyframesCkpt pretrained_models\checkpoints\keyframe_dub.pt `
        -InterpolationCkpt pretrained_models\checkpoints\interpolation_dub.pt `
        -ComputeUntil 45
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputFolder,
    [string]$FileList = "filelist_val.txt",
    [string]$FileListAudio = "None",
    [string]$KeyframesCkpt = "None",
    [string]$InterpolationCkpt = "None",
    [string]$ComputeUntil = "45",
    [string]$FixOcclusion = "False",
    [string]$Position = "None",
    [string]$StartFrame = "0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python scripts\sampling\dubbing_pipeline.py `
        --filelist=$FileList `
        --filelist_audio=$FileListAudio `
        --decoding_t 1 `
        --cond_aug 0. `
        --resize_size=512 `
        "--force_uc_zero_embeddings=[cond_frames,audio_emb]" `
        --latent_folder=videos `
        --video_folder=videos `
        --model_config=scripts/sampling/configs/interpolation.yaml `
        --model_keyframes_config=scripts/sampling/configs/keyframe.yaml `
        --chunk_size=2 `
        --landmark_folder=landmarks `
        --audio_folder=audios `
        --audio_emb_folder=audios `
        --output_folder=outputs/$OutputFolder `
        --keyframes_ckpt=$KeyframesCkpt `
        --interpolation_ckpt=$InterpolationCkpt `
        --add_zero_flag=True `
        --extra_audio=None `
        --compute_until=$ComputeUntil `
        --audio_emb_type=hubert `
        --recompute=True `
        --fix_occlusion=$FixOcclusion `
        --position=$Position `
        --start_frame=$StartFrame
    if ($LASTEXITCODE -ne 0) { throw "dubbing_pipeline.py exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
