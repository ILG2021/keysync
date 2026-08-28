<#
.SYNOPSIS
    Full KeySync pipeline: preprocess, compute embeddings, then run inference.

    Steps:
      0. Convert videos to 25 fps and audio to 16 kHz
      1. Compute landmarks
      2. Crop the videos around the face
      3. Compute video embeddings
      4. Compute audio embeddings (WavLM and HuBERT)
      5. Create the filelists
      6. Run inference

.EXAMPLE
    .\scripts\infer_and_compute_emb.ps1 -VideoDir data\videos -AudioDir data\audios `
        -OutputFolder my_animations `
        -KeyframesCkpt pretrained_models\checkpoints\keyframe_dub.pt `
        -InterpolationCkpt pretrained_models\checkpoints\interpolation_dub.pt `
        -ComputeUntil 45
#>
[CmdletBinding()]
param(
    [string]$VideoDir = "data\videos",
    [string]$AudioDir = "data\audios",
    [string]$OutputFolder = "outputs",
    [string]$KeyframesCkpt = "pretrained_models\checkpoints\keyframe_dub.pt",
    [string]$InterpolationCkpt = "pretrained_models\checkpoints\interpolation_dub.pt",
    [string]$ComputeUntil = "45",
    [string]$FixOcclusion = "False",
    [string]$Position = "None",
    [string]$StartFrame = "0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $utilDir = "scripts\util"
    $filelist = "filelist_inference.txt"
    $filelistAudio = "filelist_inference_audio.txt"

    Write-Host "video_dir:          $VideoDir"
    Write-Host "audio_dir:          $AudioDir"
    Write-Host "output_folder:      $OutputFolder"
    Write-Host "keyframes_ckpt:     $KeyframesCkpt"
    Write-Host "interpolation_ckpt: $InterpolationCkpt"
    Write-Host "compute_until:      $ComputeUntil"
    Write-Host "fix_occlusion:      $FixOcclusion"
    Write-Host "position:           $Position"
    Write-Host "start_frame:        $StartFrame"

    # Step 0: standardize video frame rate and audio sample rate
    Write-Host "`nStep 0: Pre-processing video and audio..."
    $videoDir25fps = "${VideoDir}_25fps"
    $audioDir16k = "${AudioDir}_16k"
    New-Item -ItemType Directory -Force -Path $videoDir25fps | Out-Null
    New-Item -ItemType Directory -Force -Path $audioDir16k | Out-Null

    python "$utilDir\ffmpeg_converter.py" `
        --video_dir $VideoDir `
        --video_dir_25fps $videoDir25fps `
        --audio_dir $AudioDir `
        --audio_dir_16k $audioDir16k
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg_converter.py exited with code $LASTEXITCODE" }

    $VideoDir = $videoDir25fps
    $AudioDir = $audioDir16k
    Write-Host "Pre-processing complete. Using processed files from $VideoDir and $AudioDir"

    # Step 1: landmarks
    Write-Host "`nStep 1: Computing landmarks..."
    python "$utilDir\gen_landmarks.py" $VideoDir --output_dir "landmarks_25fps" --batch_size 10
    if ($LASTEXITCODE -ne 0) { throw "gen_landmarks.py exited with code $LASTEXITCODE" }

    # Step 2: crop around the face
    Write-Host "`nStep 2: Cropping video..."
    $videoDirCropped = "${VideoDir}_cropped"
    python "$utilDir\crop_video.py" `
        --video_dir $VideoDir `
        --video_dir_cropped $videoDirCropped `
        --landmarks_dir "landmarks_25fps" `
        --landmarks_dir_cropped "landmarks_25fps_cropped"
    if ($LASTEXITCODE -ne 0) { throw "crop_video.py exited with code $LASTEXITCODE" }
    $VideoDir = $videoDirCropped

    # Step 3: video embeddings
    Write-Host "`nStep 3: Computing video embeddings..."
    python "$utilDir\video_to_latent.py" --filelist $VideoDir
    if ($LASTEXITCODE -ne 0) { throw "video_to_latent.py exited with code $LASTEXITCODE" }

    # Step 4: audio embeddings
    Write-Host "`nStep 4: Computing audio embeddings..."
    python "$utilDir\get_audio_embeddings.py" --audio_path "$AudioDir\*.wav" --model_type wavlm --skip_video
    if ($LASTEXITCODE -ne 0) { throw "get_audio_embeddings.py (wavlm) exited with code $LASTEXITCODE" }
    python "$utilDir\get_audio_embeddings.py" --audio_path "$AudioDir\*.wav" --model_type hubert --skip_video
    if ($LASTEXITCODE -ne 0) { throw "get_audio_embeddings.py (hubert) exited with code $LASTEXITCODE" }

    # Step 5: filelists
    Write-Host "`nStep 5: Creating filelist for inference..."
    python "$utilDir\create_filelist.py" --root_dir $VideoDir --dest_file $filelist --ext ".mp4"
    if ($LASTEXITCODE -ne 0) { throw "create_filelist.py (video) exited with code $LASTEXITCODE" }
    python "$utilDir\create_filelist.py" --root_dir $AudioDir --dest_file $filelistAudio --ext ".wav"
    if ($LASTEXITCODE -ne 0) { throw "create_filelist.py (audio) exited with code $LASTEXITCODE" }

    # Step 6: inference
    Write-Host "`nStep 6: Running inference..."
    & "$PSScriptRoot\inference.ps1" `
        -OutputFolder $OutputFolder `
        -FileList $filelist `
        -FileListAudio $filelistAudio `
        -KeyframesCkpt $KeyframesCkpt `
        -InterpolationCkpt $InterpolationCkpt `
        -ComputeUntil $ComputeUntil `
        -FixOcclusion $FixOcclusion `
        -Position $Position `
        -StartFrame $StartFrame

    Write-Host "`nInference pipeline completed successfully!"
}
finally {
    Pop-Location
}
