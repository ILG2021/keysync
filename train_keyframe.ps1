<#
.SYNOPSIS
    Train the KeySync keyframe model.

.DESCRIPTION
    DeepSpeed is not supported on Windows, so the strategy defaults to "auto"
    (single device) here; the Linux script uses deepspeed_stage_1. Pass
    -Strategy ddp for multi-GPU training on Windows - PyTorch Lightning then
    uses the gloo backend, since NCCL is Linux only.

.EXAMPLE
    .\train_keyframe.ps1 -FileList path\to\filelist.txt -Workers 4 -BatchSize 1 -Devices 1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FileList,
    [int]$Workers = 4,
    [int]$BatchSize = 1,
    [string]$Devices = "1",
    [string]$Strategy = "auto"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    Write-Host "Training with $FileList, workers=$Workers, batch_size=$BatchSize, devices=$Devices, strategy=$Strategy"

    python main.py --base configs/example_training/keyframes/keyframes_dub.yaml --wandb True `
        lightning.trainer.num_nodes=1 `
        lightning.strategy=$Strategy `
        lightning.trainer.precision=32 `
        model.base_learning_rate=1.e-5 `
        data.params.train.datapipeline.filelist=$FileList `
        data.params.train.datapipeline.video_folder=video_crop `
        data.params.train.datapipeline.audio_folder=audio `
        data.params.train.datapipeline.audio_emb_folder=audio_emb `
        data.params.train.datapipeline.latent_folder=video_crop_emb `
        data.params.train.datapipeline.landmarks_folder=landmarks_crop `
        data.params.train.loader.num_workers=$Workers `
        data.params.train.datapipeline.audio_in_video=False `
        data.params.train.datapipeline.load_all_possible_indexes=False `
        lightning.trainer.devices=$Devices `
        lightning.trainer.accumulate_grad_batches=1 `
        data.params.train.datapipeline.select_randomly=False `
        model.params.network_config.params.audio_cond_method=both_keyframes `
        data.params.train.datapipeline.what_mask=box `
        data.params.train.datapipeline.balance_datasets=True `
        'model.params.to_freeze=["time_"]' `
        'model.params.to_unfreeze=["time_embed"]' `
        data.params.train.loader.batch_size=$BatchSize `
        data.params.train.datapipeline.audio_emb_type=hubert `
        model.params.loss_fn_config.params.weight_pixel=1 `
        'model.params.loss_fn_config.params.what_pixel_losses=["l2"]' `
        model.params.loss_fn_config.params.lambda_lower=1
    if ($LASTEXITCODE -ne 0) { throw "main.py exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
