
<h1 align="center">KeySync: A Robust Approach for Leakage-free Lip Synchronization in High Resolution</h1>

<div align="center">
    <a href="https://scholar.google.com/citations?user=LuIdiV8AAAAJ" target="_blank">Antoni Bigata</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=08YfKjcAAAAJ" target="_blank">Rodrigo Mira</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=zdg4dj0AAAAJ" target="_blank">Stella Bounareli</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=ty2OYvcAAAAJ" target="_blank">Michał Stypułkowski</a><sup>2</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=WwLpK44AAAAJ" target="_blank">Konstantinos Vougioukas</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=6v-UKEMAAAAJ" target="_blank">Stavros Petridis</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=ygpxbK8AAAAJ" target="_blank">Maja Pantic</a><sup>1</sup>
</div>

<br>

<div align="center">
<div class="is-size-5 publication-authors" style="margin-top: 1rem;">
          <span class="author-block"><sup>1</sup>Imperial College London,</span>
          <span class="author-block"><sup>2</sup>University of Wrocław,</span>
</div>
</div>

<br>

<div align="center">
    <a href="https://antonibigata.github.io/KeySync/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/toninio19/keysync"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow"></a>
    &nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/spaces/toninio19/keysync-demo"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow></a>
    &nbsp;&nbsp;&nbsp;  
    <a href="https://arxiv.org/abs/2505.00497"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
</div>

## 📋 Table of Contents
- [Abstract](#abstract)
- [Demo Examples](#demo-examples)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Batch Benchmarking](#batch-benchmarking)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Abstract

Lip synchronization, known as the task of aligning lip movements in an existing video with new input audio, is typically framed as a simpler variant of audio-driven facial animation. However, as well as suffering from the usual issues in talking head generation (e.g., temporal consistency), lip synchronization presents significant new challenges such as expression leakage from the input video and facial occlusions, which can severely impact real-world applications like automated dubbing, but are often neglected in existing works. To address these shortcomings, we present
KeySync, a two-stage framework that succeeds in solving the issue of temporal consistency, while also incorporating solutions for leakage and occlusions using a carefully designed masking strategy. We show that KeySync achieves state-of-the-art results in lip reconstruction and cross-synchronization, improving visual quality and reducing expression leakage according to LipLeak, our novel leakage metric. Furthermore, we demonstrate the effectivness of our new masking approach in handling occlusions and validate our architectural choices through several ablation studies.

### Media

<table>
  <tr>
    <td><img src="assets/media/vid_dub_1.gif" alt="Video 1"/></td>
    <td><img src="assets/media/vid_dub_2.gif" alt="Video 2"/></td>
    <td><img src="assets/media/vid_dub_3.gif" alt="Video 3"/></td>
    <td><img src="assets/media/vid_dub_4.gif" alt="Video 4"/></td>
  </tr>
</table>

For more visualizations, please visit [https://antonibigata.github.io/KeySync/](https://antonibigata.github.io/KeySync/)

### Online Demo

We provide an interactive demo of KeySync at [https://huggingface.co/spaces/toninio19/keysync-demo](https://huggingface.co/spaces/toninio19/keysync-demo) where you can upload your own video and audio files to create synchronized videos. Due to GPU restrictions on Hugging Face Spaces, the demo is limited to processing videos of maximum 6 seconds in length. For longer videos or better performance, we recommend using the inference scripts provided in this repository to run KeySync locally on your own hardware.

## Architecture

<div align="center">
  <img src="assets/media/drawing-1.png" width="100%">
</div>

## Installation

This fork targets **Windows first**. Everything below is written for PowerShell
on Windows 10/11; the Linux/macOS instructions follow at the end of each
section and use the original `.sh` scripts, which are still maintained.

### Prerequisites
- NVIDIA GPU with a recent driver (CUDA 12.1 compatible)
- Python 3.11
- Conda package manager (Miniconda/Anaconda) - or a plain `venv` plus a separate ffmpeg install
- FFmpeg on `PATH` (the conda command below installs it for you)

### Setup Environment (Windows, PowerShell)

```powershell
# Create conda environment with necessary dependencies
conda create -n KeySync python=3.11 conda-forge::ffmpeg -y
conda activate KeySync

# Install requirements
python -m pip install -r requirements.txt --no-deps

# Install PyTorch with CUDA support
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` installs cleanly on Windows: it no longer contains
`xformers`, and the packages that only ship Linux wheels (`triton`,
`deepspeed`, the `nvidia-*-cu12` runtime wheels) carry a `sys_platform`
marker, so pip skips them on Windows and still installs them on Linux.

Optional extras that need the Visual Studio Build Tools (C++) and CMake on
Windows live in a separate file, install them only if you need them:

```powershell
python -m pip install -r requirements-optional.txt
```

The occlusion-handling pipeline additionally needs SAM 2:

```powershell
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e . --no-deps
cd ..
```

If PowerShell refuses to run the `.ps1` scripts in this repo
("running scripts is disabled on this system"), allow local scripts for your
user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Attention backend: SDPA instead of xformers

All attention now runs through PyTorch's built-in
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which is part of the
regular Windows CUDA wheels - xformers, which has no official Windows build, is
no longer a dependency. The config keys are `softmax-sdpa` and `vanilla-sdpa`;
the old `softmax-xformers` / `vanilla-xformers` values still work as deprecated
aliases, so existing configs and checkpoints keep loading unchanged.

On Windows, SDPA is restricted to the memory-efficient and math kernels,
because the Windows builds of PyTorch ship without FlashAttention.

### Setup Environment (Linux/macOS)

```bash
conda create -n KeySync python=3.11 conda-forge::ffmpeg -y
conda activate KeySync
python -m pip install -r requirements.txt --no-deps
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# OPTIONAL
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e . --no-deps
```

### Known Issues

- On some machines, you may need to install `nvidia::cuda-nvcc`. If this is the case, you can do so by running:

```bash
conda install nvidia::cuda-nvcc
```

- If you encounter synchronization issues between omegaconf and antlr4, you can fix them by running:


```bash
python -m pip uninstall omegaconf antlr4-python3-runtime -y
python -m pip install "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3"
```

- **Windows:** DeepSpeed is Linux-only. The training scripts default to the
  `auto` strategy; pass `-Strategy ddp` for multi-GPU training (PyTorch
  Lightning then uses the gloo backend, since NCCL is Linux-only too).

- **Windows:** if a checkpoint or dataset path fails with "path too long",
  enable long paths:
  `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1`
  (needs an elevated PowerShell and a reboot).


### Download Pretrained Models

The KeySync checkpoints live on the Hugging Face Hub and have to be downloaded
manually. Run this from the repository root:

```powershell
git lfs install
git clone https://huggingface.co/toninio19/keysync pretrained_models
```

Without Git LFS you can use the Hugging Face CLI instead, which is already
installed as part of `huggingface-hub`:

```powershell
huggingface-cli download toninio19/keysync --local-dir pretrained_models
```

Either way you should end up with:

```
pretrained_models\checkpoints\keyframe_dub.pt         # keyframe model
pretrained_models\checkpoints\interpolation_dub.pt    # interpolation model
pretrained_models\checkpoints\WavLM-Base+.pt          # audio encoder
```

Those three paths are the defaults every script expects; pass
`-KeyframesCkpt` / `-InterpolationCkpt` (or `--wavlm_ckpt`) if you put them
elsewhere.

For the occlusion pipeline (`--fix_occlusion`) you also need the SAM 2
checkpoint at `pretrained_models\checkpoints\sam2.1_hiera_large.pt`,
downloaded from the [SAM 2 repository](https://github.com/facebookresearch/sam2).

### Models downloaded automatically on first run

Three more models are pulled from the Hub the first time you run inference and
then cached in `%USERPROFILE%\.cache\huggingface` and `%USERPROFILE%\.cache\torch`:

| Model | Used for |
|-------|----------|
| `facebook/hubert-base-ls960` | audio embeddings |
| `stabilityai/stable-video-diffusion-img2vid` (VAE only) | video latents |
| `face-alignment` s3fd + 2DFAN4 weights | facial landmarks |

The Stable Video Diffusion repository is **gated**: accept its license on the
[model page](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
while logged in, then authenticate locally, otherwise the first run fails with
a 401/403:

```powershell
huggingface-cli login
```

So the first run needs an internet connection and is noticeably slower than the
following ones.

## Quick Start Guide

### 1. Data Preparation

To use KeySync with your own data, for simplicity organize your files as follows:
- Place video files (`.mp4`) in the `data/videos/` directory
- Place audio files (`.wav`) in the `data/audios/` directory
Otherwise you need to specify a different video_dir and audio_dir.

To prepare the data, we provide some scripts in `scripts/util`:
- **Standardize video and audio formats:** Use `ffmpeg_converter.py` to convert videos to 25 fps and audio to 16000Hz. This ensures compatibility with the models.

- **Crop videos around the face:** Use `crop_video.py` to crop the videos, which can improve performance and focus the model's attention on the relevant facial region.


### 2. Running Inference

For inference you need to have the audio and video embeddings precomputed.
The simplest way to run inference on your own data is the `infer_raw.ps1`
script, which computes those embeddings for you:

```powershell
.\scripts\infer_raw.ps1 `
  -FileList "data\videos" `
  -FileListAudio "data\audios" `
  -OutputFolder "my_animations" `
  -KeyframesCkpt "path\to\keyframe_dub.pt" `
  -InterpolationCkpt "path\to\interpolation_dub.pt" `
  -ComputeUntil 45
```

This script handles the entire pipeline:
1. Extracts video embeddings
2. Computes landmarks
3. Computes audio embeddings (using WavLM, and Hubert)
4. Creates a filelist for inference
5. Runs the full animation pipeline

For more control over the inference process, you can directly use
`inference.ps1`:

```powershell
.\scripts\inference.ps1 `
  -OutputFolder "output_folder_name" `
  -FileList "path\to\filelist.txt" `
  -KeyframesCkpt "path\to\keyframes_model.ckpt" `
  -InterpolationCkpt "path\to\interpolation_model.ckpt" `
  -ComputeUntil 45
```

or, if you also want the intermediate embeddings written to disk for faster
recomputes:

```powershell
.\scripts\infer_and_compute_emb.ps1 `
  -VideoDir "data\videos" `
  -AudioDir "data\audios" `
  -OutputFolder "my_animations" `
  -KeyframesCkpt "path\to\keyframes_model.ckpt" `
  -InterpolationCkpt "path\to\interpolation_model.ckpt" `
  -ComputeUntil 45
```

<details>
<summary>Linux/macOS equivalents (bash)</summary>

```bash
bash scripts/infer_raw.sh \
  --file_list "data/videos" \
  --file_list_audio "data/audios" \
  --output_folder "my_animations" \
  --keyframes_ckpt "path/to/keyframe_dub.pt" \
  --interpolation_ckpt "path/to/interpolation_dub.pt" \
  --compute_until 45

bash scripts/inference.sh \
  --output_folder "output_folder_name" \
  --file_list "path/to/filelist.txt" \
  --keyframes_ckpt "path/to/keyframes_model.ckpt" \
  --interpolation_ckpt "path/to/interpolation_model.ckpt" \
  --compute_until "compute_until"

bash scripts/infer_and_compute_emb.sh \
  --video_dir "data/videos" \
  --audio_dir "data/audios" \
  --output_folder "my_animations" \
  --keyframes_ckpt "path/to/keyframes_model.ckpt" \
  --interpolation_ckpt "path/to/interpolation_model.ckpt" \
  --compute_until 45
```

</details>

### 3. Training Your Own Models

The dataloader needs the path to all the videos you want to train on. Then you need to separate the audio and video as follows:
- root_folder:
  - videos: raw videos
  - videos_emb: embedding for your videos
  - audios: raw audios
  - audios_emb: precomputed embeddigns for the audios
  - landmarks_folder: landmarks computed from raw video
  
You can have different folders but make sure to change them in the training scripts.

KeySYnc uses a two-stage model approach. You can train each component separately:

#### KeySync Model Training

```powershell
.\train_keyframe.ps1 -FileList path\to\filelist.txt -Workers 4 -BatchSize 1 -Devices 1
```

#### Interpolation Model Training

```powershell
.\train_interpolation.ps1 -FileList path\to\filelist.txt -Workers 4 -BatchSize 1 -Devices 1
```

Both scripts default to `-Strategy auto`, since DeepSpeed (used by the Linux
scripts) has no Windows support. For multi-GPU training on Windows pass
`-Strategy ddp`.

<details>
<summary>Linux/macOS equivalents (bash)</summary>

```bash
bash train_keyframe.sh path/to/filelist.txt [num_workers] [batch_size] [num_devices]
bash train_interpolation.sh path/to/filelist.txt [num_workers] [batch_size] [num_devices]
```

</details>

## Batch Benchmarking

`scripts/benchmark_folder.py` runs the whole folder as a batch of test cases and
reports how fast and how memory-hungry the pipeline is. Point it at a folder
that holds both the videos and the audio files - they are paired by file name
(`interview.mp4` + `interview.wav`).

```powershell
.\scripts\benchmark.ps1 -InputDir data\samples -OutputFolder output
```

```powershell
# separate folders, cap each case at 20 seconds, discard one warm-up run
.\scripts\benchmark.ps1 -VideoDir data\videos -AudioDir data\audios `
    -OutputFolder output -ComputeUntil 20 -Warmup 1
```

Or call the script directly (works the same on Linux):

```bash
python scripts/benchmark_folder.py \
  --input_dir data/samples \
  --output_folder output \
  --keyframes_ckpt pretrained_models/checkpoints/keyframe_dub.pt \
  --interpolation_ckpt pretrained_models/checkpoints/interpolation_dub.pt
```

### What it produces

Everything lands in the output folder (`output/` by default):

| File | Contents |
|------|----------|
| `<name>.mp4` | the lip-synced video for each case |
| `benchmark_results.csv` | one row per case, all measurements |
| `benchmark_summary.json` | aggregate stats, machine readable |

Per case it records the audio duration, the generated video duration, the
wall-clock runtime, **runtime / audio duration** (seconds of compute per second
of audio; its inverse is printed as the realtime factor) and the **peak VRAM**.
A table and a summary are printed at the end:

```
  #  case                    audio(s)   out(s)   time(s)  time/audio    xRT  GPU peak(MB)  torch(MB)
  1  interview.mp4              12.40    12.40    143.20       11.55   0.09         11480       9840
  2  podcast.mp4                30.10    30.10    342.70       11.39   0.09         11512       9840

  Total audio duration     : 42.50 s
  Total runtime            : 485.90 s
  Runtime / audio duration : 11.433 (median per case 11.470)
  Peak VRAM (whole GPU)    : 11512 MB
```

### Notes

- The models are loaded **once** and reused, so the per-case runtimes exclude
  model loading; that time is reported separately in the summary.
- The first case is always slower (cuDNN autotuning, lazy CUDA init). Use
  `-Warmup 1` to run it but keep it out of the averages.
- Peak VRAM is reported two ways: `GPU peak` is sampled from NVML and matches
  what `nvidia-smi` shows for the whole device (close other GPU apps for a
  clean number), while `torch` is the PyTorch allocator's own peak.
- Inputs are expected at 25 fps and 16 kHz. The script warns about other frame
  rates - convert first with `scripts/util/ffmpeg_converter.py`.
- Useful options: `--pair_mode index` pairs by sorted order instead of by name,
  `--pair_mode cross` runs every video against every audio file (cross-sync
  tests), `--limit N` runs only the first N cases, `--skip_existing` resumes an
  interrupted run.

## Advanced Usage

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `video_dir` | Directory with input videos | `data/videos` |
| `audio_dir` | Directory with input audio files | `data/audios` |
| `output_folder` | Where to save generated animations | - |
| `keyframes_ckpt` | Keyframe model checkpoint path | - |
| `interpolation_ckpt` | Interpolation model checkpoint path | - |
| `compute_until` | Animation length in seconds | 45 |
| `fix_occlusion` | Enable occlusion handling to mask objects that block the face | False |
| `position` | Coordinates of the object to mask in the occlusion pipeline (format: x,y, e.g., "450,450") | None |
| `start_frame` | Frame number where the specified position coordinates apply (using the first frame typically works best) | 0 |

The PowerShell scripts use the same options in `-PascalCase` form
(`-VideoDir`, `-AudioDir`, `-OutputFolder`, `-KeyframesCkpt`,
`-InterpolationCkpt`, `-ComputeUntil`, `-FixOcclusion`, `-Position`,
`-StartFrame`).

### Advanced Configuration

For more fine-grained control, you can edit the configuration files in the `configs/` directory.

## LipScore Evaluation

KeySync can be evaluated using the LipScore metric available in the `evaluation/` folder. This metric measures the lip synchronization quality between generated and ground truth videos.

To use the LipScore evaluation, you'll need to install the following dependencies:

1. Face detection library: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection)
2. Face alignment library: [https://github.com/ibug-group/face_alignment](https://github.com/ibug-group/face_alignment)

Once installed, you can use the LipScore class in `evaluation/lipscore.py` to evaluate your generated animations:

## Known issue

The current approach uses different audio models for the keyframe and interpolation models. If you’re retraining and plan to use the same audio model for both, you may need to update this section of the inference code:
https://github.com/antonibigata/keysync/blob/f7827b041d9d30740a96998f0f9946ae19b2d248/scripts/sampling/dubbing_pipeline.py#L500-L503

## Citation

If you use KeySync in your research, please cite our paper:

```bibtex
@misc{bigata2025keysyncrobustapproachleakagefree,
      title={KeySync: A Robust Approach for Leakage-free Lip Synchronization in High Resolution}, 
      author={Antoni Bigata and Rodrigo Mira and Stella Bounareli and Michał Stypułkowski and Konstantinos Vougioukas and Stavros Petridis and Maja Pantic},
      year={2025},
      eprint={2505.00497},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.00497}, 
}
```

## Contributors

<a href="https://github.com/kacperkan">
  <img src="https://github.com/kacperkan.png?size=100" alt="kacperkan" width="100">
</a>

## Acknowledgements

This project builds upon the foundation provided by [Stability AI's Generative Models](https://github.com/Stability-AI/generative-models). We thank the Stability AI team for their excellent work and for making their code publicly available.
