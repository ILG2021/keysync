"""Batch-run KeySync over a folder of matching video/audio files and report timings.

Walks a folder that holds both the videos (`.mp4`) and the audio files
(`.wav`), pairs them up by file name, runs the raw dubbing pipeline on every
pair, and records for each case:

  * the audio duration and the generated video duration
  * the wall-clock runtime, and the runtime / audio-duration ratio
  * the peak VRAM (PyTorch allocator, and the whole GPU as nvidia-smi sees it)

The lip-synced videos are written to the output folder (``output`` by default),
together with ``benchmark_results.csv`` and ``benchmark_summary.json``.

The models are loaded once and reused for every case, so the reported runtimes
do not include model loading (that is reported separately in the summary).

Inputs are expected at 25 fps / 16 kHz, like the rest of the pipeline; run
``scripts/util/ffmpeg_converter.py`` first if yours are not, otherwise the
lip sync will drift. The script warns when it sees a different frame rate.

Example (PowerShell):

    python scripts\\benchmark_folder.py `
        --input_dir data\\samples `
        --output_folder output `
        --keyframes_ckpt pretrained_models\\checkpoints\\keyframe_dub.pt `
        --interpolation_ckpt pretrained_models\\checkpoints\\interpolation_dub.pt
"""

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

VIDEO_EXTS = (".mp4",)
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
MB = 1024.0 * 1024.0


# --------------------------------------------------------------------------- #
# VRAM measurement
# --------------------------------------------------------------------------- #
class GpuMemoryMonitor:
    """Samples total GPU memory usage in the background, via NVML.

    `torch.cuda.max_memory_*` only sees the PyTorch caching allocator; the CUDA
    context, cuDNN workspaces and any other library add several hundred MB on
    top. NVML gives the number nvidia-smi would show. Per-process accounting is
    not available under the Windows WDDM driver model, so this samples the
    whole device - close other GPU applications for a clean measurement.
    """

    def __init__(self, device_index=0, interval=0.05):
        self.device_index = device_index
        self.interval = interval
        self.available = False
        self.peak_bytes = 0
        self._handle = None
        self._pynvml = None
        self._thread = None
        self._stop = threading.Event()

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception as e:  # pynvml missing, or no NVML on this machine
            print(f"[benchmark] NVML unavailable ({e}); reporting torch memory only.")

    def _sample(self):
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return int(info.used)
        except Exception:
            return 0

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak_bytes = max(self.peak_bytes, self._sample())

    def start(self):
        """Reset the peak and start sampling."""
        if not self.available:
            return
        self.peak_bytes = self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and return the peak in bytes (0 if unavailable)."""
        if not self.available:
            return 0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1.0)
            self._thread = None
        self.peak_bytes = max(self.peak_bytes, self._sample())
        return self.peak_bytes

    def close(self):
        if self.available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Media helpers
# --------------------------------------------------------------------------- #
def audio_duration_seconds(path):
    """Duration of an audio file in seconds, or None if it cannot be read."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        import torchaudio

        info = torchaudio.info(path)
        return info.num_frames / float(info.sample_rate)
    except Exception as e:
        print(f"[benchmark] could not read audio duration of {path}: {e}")
        return None


def video_info(path):
    """(duration_seconds, fps, n_frames) of a video, or (None, None, None)."""
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if fps > 0 and n_frames > 0:
                return n_frames / fps, fps, n_frames
    except Exception:
        pass
    try:
        import decord

        vr = decord.VideoReader(path)
        fps = float(vr.get_avg_fps())
        n_frames = len(vr)
        if fps > 0:
            return n_frames / fps, fps, n_frames
    except Exception as e:
        print(f"[benchmark] could not read video info of {path}: {e}")
    return None, None, None


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def list_media(folder, extensions):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a directory: {folder}")
    files = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(extensions)
    ]
    return files


def collect_pairs(video_dir, audio_dir, pair_mode):
    """Return a list of (video_path, audio_path) tuples."""
    videos = list_media(video_dir, VIDEO_EXTS)
    audios = list_media(audio_dir, AUDIO_EXTS)

    if not videos:
        raise FileNotFoundError(f"No {'/'.join(VIDEO_EXTS)} files found in {video_dir}")
    if not audios:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    if pair_mode == "cross":
        return [(v, a) for v in videos for a in audios]

    if pair_mode == "index":
        if len(videos) != len(audios):
            print(
                f"[benchmark] warning: {len(videos)} videos vs {len(audios)} audio "
                f"files, pairing the first {min(len(videos), len(audios))} by order."
            )
        return list(zip(videos, audios))

    # pair_mode == "name": match on the file name without extension
    audio_by_stem = {}
    for path in audios:
        audio_by_stem.setdefault(os.path.splitext(os.path.basename(path))[0], path)

    pairs, unmatched_videos = [], []
    for video in videos:
        stem = os.path.splitext(os.path.basename(video))[0]
        if stem in audio_by_stem:
            pairs.append((video, audio_by_stem[stem]))
        else:
            unmatched_videos.append(video)

    matched_audio = {a for _, a in pairs}
    unmatched_audios = [a for a in audios if a not in matched_audio]

    if unmatched_videos:
        print(
            f"[benchmark] {len(unmatched_videos)} video(s) without a matching audio "
            f"file: {', '.join(os.path.basename(v) for v in unmatched_videos[:5])}"
            f"{' ...' if len(unmatched_videos) > 5 else ''}"
        )
    if unmatched_audios:
        print(
            f"[benchmark] {len(unmatched_audios)} audio file(s) without a matching "
            f"video: {', '.join(os.path.basename(a) for a in unmatched_audios[:5])}"
            f"{' ...' if len(unmatched_audios) > 5 else ''}"
        )
    if not pairs:
        raise FileNotFoundError(
            "No video/audio pairs share a file name. Use --pair_mode index to pair "
            "them by sorted order instead, or --pair_mode cross for all combinations."
        )
    return pairs


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_pipeline_module():
    """Import scripts/sampling/dubbing_pipeline_raw.py as a module."""
    path = os.path.join(REPO_ROOT, "scripts", "sampling", "dubbing_pipeline_raw.py")
    spec = importlib.util.spec_from_file_location("keysync_dubbing_pipeline_raw", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_models(pipeline, args):
    """Load every model the pipeline needs, once."""
    model, _, n_batch = pipeline.load_model(
        args.model_config,
        args.device,
        args.num_frames,
        "latents",
        None if args.interpolation_ckpt in (None, "None") else args.interpolation_ckpt,
    )
    model_keyframes, _, n_batch_keyframes = pipeline.load_model(
        args.model_keyframes_config,
        args.device,
        args.num_frames,
        "latents",
        None if args.keyframes_ckpt in (None, "None") else args.keyframes_ckpt,
    )
    hubert_model = pipeline.HubertModel.from_pretrained(
        "facebook/hubert-base-ls960"
    ).to(args.device)
    wavlm_model = pipeline.WavLM_wrapper(
        model_size="Base+",
        feed_as_frames=False,
        merge_type="None",
        model_path=args.wavlm_ckpt,
    ).to(args.device)
    vae_model = pipeline.VaeWrapper("video")
    landmarks_model = pipeline.LandmarksExtractor()

    from sgm.util import set_diffusion_precision

    for engine in (model, model_keyframes):
        set_diffusion_precision(engine, args.precision)

    if args.cpu_offload:
        # sample() pulls each model onto the GPU only while it is in use.
        for offloaded in (model, model_keyframes, hubert_model, wavlm_model, vae_model):
            offloaded.to("cpu")
        torch.cuda.empty_cache()

    return {
        "model": model,
        "model_keyframes": model_keyframes,
        "n_batch": n_batch,
        "n_batch_keyframes": n_batch_keyframes,
        "hubert_model": hubert_model,
        "wavlm_model": wavlm_model,
        "vae_model": vae_model,
        "landmarks_model": landmarks_model,
    }


def output_naming(video_path, audio_path):
    """(extra_naming, output_basename) - mirrors how sample() names its output.

    sample() writes <video_stem>_<extra_naming>.mp4, so when the video and the
    audio share a name we drop the suffix and keep the plain <stem>.mp4.
    """
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    audio_stem = os.path.splitext(os.path.basename(audio_path))[0]
    if video_stem == audio_stem:
        return "", f"{video_stem}.mp4"
    return audio_stem, f"{video_stem}_{audio_stem}.mp4"


def expected_output_path(output_folder, video_path, audio_path):
    return os.path.join(output_folder, output_naming(video_path, audio_path)[1])


def run_one(pipeline, models, args, video_path, audio_path):
    """Run a single pair, return the sample() keyword arguments applied."""
    pipeline.sample(
        models["model"],
        models["model_keyframes"],
        video_path=video_path,
        audio_path=audio_path,
        num_frames=args.num_frames,
        resize_size=args.resize_size,
        version="svd",
        fps_id=args.fps_id,
        cond_aug=args.cond_aug,
        seed=args.seed,
        decoding_t=args.decoding_t,
        device=args.device,
        output_folder=args.output_folder,
        force_uc_zero_embeddings=["cond_frames", "audio_emb"],
        chunk_size=args.chunk_size,
        add_zero_flag=True,
        n_batch=models["n_batch"],
        n_batch_keyframes=models["n_batch_keyframes"],
        compute_until=args.compute_until,
        extra_audio=None,
        audio_emb_type=args.audio_emb_type,
        extra_naming=output_naming(video_path, audio_path)[0],
        what_mask=args.what_mask,
        cpu_offload=args.cpu_offload,
        precision=args.precision,
        paste_back=args.paste_back,
        recompute=not args.skip_existing,
        hubert_model=models["hubert_model"],
        wavlm_model=models["wavlm_model"],
        vae_model=models["vae_model"],
        landmarks_model=models["landmarks_model"],
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "index",
    "warmup",
    "status",
    "video",
    "audio",
    "output",
    "audio_sec",
    "input_video_sec",
    "input_video_fps",
    "output_video_sec",
    "runtime_sec",
    "runtime_per_audio_sec",
    "realtime_factor",
    "peak_gpu_used_mb",
    "peak_torch_alloc_mb",
    "peak_torch_reserved_mb",
    "error",
]


def fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(rows):
    header = (
        f"{'#':>3}  {'case':<40} {'audio(s)':>9} {'out(s)':>8} {'time(s)':>9} "
        f"{'time/audio':>11} {'xRT':>6} {'GPU peak(MB)':>13} {'torch(MB)':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        case = os.path.basename(row["output"] or row["video"])
        if len(case) > 40:
            case = case[:37] + "..."
        if row["status"] == "error":
            flag = "!"
        elif row["status"] == "skipped":
            flag = "s"
        else:
            flag = "w" if row["warmup"] else " "
        print(
            f"{row['index']:>3}{flag} {case:<40} "
            f"{fmt(row['audio_sec']):>9} {fmt(row['output_video_sec']):>8} "
            f"{fmt(row['runtime_sec']):>9} {fmt(row['runtime_per_audio_sec']):>11} "
            f"{fmt(row['realtime_factor']):>6} "
            f"{fmt(row['peak_gpu_used_mb'], 0):>13} "
            f"{fmt(row['peak_torch_reserved_mb'], 0):>10}"
        )


def build_summary(rows, model_load_sec, args):
    counted = [r for r in rows if r["status"] == "ok" and not r["warmup"]]
    failed = [r for r in rows if r["status"] == "error"]
    skipped = [r for r in rows if r["status"] == "skipped"]

    total_audio = sum(r["audio_sec"] or 0.0 for r in counted)
    total_runtime = sum(r["runtime_sec"] or 0.0 for r in counted)
    ratios = [
        r["runtime_per_audio_sec"] for r in counted if r["runtime_per_audio_sec"]
    ]

    def peak(field):
        values = [r[field] for r in rows if r[field]]
        return max(values) if values else None

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "torch_version": torch.__version__,
        "cases_total": len(rows),
        "cases_ok": len([r for r in rows if r["status"] == "ok"]),
        "cases_failed": len(failed),
        "cases_skipped": len(skipped),
        "cases_counted": len(counted),
        "warmup_cases": args.warmup,
        "model_load_sec": round(model_load_sec, 2),
        "total_audio_sec": round(total_audio, 2),
        "total_runtime_sec": round(total_runtime, 2),
        "overall_runtime_per_audio_sec": round(total_runtime / total_audio, 3)
        if total_audio
        else None,
        "overall_realtime_factor": round(total_audio / total_runtime, 3)
        if total_runtime
        else None,
        "mean_runtime_per_audio_sec": round(statistics.fmean(ratios), 3)
        if ratios
        else None,
        "median_runtime_per_audio_sec": round(statistics.median(ratios), 3)
        if ratios
        else None,
        "peak_gpu_used_mb": peak("peak_gpu_used_mb"),
        "peak_torch_alloc_mb": peak("peak_torch_alloc_mb"),
        "peak_torch_reserved_mb": peak("peak_torch_reserved_mb"),
        "failed_cases": [
            {"video": r["video"], "audio": r["audio"], "error": r["error"]}
            for r in failed
        ],
    }


def print_summary(summary, output_folder):
    print("\n" + "=" * 78)
    print("Benchmark summary")
    print("=" * 78)
    print(f"  GPU                      : {summary['device_name']}")
    print(
        f"  Cases                    : {summary['cases_ok']} ok, "
        f"{summary['cases_failed']} failed, {summary['cases_skipped']} skipped "
        f"({summary['cases_counted']} counted, {summary['warmup_cases']} warmup)"
    )
    print(f"  Model load time          : {fmt(summary['model_load_sec'])} s")
    print(f"  Total audio duration     : {fmt(summary['total_audio_sec'])} s")
    print(f"  Total runtime            : {fmt(summary['total_runtime_sec'])} s")
    print(
        f"  Runtime / audio duration : "
        f"{fmt(summary['overall_runtime_per_audio_sec'], 3)} "
        f"(median per case {fmt(summary['median_runtime_per_audio_sec'], 3)})"
    )
    print(
        f"  Realtime factor          : "
        f"{fmt(summary['overall_realtime_factor'], 3)} x realtime"
    )
    print(f"  Peak VRAM (whole GPU)    : {fmt(summary['peak_gpu_used_mb'], 0)} MB")
    print(
        f"  Peak VRAM (torch alloc)  : {fmt(summary['peak_torch_alloc_mb'], 0)} MB "
        f"/ reserved {fmt(summary['peak_torch_reserved_mb'], 0)} MB"
    )
    print(f"  Videos + reports         : {os.path.abspath(output_folder)}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def get_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    io_group = p.add_argument_group("input / output")
    io_group.add_argument(
        "--input_dir",
        default=None,
        help="Folder holding both the videos and the audio files.",
    )
    io_group.add_argument(
        "--video_dir", default=None, help="Video folder (defaults to --input_dir)."
    )
    io_group.add_argument(
        "--audio_dir", default=None, help="Audio folder (defaults to --input_dir)."
    )
    io_group.add_argument(
        "--output_folder", default="output", help="Where to write videos and reports."
    )
    io_group.add_argument(
        "--pair_mode",
        choices=["name", "index", "cross"],
        default="name",
        help="How to pair videos with audio: by file name (default), by sorted "
        "order, or every combination.",
    )
    io_group.add_argument(
        "--limit", type=int, default=None, help="Only run the first N pairs."
    )
    io_group.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Run the first N pairs but exclude them from the aggregate stats.",
    )
    io_group.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip pairs whose output video already exists.",
    )
    io_group.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop at the first failure instead of continuing.",
    )

    model_group = p.add_argument_group("models")
    model_group.add_argument(
        "--keyframes_ckpt",
        default=os.path.join("pretrained_models", "checkpoints", "keyframe_dub.pt"),
    )
    model_group.add_argument(
        "--interpolation_ckpt",
        default=os.path.join(
            "pretrained_models", "checkpoints", "interpolation_dub.pt"
        ),
    )
    model_group.add_argument(
        "--wavlm_ckpt",
        default=os.path.join("pretrained_models", "checkpoints", "WavLM-Base+.pt"),
    )
    model_group.add_argument(
        "--model_config",
        default=os.path.join("scripts", "sampling", "configs", "interpolation.yaml"),
    )
    model_group.add_argument(
        "--model_keyframes_config",
        default=os.path.join("scripts", "sampling", "configs", "keyframe.yaml"),
    )

    run_group = p.add_argument_group("inference settings (defaults match infer_raw)")
    run_group.add_argument("--device", default="cuda")
    run_group.add_argument("--num_frames", type=int, default=14)
    run_group.add_argument("--resize_size", type=int, default=512)
    run_group.add_argument("--decoding_t", type=int, default=1)
    run_group.add_argument("--chunk_size", type=int, default=2)
    run_group.add_argument("--cond_aug", type=float, default=0.0)
    run_group.add_argument("--fps_id", type=int, default=24)
    run_group.add_argument("--seed", type=int, default=23)
    run_group.add_argument("--audio_emb_type", default="hubert")
    run_group.add_argument("--what_mask", default="box")
    run_group.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Offload models to CPU when idle: only the model that is currently "
        "working stays on the GPU, the rest sit in system RAM. Lets the two "
        "stage models fit in ~16 GB of VRAM, at the cost of a CPU<->GPU "
        "transfer per chunk.",
    )
    run_group.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Precision for the UNet and conditioner (the VAE always stays "
        "fp32). bf16 roughly halves their weights and is faster on Ada/Ampere.",
    )
    run_group.add_argument(
        "--paste_back",
        action="store_true",
        help="Crop a square around the face, animate it, then paste the result "
        "back into the original-resolution frames. Non-square input is restored "
        "automatically; this flag enables the same behavior for square input.",
    )
    run_group.add_argument(
        "--compute_until",
        default="end",
        help='Seconds of animation to generate per case, or "end".',
    )
    return p


def main():
    args = get_parser().parse_args()

    video_dir = args.video_dir or args.input_dir
    audio_dir = args.audio_dir or args.input_dir
    if not video_dir or not audio_dir:
        raise SystemExit(
            "Pass --input_dir (a folder with both videos and audio), or both "
            "--video_dir and --audio_dir."
        )
    if args.compute_until != "end":
        # sample() turns this into a frame count, so it has to stay an int
        args.compute_until = int(float(args.compute_until))

    os.makedirs(args.output_folder, exist_ok=True)

    pairs = collect_pairs(video_dir, audio_dir, args.pair_mode)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"[benchmark] {len(pairs)} case(s) to run, output -> {args.output_folder}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; KeySync needs an NVIDIA GPU.")

    monitor = GpuMemoryMonitor(
        device_index=torch.cuda.current_device() if torch.cuda.is_available() else 0
    )

    pipeline = load_pipeline_module()
    print("[benchmark] loading models...")
    load_start = time.perf_counter()
    models = load_models(pipeline, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_sec = time.perf_counter() - load_start
    print(f"[benchmark] models ready in {model_load_sec:.1f} s")

    rows = []
    error_log = []
    for i, (video_path, audio_path) in enumerate(pairs, start=1):
        out_path = expected_output_path(args.output_folder, video_path, audio_path)
        row = {field: None for field in CSV_FIELDS}
        row.update(
            {
                "index": i,
                "warmup": i <= args.warmup,
                "status": "ok",
                "video": video_path,
                "audio": audio_path,
                "output": out_path,
                "error": "",
            }
        )

        if args.skip_existing and os.path.exists(out_path):
            row["status"] = "skipped"
            rows.append(row)
            print(f"[benchmark] ({i}/{len(pairs)}) skipping, output exists: {out_path}")
            continue

        row["audio_sec"] = audio_duration_seconds(audio_path)
        in_sec, in_fps, _ = video_info(video_path)
        row["input_video_sec"] = in_sec
        row["input_video_fps"] = in_fps
        if in_fps is not None and abs(in_fps - 25.0) > 0.5:
            print(
                f"[benchmark] warning: {os.path.basename(video_path)} is {in_fps:.2f} "
                f"fps, the pipeline assumes 25 fps. Convert it with "
                f"scripts/util/ffmpeg_converter.py for correct lip sync."
            )

        print(
            f"\n[benchmark] ({i}/{len(pairs)}) "
            f"{os.path.basename(video_path)} + {os.path.basename(audio_path)} "
            f"(audio {fmt(row['audio_sec'])} s)"
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        monitor.start()
        start = time.perf_counter()
        stop_after = False
        try:
            run_one(pipeline, models, args, video_path, audio_path)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:  # keep going, record the failure
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            error_log.append(
                f"### {video_path} + {audio_path}\n{tb}"
            )
            print(f"[benchmark] FAILED: {row['error']}")
            print(tb, file=sys.stderr)
            stop_after = args.fail_fast
        finally:
            row["runtime_sec"] = time.perf_counter() - start
            gpu_peak = monitor.stop()

        row["peak_gpu_used_mb"] = gpu_peak / MB if gpu_peak else None
        if torch.cuda.is_available():
            row["peak_torch_alloc_mb"] = torch.cuda.max_memory_allocated() / MB
            row["peak_torch_reserved_mb"] = torch.cuda.max_memory_reserved() / MB

        if row["status"] == "ok":
            out_sec, _, _ = video_info(out_path)
            row["output_video_sec"] = out_sec
            if row["audio_sec"]:
                row["runtime_per_audio_sec"] = row["runtime_sec"] / row["audio_sec"]
                row["realtime_factor"] = row["audio_sec"] / row["runtime_sec"]
            print(
                f"[benchmark] done in {row['runtime_sec']:.1f} s "
                f"({fmt(row['runtime_per_audio_sec'], 3)} s per audio second), "
                f"peak VRAM {fmt(row['peak_gpu_used_mb'], 0)} MB"
            )

        rows.append(row)
        if stop_after:
            break

    monitor.close()

    csv_path = os.path.join(args.output_folder, "benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )

    if error_log:
        error_path = os.path.join(args.output_folder, "benchmark_errors.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(error_log))
        print(f"\n[benchmark] full tracebacks written to {error_path}")

    summary = build_summary(rows, model_load_sec, args)
    summary_path = os.path.join(args.output_folder, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_table(rows)
    print_summary(summary, args.output_folder)
    print(f"\nPer-case results: {csv_path}\nSummary:          {summary_path}")

    if summary["cases_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
