import os
import sys
import subprocess
import argparse
import cv2
import shutil
from pathlib import Path

# Directory structure (per video):
# outputs/single_results/
#   work_{video_stem}/
#     videos/
#       video.mp4                              ← Step 0: 25fps video (original for paste-back)
#       video.npy                              ← Step 1: Landmarks (full-body)
#       video.mp4  (cropped, replaces above)   ← Step 2: Cropped 512x512 face
#       video_video_512_latent.safetensors     ← Step 3: VAE latent
#     landmarks/
#       video.npy                              ← Step 1: Landmarks copy for crop
#     audios/
#       audio.wav                              ← Step 0: 16k mono audio
#       audio_hubert_emb.safetensors           ← Step 4: Hubert embedding
#       audio_wavlm_emb.safetensors            ← Step 4: WavLM embedding
#
# This mirrors the folder layout expected by dubbing_pipeline.py / inference.sh:
#   --video_folder=videos --landmark_folder=landmarks --audio_folder=audios --audio_emb_folder=audios

# Roots
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

# Env for subprocesses
ENV = os.environ.copy()
ENV["PYTHONPATH"] = ROOT_DIR + os.pathsep + ENV.get("PYTHONPATH", "")

def run_cmd(cmd):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=ENV, cwd=ROOT_DIR)

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def main():
    parser = argparse.ArgumentParser(description="End-to-end full body lip-sync pipeline.")
    parser.add_argument("--video", required=True, help="Path to your input full-body video (.mp4)")
    parser.add_argument("--audio", required=True, help="Path to your target audio (.wav/.mp3)")
    parser.add_argument("--keyframes_ckpt", default="pretrained_models/checkpoints/keyframe_dub.pt")
    parser.add_argument("--interpolation_ckpt", default="pretrained_models/checkpoints/interpolation_dub.pt")
    parser.add_argument("--keyframes_config", default="scripts/sampling/configs/keyframe.yaml")
    parser.add_argument("--interpolation_config", default="scripts/sampling/configs/interpolation.yaml")
    parser.add_argument("--output_dir", default="outputs/single_results")
    parser.add_argument("--decoding_t", type=int, default=1, help="Number of frames decoded at a time. Lower to save VRAM.")
    parser.add_argument("--chunk_size", type=int, default=2, help="Inference chunk size. Lower to save RAM.")
    parser.add_argument("--paste_back_to_body", type=str, default="True", help="Stitch face back to original body (True/False)")
    parser.add_argument("--compute_until", type=str, default="end", help="Compute until N seconds or 'end'")
    args = parser.parse_args()

    # Path setup
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.abspath(args.video)
    audio_path = os.path.abspath(args.audio)
    video_stem = Path(video_path).stem

    # Work directory — mirrors the structure expected by dubbing_pipeline / inference.sh
    work_dir = os.path.join(args.output_dir, f"work_{video_stem}")
    # These folder NAMES are what dubbing_pipeline uses for string replacement
    videos_dir = os.path.join(work_dir, "videos")       # cropped video + latent
    videos_orig_dir = os.path.join(work_dir, "videos_orig")  # original 25fps (for paste-back)
    landmarks_dir = os.path.join(work_dir, "landmarks")  # landmarks for cropped
    landmarks_orig_dir = os.path.join(work_dir, "landmarks_orig")  # landmarks for original
    audios_dir = os.path.join(work_dir, "audios")        # audio + embeddings
    for d in [videos_dir, videos_orig_dir, landmarks_dir, landmarks_orig_dir, audios_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"=== Starting Pipeline for: {video_stem} ===")

    # ─── Step 0: Convert video to 25fps, audio to 16k mono ───
    orig_video = os.path.join(videos_orig_dir, "video.mp4")
    fps, width, height = get_video_info(video_path)
    if not os.path.exists(orig_video):
        if abs(fps - 25.0) > 0.1:
            print(f"-> Detected {fps}fps. Converting to 25fps via FFmpeg...")
            run_cmd(["ffmpeg", "-i", video_path, "-r", "25", "-c:v", "libx264", "-crf", "18", "-y", orig_video])
        else:
            print("-> Video is already 25fps, copying...")
            shutil.copy2(video_path, orig_video)
    else:
        print("-> 25fps video already exists, skipping.")

    audio_wav = os.path.join(audios_dir, "audio.wav")
    if not os.path.exists(audio_wav):
        print("-> Converting audio to 16k mono...")
        run_cmd(["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", audio_wav])
    else:
        print("-> 16k audio already exists, skipping.")

    # ─── Step 1: Extract landmarks from original video ───
    orig_lmk = os.path.join(landmarks_orig_dir, "video.npy")
    if not os.path.exists(orig_lmk):
        print("-> Extracting face landmarks...")
        run_cmd([sys.executable, "scripts/util/gen_landmarks.py", videos_orig_dir, "--output_dir", landmarks_orig_dir])
    else:
        print("-> Landmarks already exist, skipping.")

    if not os.path.exists(orig_lmk):
        print(f"[ERROR] Landmarks not found at {orig_lmk}")
        sys.exit(1)

    # ─── Step 2: Crop face region (512x512) ───
    cropped_video = os.path.join(videos_dir, "video.mp4")
    if not os.path.exists(cropped_video):
        print("-> Cropping face region (memory-efficient)...")
        run_cmd([
            sys.executable, "scripts/util/crop_video.py",
            "--video_dir", videos_orig_dir,
            "--video_dir_cropped", videos_dir,
            "--landmarks_dir", landmarks_orig_dir,
            "--landmarks_dir_cropped", landmarks_dir
        ])
    else:
        print("-> Cropped face video already exists, skipping.")

    # ─── Step 3: Compute VAE latents for cropped video ───
    latent_file = os.path.join(videos_dir, "video_video_512_latent.safetensors")
    if not os.path.exists(latent_file):
        print("-> Generating VAE latent vectors...")
        run_cmd([sys.executable, "scripts/util/video_to_latent.py", "--filelist", videos_dir])
    else:
        print("-> VAE latents already exist, skipping.")

    # ─── Step 4: Compute audio embeddings (hubert + wavlm) ───
    # Reference: infer_and_compute_emb.sh uses wavlm + hubert, inference.sh uses --audio_emb_type=hubert
    audio_glob = os.path.join(audios_dir, "*.wav")
    hubert_emb = os.path.join(audios_dir, "audio_hubert_emb.safetensors")
    wavlm_emb = os.path.join(audios_dir, "audio_wavlm_emb.safetensors")

    if not os.path.exists(hubert_emb):
        print("-> Generating hubert audio embeddings...")
        run_cmd([
            sys.executable, "scripts/util/get_audio_embeddings.py",
            "--audio_path", audio_glob,
            "--model_type", "hubert",
            "--skip_video"
        ])
    else:
        print("-> Hubert audio embeddings already exist, skipping.")

    if not os.path.exists(wavlm_emb):
        print("-> Generating wavlm audio embeddings...")
        run_cmd([
            sys.executable, "scripts/util/get_audio_embeddings.py",
            "--audio_path", audio_glob,
            "--model_type", "wavlm",
            "--skip_video"
        ])
    else:
        print("-> WavLM audio embeddings already exist, skipping.")

    # ─── Step 5: Create filelists ───
    filelist_path = os.path.join(work_dir, "filelist.txt")
    filelist_audio_path = os.path.join(work_dir, "filelist_audio.txt")
    with open(filelist_path, "w") as f:
        f.write(cropped_video + "\n")
    with open(filelist_audio_path, "w") as f:
        f.write(audio_wav + "\n")

    # ─── Step 6: Run inference ───
    # Match inference.sh parameters exactly
    paste_back = args.paste_back_to_body.lower() in ("true", "1", "yes")
    final_v_path = os.path.join(args.output_dir, "video.mp4")

    if os.path.exists(final_v_path):
        print(f"\n[INFO] Final output already exists at {final_v_path}.")
        print("Delete it to re-run, or change --output_dir.")
    else:
        print("-> Running inference...")
        cmd = [
            sys.executable, "scripts/sampling/dubbing_pipeline.py",
            "--filelist", filelist_path,
            "--filelist_audio", filelist_audio_path,
            "--keyframes_ckpt", args.keyframes_ckpt,
            "--interpolation_ckpt", args.interpolation_ckpt,
            "--model_config", args.interpolation_config,
            "--model_keyframes_config", args.keyframes_config,
            "--output_folder", args.output_dir,
            "--decoding_t", str(args.decoding_t),
            "--cond_aug", "0.",
            "--resize_size", "512",
            "--chunk_size", str(args.chunk_size),
            # Folder layout (for string replacement in dubbing_pipeline)
            "--video_folder", "videos",
            "--latent_folder", "videos",
            "--landmark_folder", "landmarks",
            "--audio_folder", "audios",
            "--audio_emb_folder", "audios",
            # Model settings (from inference.sh)
            "--audio_emb_type", "hubert",
            "--add_zero_flag", "True",
            "--extra_audio", "None",
            "--compute_until", args.compute_until,
            "--recompute", "True",
            "--what_mask", "box",
        ]
        if paste_back:
            cmd += ["--paste_back_to_body", "True"]
        run_cmd(cmd)

    print(f"\n[SUCCESS] Final output saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
