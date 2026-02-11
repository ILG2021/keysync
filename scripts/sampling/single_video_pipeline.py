import os
import sys
import subprocess
import argparse
import cv2
import shutil
from pathlib import Path

# Roots
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

# 设置 Python 路径环境，确保子进程能找到项目模块
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
    parser.add_argument("--keyframes_ckpt", default="pretrained_models/checkpoints/keyframes.ckpt")
    parser.add_argument("--interpolation_ckpt", default="pretrained_models/checkpoints/interpolation.ckpt")
    parser.add_argument("--output_dir", default="outputs/single_results")
    parser.add_argument("--decoding_t", type=int, default=14, help="Lower if OOM (default 14)")
    args = parser.parse_args()

    # 路径初始化
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.abspath(args.video)
    audio_path = os.path.abspath(args.audio)
    video_stem = Path(video_path).stem
    
    # 建立独立的工作子目录
    work_dir = os.path.join(args.output_dir, f"work_{video_stem}")
    video_in_dir = os.path.join(work_dir, "input_v")
    video_crop_dir = os.path.join(work_dir, "crop_v")
    os.makedirs(video_in_dir, exist_ok=True)
    os.makedirs(video_crop_dir, exist_ok=True)

    print(f"=== Starting Pipeline for: {video_stem} ===")

    # 1. 检查并转换视频 FPS (25) & 音频采样 (16k)
    fps, width, height = get_video_info(video_path)
    current_v_25 = os.path.join(video_in_dir, "video.mp4")
    
    if abs(fps - 25.0) > 0.1:
        print(f"-> Detected {fps}fps. Converting to 25fps via FFmpeg...")
        run_cmd(["ffmpeg", "-i", video_path, "-r", "25", "-c:v", "libx264", "-crf", "18", "-y", current_v_25])
    else:
        if not os.path.exists(current_v_25):
            shutil.copy2(video_path, current_v_25)

    current_a_16k = os.path.join(work_dir, "audio_16k.wav")
    print("-> Converting audio to 16k mono...")
    run_cmd(["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", current_a_16k])

    # 2. 提取全视频关键点 (Landmarks)
    print("-> Extracting face landmarks...")
    run_cmd([sys.executable, "scripts/util/gen_landmarks.py", video_in_dir, "--output_dir", video_in_dir])
    
    # 3. 裁剪面部 (512x512) 用于 Embedding 计算
    print("-> Cropping face region (memory-efficient)...")
    run_cmd([
        sys.executable, "scripts/util/crop_video.py",
        "--video_dir", video_in_dir,
        "--video_dir_cropped", video_crop_dir,
        "--landmarks_dir", video_in_dir,
        "--landmarks_dir_cropped", video_crop_dir
    ])
    cropped_v_path = os.path.join(video_crop_dir, "video.mp4")

    # 4. 生成 Latents (Embedding)
    print("-> Generating VAE latent vectors for the face...")
    run_cmd([sys.executable, "scripts/util/video_to_latent.py", "--filelist", cropped_v_path])

    # 5. 执行推理与缝合 (Paste-back)
    print("-> Running inference and stitching back to body...")
    run_cmd([
        sys.executable, "scripts/sampling/dubbing_pipeline.py",
        "--filelist", current_v_25,
        "--filelist_audio", current_a_16k,
        "--keyframes_ckpt", args.keyframes_ckpt,
        "--interpolation_ckpt", args.interpolation_ckpt,
        "--output_folder", args.output_dir,
        "--paste_back_to_body", "True",
        "--recompute", "True",
        "--decoding_t", str(args.decoding_t),
        "--what_mask", "box"
    ])

    print(f"\n[SUCCESS] Final output saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
