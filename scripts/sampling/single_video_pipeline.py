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
    parser.add_argument("--keyframes_ckpt", default="pretrained_models/checkpoints/keyframe_dub.pt")
    parser.add_argument("--interpolation_ckpt", default="pretrained_models/checkpoints/interpolation_dub.pt")
    parser.add_argument("--keyframes_config", default="configs/example_training/keyframes/keyframes_dub.yaml")
    parser.add_argument("--interpolation_config", default="configs/example_training/interpolation/interpolation_dub.yaml")
    parser.add_argument("--output_dir", default="outputs/single_results")
    parser.add_argument("--decoding_t", type=int, default=4, help="Number of frames decoded at a time. Lower (e.g., 2-4) to save VRAM.")
    parser.add_argument("--chunk_size", type=int, default=4, help="Inference chunk size. Lower (e.g., 4) to save RAM.")
    parser.add_argument("--paste_back_to_body", type=str, default="True", help="Whether to stitch the face back to the original video (True/False)")
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
        if not os.path.exists(current_v_25):
            print(f"-> Detected {fps}fps. Converting to 25fps via FFmpeg...")
            run_cmd(["ffmpeg", "-i", video_path, "-r", "25", "-c:v", "libx264", "-crf", "18", "-y", current_v_25])
        else:
            print("-> 25fps video already exists, skipping conversion.")
    else:
        if not os.path.exists(current_v_25):
            shutil.copy2(video_path, current_v_25)

    current_a_16k = os.path.join(work_dir, "audio_16k.wav")
    if not os.path.exists(current_a_16k):
        print("-> Converting audio to 16k mono...")
        run_cmd(["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", current_a_16k])
    else:
        print("-> 16k audio already exists, skipping conversion.")

    # 2. 提取全视频关键点 (Landmarks)
    # NOTE: gen_landmarks.py uses string replacement for output path:
    #   output = video_path.replace(parent_dir_name, output_dir)
    # So output_dir must be a simple dir name (e.g. "input_v"), NOT a full path.
    video_in_dir_name = os.path.basename(video_in_dir)  # "input_v"
    v_lmk_path = os.path.join(video_in_dir, "video.npy")
    if not os.path.exists(v_lmk_path):
        print("-> Extracting face landmarks...")
        run_cmd([sys.executable, "scripts/util/gen_landmarks.py", video_in_dir, "--output_dir", video_in_dir_name])
    else:
        print("-> Landmarks already exist, skipping extraction.")
    
    # Safety check: landmarks must exist to continue
    if not os.path.exists(v_lmk_path):
        print(f"[ERROR] Landmarks file not found at {v_lmk_path}")
        print("gen_landmarks.py may have failed. Check its output above.")
        sys.exit(1)
    
    # 3. 裁剪面部 (512x512) 用于 Embedding 计算
    cropped_v_path = os.path.join(video_crop_dir, "video.mp4")
    if not os.path.exists(cropped_v_path):
        print("-> Cropping face region (memory-efficient)...")
        run_cmd([
            sys.executable, "scripts/util/crop_video.py",
            "--video_dir", video_in_dir,
            "--video_dir_cropped", video_crop_dir,
            "--landmarks_dir", video_in_dir,
            "--landmarks_dir_cropped", video_crop_dir
        ])
    else:
        print("-> Cropped face video already exists, skipping crop.")

    # 4. 生成 Latents (Embedding)
    # video_to_latent.py has internal skip logic
    print("-> Generating VAE latent vectors for the face...")
    run_cmd([sys.executable, "scripts/util/video_to_latent.py", "--filelist", cropped_v_path])

    # 5. 执行推理与缝合 (Paste-back)
    # dubbing_pipeline expects:
    #   --filelist: the video to process (cropped for face-only, or original for paste-back)
    #   --video_folder / --landmark_folder / --latent_folder: to locate .npy and .safetensors
    # When paste_back_to_body=True, it needs the ORIGINAL video as --filelist
    #   and the landmarks/latents from the cropped version.
    # When paste_back_to_body=False, it uses the cropped video directly.
    
    paste_back = args.paste_back_to_body.lower() in ("true", "1", "yes")
    
    final_v_path = os.path.join(args.output_dir, "video.mp4")
    if os.path.exists(final_v_path):
        print(f"\n[INFO] Final output already exists at {final_v_path}.")
        print("If you want to re-run, delete the output file or change output_dir.")
    else:
        print("-> Running inference and stitching back to body...")
        
        if paste_back:
            # For paste-back: feed original video, landmarks from input_v, latents from crop_v
            run_cmd([
                sys.executable, "scripts/sampling/dubbing_pipeline.py",
                "--filelist", current_v_25,
                "--filelist_audio", current_a_16k,
                "--keyframes_ckpt", args.keyframes_ckpt,
                "--interpolation_ckpt", args.interpolation_ckpt,
                "--model_config", args.interpolation_config,
                "--model_keyframes_config", args.keyframes_config,
                "--output_folder", args.output_dir,
                "--video_folder", video_in_dir,
                "--landmark_folder", video_in_dir,
                "--latent_folder", video_crop_dir,
                "--paste_back_to_body", "True",
                "--recompute", "True",
                "--decoding_t", str(args.decoding_t),
                "--chunk_size", str(args.chunk_size),
                "--what_mask", "box"
            ])
        else:
            # For face-only: feed cropped video directly
            run_cmd([
                sys.executable, "scripts/sampling/dubbing_pipeline.py",
                "--filelist", cropped_v_path,
                "--filelist_audio", current_a_16k,
                "--keyframes_ckpt", args.keyframes_ckpt,
                "--interpolation_ckpt", args.interpolation_ckpt,
                "--model_config", args.interpolation_config,
                "--model_keyframes_config", args.keyframes_config,
                "--output_folder", args.output_dir,
                "--video_folder", video_crop_dir,
                "--landmark_folder", video_crop_dir,
                "--latent_folder", video_crop_dir,
                "--paste_back_to_body", "False",
                "--recompute", "True",
                "--decoding_t", str(args.decoding_t),
                "--chunk_size", str(args.chunk_size),
                "--what_mask", "box"
            ])

    print(f"\n[SUCCESS] Final output saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
