from torchvision.io import read_video
import argparse
import os
import sys
from torchvision.io import write_video
from einops import rearrange
import numpy as np
from tqdm import tqdm

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from scripts.util.video_processor import VideoPreProcessor  # noqa


def main(input_video_dir, output_video_dir, input_landmarks_dir, output_landmarks_dir):
    """
    Preprocess videos and landmarks, converting videos to 25fps and saving landmarks.
    """
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    

    video_files = [f for f in os.listdir(input_video_dir) if f.endswith('.mp4')]

    video_preprocessor = VideoPreProcessor()
    
    for video_file in tqdm(video_files, desc="Processing videos", total=len(video_files)):
        input_video_path = os.path.join(input_video_dir, video_file)
        output_video_path = os.path.join(output_video_dir, video_file)

        video_parent_dir = os.path.basename(os.path.dirname(input_video_path))
        
        # Read the video
        video, _, info = read_video(input_video_path, output_format="TCHW")

        landmarks = np.load(input_video_path.replace('.mp4', '.npy').replace(video_parent_dir, input_landmarks_dir))
        
        video_preprocessor_output = video_preprocessor(video, landmarks)
        cropped_video = video_preprocessor_output.video
        landmarks = video_preprocessor_output.landmarks
        
        # Save the processed video
        write_video(
            output_video_path,
            rearrange(cropped_video, "t c h w -> t h w c"),
            fps=info["video_fps"],
            video_codec="libx264",
        )
        
        # Process landmarks
        output_landmarks_path = input_video_path.replace('.mp4', '.npy').replace(video_parent_dir, output_landmarks_dir)

        os.makedirs(os.path.dirname(output_landmarks_path), exist_ok=True)
        
        np.save(output_landmarks_path, landmarks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a video and save the output.")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with original videos.')
    parser.add_argument('--video_dir_cropped', type=str, required=True, help='Output directory for 25fps videos.')
    parser.add_argument('--landmarks_dir', type=str, required=True, help='Directory with original landmarks.')
    parser.add_argument('--landmarks_dir_cropped', type=str, required=True, help='Output directory for landmarks.')
    args = parser.parse_args()
    main(args.video_dir, args.video_dir_cropped, args.landmarks_dir, args.landmarks_dir_cropped)