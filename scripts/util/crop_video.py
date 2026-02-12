import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
import torch

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
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_video_path = os.path.join(input_video_dir, video_file)
        output_video_path = os.path.join(output_video_dir, video_file)
        lmk_file = video_file.replace('.mp4', '.npy')
        
        # Open video and get properties
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        landmarks_path = os.path.join(input_landmarks_dir, lmk_file)
        if not os.path.exists(landmarks_path):
            print(f"Skipping {video_file} because landmarks not found at {landmarks_path}")
            cap.release()
            continue
            
        landmarks = np.load(landmarks_path)
        if len(landmarks.shape) == 2:
            landmarks = landmarks[None, ...]
            
        # Calculate crop data for all frames
        crop_data_container = video_preprocessor.get_crop_data(landmarks[:, :, :2], height, width)
        
        # Internal VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_preprocessor.resize_size, video_preprocessor.resize_size))
        
        processed_landmarks = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if i >= len(landmarks) or i >= len(crop_data_container):
                break

            # Convert BGR to RGB and then to torch tensor [C, H, W]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
            
            cropped_frame, new_lmk, _, _ = video_preprocessor.crop_single_frame(
                frame_tensor, landmarks[i, :, :2], crop_data_container[i]
            )
            
            # Convert back to uint8 BGR for OpenCV writer
            res_frame = cropped_frame.permute(1, 2, 0).byte().numpy()
            res_frame_bgr = cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR)
            out.write(res_frame_bgr)
            
            processed_landmarks.append(new_lmk)
        
        cap.release()
        out.release()
        
        # Process landmarks
        output_landmarks_path = os.path.join(output_landmarks_dir, lmk_file)
        os.makedirs(os.path.dirname(output_landmarks_path), exist_ok=True)
        np.save(output_landmarks_path, np.stack(processed_landmarks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a video and save the output.")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with original videos.')
    parser.add_argument('--video_dir_cropped', type=str, required=True, help='Output directory for 25fps videos.')
    parser.add_argument('--landmarks_dir', type=str, required=True, help='Directory with original landmarks.')
    parser.add_argument('--landmarks_dir_cropped', type=str, required=True, help='Output directory for landmarks.')
    args = parser.parse_args()
    main(args.video_dir, args.video_dir_cropped, args.landmarks_dir, args.landmarks_dir_cropped)