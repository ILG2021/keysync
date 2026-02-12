import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from landmarks_extractor import LandmarksExtractor


def process_video(video_path, landmarks_extractor, output_dir, batch_size=32):
    video_filename = os.path.basename(video_path).replace(".mp4", ".npy")
    output_path = os.path.join(output_dir, video_filename)

    if os.path.exists(output_path):
        print(f"Landmarks already exist for {video_path}, skipping...")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize list to store landmarks for all frames
    all_landmarks = []

    # Process frames in batches
    frames_batch = []
    for _ in tqdm(range(total_frames), desc=f"Processing {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_batch.append(frame_rgb)

        # Process batch when it reaches the batch size or at the end
        if len(frames_batch) == batch_size or _ == total_frames - 1:
            # Convert list of frames to numpy array
            frames_array = torch.from_numpy(np.array(frames_batch)).permute(0, 3, 1, 2)

            # Extract landmarks for the batch
            batch_landmarks = landmarks_extractor.extract_landmarks(frames_array)

            # Check if landmarks were detected for each frame
            for i in range(len(batch_landmarks)):
                # If no face detected or landmarks are None
                if batch_landmarks[i] is None or len(batch_landmarks[i]) == 0:
                    # Find the closest valid landmark (before or after)
                    valid_indices = [
                        j
                        for j, lm in enumerate(all_landmarks)
                        if lm is not None and len(lm) > 0
                    ]
                    if valid_indices:
                        closest_idx = min(
                            valid_indices,
                            key=lambda idx: abs(idx - len(all_landmarks) - i),
                        )
                        batch_landmarks[i] = all_landmarks[closest_idx]
                    elif (
                        len(batch_landmarks) > i + 1
                        and batch_landmarks[i + 1] is not None
                        and len(batch_landmarks[i + 1]) > 0
                    ):
                        # Use the next valid landmark in the current batch if available
                        next_valid_idx = next(
                            (
                                j
                                for j in range(i + 1, len(batch_landmarks))
                                if batch_landmarks[j] is not None
                                and len(batch_landmarks[j]) > 0
                            ),
                            None,
                        )
                        if next_valid_idx is not None:
                            batch_landmarks[i] = batch_landmarks[next_valid_idx]
                        else:
                            # If no valid landmarks found, set an error
                            raise ValueError(
                                f"Error: No valid landmarks found for frame {_ - len(frames_batch) + i + 1} in {video_path}"
                            )
                            batch_landmarks[i] = np.zeros((68, 2))
                    else:
                        # If no valid landmarks found, set an error
                        raise ValueError(
                            f"Error: No valid landmarks found for frame {_ - len(frames_batch) + i + 1} in {video_path}"
                        )
                        batch_landmarks[i] = np.zeros((68, 2))

                # If multiple faces detected, use the first one
                if isinstance(batch_landmarks[i], list) and len(batch_landmarks[i]) > 0:
                    batch_landmarks[i] = batch_landmarks[i][0]

                # Ensure the landmark has the correct shape (68x2)
                if batch_landmarks[i].shape != (68, 2):
                    # If the shape is wrong, use the closest valid landmark or set an error
                    valid_indices = [
                        j
                        for j, lm in enumerate(all_landmarks)
                        if lm is not None
                        and hasattr(lm, "shape")
                        and lm.shape == (68, 2)
                    ]
                    if valid_indices:
                        closest_idx = min(
                            valid_indices,
                            key=lambda idx: abs(idx - len(all_landmarks) - i),
                        )
                        batch_landmarks[i] = all_landmarks[closest_idx]
                    else:
                        raise ValueError(
                            f"Error: No valid landmarks with shape (68, 2) found for frame {_ - len(frames_batch) + i + 1} in {video_path}"
                        )
                        batch_landmarks[i] = np.zeros((68, 2))

            all_landmarks.extend(batch_landmarks)

            # Clear the batch
            frames_batch = []

    # Release video capture
    cap.release()

    # Save landmarks
    np.save(output_path, np.array(all_landmarks))
    print(f"Saved landmarks to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract facial landmarks from videos")
    parser.add_argument("video_dir", help="Directory containing video files")
    parser.add_argument(
        "--output_dir", default="landmarks", help="Directory to save landmarks"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run face alignment on (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for processing frames"
    )
    args = parser.parse_args()

    # Initialize landmarks extractor
    landmarks_extractor = LandmarksExtractor(device=args.device)

    # Process all video files in the directory
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    total_errors = 0
    for filename in os.listdir(args.video_dir):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(args.video_dir, filename)
            try:
                process_video(
                    video_path, landmarks_extractor, args.output_dir, args.batch_size
                )
            except Exception as e:
                total_errors += 1
                print(f"Error processing {video_path}: {e}")

    if total_errors > 0:
        print(f"Finished processing videos with {total_errors} errors out of {len(os.listdir(args.video_dir))} files.")


if __name__ == "__main__":
    main()
