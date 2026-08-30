"""Compare two videos frame by frame and report which region actually changed.

Useful for checking a --paste_back run: the output should be pixel-identical to
the input outside the face box, and clearly different inside it. If nothing
changed at all, the pipeline did not touch the frames; if everything changed,
the whole frame was regenerated rather than the crop pasted back.

Usage:
    python scripts/util/diff_videos.py data/videos/clip.mp4 output/clip.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np


def read_frames(path, limit=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {path}")
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original")
    parser.add_argument("generated")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Per-pixel difference counted as a real change (0-255).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only compare the first N frames.")
    args = parser.parse_args()

    a = read_frames(args.original, args.limit)
    b = read_frames(args.generated, args.limit)

    print(f"original : {args.original}  {a.shape[2]}x{a.shape[1]}, {len(a)} frames")
    print(f"generated: {args.generated}  {b.shape[2]}x{b.shape[1]}, {len(b)} frames")

    if a.shape[1:3] != b.shape[1:3]:
        print("\nResolutions differ -> the output was not pasted back into the "
              "original frames (with --paste_back they should match).")
        return

    n = min(len(a), len(b))
    diff = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16)).max(axis=3)
    changed = diff > args.threshold

    total_ratio = changed.mean()
    print(f"\nchanged pixels: {total_ratio * 100:.2f}% of the frame area")

    if not changed.any():
        print("\nNothing changed: the output is identical to the input. The face "
              "was not animated at all.")
        return

    ys, xs = np.where(changed.any(axis=0))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = a.shape[1], a.shape[2]
    print(f"changed region: x[{x0}:{x1}] y[{y0}:{y1}]  "
          f"({x1 - x0}x{y1 - y0} out of {w}x{h})")

    if (x1 - x0) >= w * 0.98 and (y1 - y0) >= h * 0.98:
        print("\nThe whole frame changed -> this looks like a full-frame resize, "
              "not a face crop pasted back.")
    else:
        print("\nOnly a sub-region changed -> the crop was pasted back as expected.")

    per_frame = changed.reshape(n, -1).mean(axis=1)
    print(f"\nper-frame changed area: min {per_frame.min() * 100:.2f}%, "
          f"max {per_frame.max() * 100:.2f}%, mean {per_frame.mean() * 100:.2f}%")
    if per_frame.max() < 0.001:
        print("Changes are negligible - check that the audio embeddings and the "
              "mouth mask are actually being applied.")


if __name__ == "__main__":
    main()
