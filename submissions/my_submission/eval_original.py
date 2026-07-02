"""Run SegNet and PoseNet on the original video and print distortion values.

Usage:
    python submissions/my_submission/eval_original.py
    python submissions/my_submission/eval_original.py --device cpu
"""
import sys
import math
import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent.parent   # comma_video_compression_challenge/
sys.path.insert(0, str(CHALLENGE_ROOT))

import av
from frame_utils import yuv420_to_rgb
from modules import DistortionNet, segnet_sd_path, posenet_sd_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--video", default=None, help="Path to video file (default: auto-detect)")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Video path
    if args.video:
        video_path = Path(args.video)
    else:
        names_file = CHALLENGE_ROOT / "public_test_video_names.txt"
        name = names_file.read_text().strip().splitlines()[0]
        video_path = CHALLENGE_ROOT / "videos" / name
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    print(f"Video: {video_path}")

    # Load frozen DistortionNet
    print("Loading DistortionNet (SegNet + PoseNet)...")
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    # Stream video into pairs, run GT against itself (self-distortion = baseline zero-error)
    # This prints what perfect reconstruction would score on seg/pose.
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    seg_total, pose_total, count = 0.0, 0.0, 0
    prev = None
    batch_gt, batch_comp = [], []

    def flush(batch_gt, batch_comp):
        nonlocal seg_total, pose_total, count
        if not batch_gt:
            return
        gt = torch.stack(batch_gt).to(device)     # (B, 2, H, W, 3) uint8
        comp = torch.stack(batch_comp).to(device)
        with torch.inference_mode():
            pose_d, seg_d = net.compute_distortion(gt, comp)
        seg_total += seg_d.sum().item()
        pose_total += pose_d.sum().item()
        count += gt.shape[0]

    for frame in container.decode(stream):
        f = yuv420_to_rgb(frame)              # (H, W, 3) uint8
        if prev is None:
            prev = f
            continue
        pair = torch.stack([prev, f])          # (2, H, W, 3)
        batch_gt.append(pair)
        batch_comp.append(pair)                # same video = zero distortion reference
        prev = None
        if len(batch_gt) == args.batch:
            flush(batch_gt, batch_comp)
            batch_gt, batch_comp = [], []

    flush(batch_gt, batch_comp)
    container.close()

    if count == 0:
        print("No pairs found.")
        return

    seg_dist = seg_total / count
    pose_dist = pose_total / count
    print(f"\n=== Results over {count} frame pairs ===")
    print(f"  SegNet  distortion : {seg_dist:.8f}")
    print(f"  PoseNet distortion : {pose_dist:.8f}")
    print(f"\n  (Self-distortion — original vs itself — should be 0.0 for both)")


if __name__ == "__main__":
    main()
