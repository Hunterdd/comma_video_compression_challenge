"""Evaluate all 3 models at a given training stage and print the combined score.

Usage (from challenge root):
    python submissions/my_submission/eval_stage.py --stage 1
    python submissions/my_submission/eval_stage.py --stage 8  # final score
    python submissions/my_submission/eval_stage.py            # defaults to last stage

Reads:  full_run/model_{k}/{stage_name}/best_archive.bin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import av

HERE           = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent.parent
HNERV_SRC      = HERE.parent / "hnerv_muon" / "src"

sys.path.insert(0, str(CHALLENGE_ROOT))
sys.path.insert(0, str(HNERV_SRC))
sys.path.insert(0, str(HERE))

from codec       import parse_archive                          # noqa: E402
from model       import HNeRVDecoder                           # noqa: E402
from score       import compute_score, total_video_bytes       # noqa: E402
from data        import get_default_video_path                 # noqa: E402
from frame_utils import yuv420_to_rgb                         # noqa: E402

# Stage name list — must match train_full.py
STAGE_NAMES = [
    "s1_ce", "s2_softplus", "s3_smooth", "s4_qat",
    "s5_c1a", "s6_lambda", "s7_sigma", "s8_muon",
]

BASE_CHANNELS    = 20
STEM_DIM         = 14
LATENT_DIM       = 28
EVAL_SIZE        = (384, 512)
N_MODELS         = 3
PAIRS_PER_MODEL  = 100
OUT_DIR          = HERE / "full_run"


@torch.inference_mode()
def evaluate_slice(decoder, latents, distortion_net, video_path,
                   pair_offset, device, batch_pairs=16):
    decoder.eval()
    n_pairs = latents.shape[0]

    container = av.open(str(video_path))
    stream    = container.decode(container.streams.video[0])
    for _ in range(pair_offset * 2):
        next(stream)

    gt_pairs = []
    prev = None
    for frame in stream:
        f = yuv420_to_rgb(frame)
        if prev is None:
            prev = f
            continue
        gt_pairs.append(torch.stack([prev, f]))
        prev = None
        if len(gt_pairs) == n_pairs:
            break
    container.close()

    seg_sum = 0.0; pose_sum = 0.0; count = 0
    for start in range(0, len(gt_pairs), batch_pairs):
        batch_gt = torch.stack(gt_pairs[start:start + batch_pairs]).to(device)
        B = batch_gt.shape[0]
        decoded = decoder(latents[start:start + B])
        flat = decoded.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
        up   = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
        out  = (up.reshape(B, 2, 3, 874, 1164)
                   .permute(0, 1, 3, 4, 2)
                   .clamp(0, 255).round().to(torch.uint8))
        pose_d, seg_d = distortion_net.compute_distortion(batch_gt, out)
        seg_sum  += seg_d.sum().item()
        pose_sum += pose_d.sum().item()
        count    += B

    return seg_sum / max(count, 1), pose_sum / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="Score models at a specific training stage")
    parser.add_argument("--stage", type=int, default=len(STAGE_NAMES),
                        help=f"Stage number 1-{len(STAGE_NAMES)} (default: last)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not 1 <= args.stage <= len(STAGE_NAMES):
        parser.error(f"--stage must be 1-{len(STAGE_NAMES)}")

    stage_name = STAGE_NAMES[args.stage - 1]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Stage   : {args.stage} ({stage_name})")
    print(f"Device  : {device}")

    # Load archives
    archives = {}
    for k in range(N_MODELS):
        path = OUT_DIR / f"model_{k}" / stage_name / "best_archive.bin"
        if not path.exists():
            print(f"  ERROR: {path} not found — run train_full.py first")
            return
        archives[k] = path.read_bytes()
        print(f"  model_{k}: {len(archives[k]):,}B  ({path})")

    video_path = get_default_video_path()
    tvb = total_video_bytes(video_path)
    print(f"\nVideo : {video_path}  ({tvb:,}B)")
    print("Loading distortion net...")

    from data import precompute_targets
    distortion_net, _, _, _, _ = precompute_targets(video_path, device)

    combined_bytes = sum(len(a) for a in archives.values())
    seg_sum = 0.0; pose_sum = 0.0
    print(f"\nEvaluating each model on its video slice...")

    for k in range(N_MODELS):
        eval_sd, eval_lat, _ = parse_archive(archives[k])
        eval_dec = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE,
                                stem_dim=STEM_DIM).to(device)
        eval_dec.load_state_dict(eval_sd)
        seg_d, pose_d = evaluate_slice(eval_dec, eval_lat.to(device), distortion_net,
                                       video_path, pair_offset=k * PAIRS_PER_MODEL, device=device)
        del eval_dec
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  model_{k}: seg={seg_d:.5f}  pose={pose_d:.6f}")
        seg_sum  += seg_d
        pose_sum += pose_d

    avg_seg  = seg_sum  / N_MODELS
    avg_pose = pose_sum / N_MODELS
    result   = compute_score(avg_seg, avg_pose, combined_bytes, tvb)

    print(f"\n{'='*50}")
    print(f"Stage {args.stage} ({stage_name})  combined score: {result['score']:.4f}")
    print(f"  seg_distortion  = {avg_seg:.6f}")
    print(f"  pose_distortion = {avg_pose:.6f}")
    print(f"  archive_bytes   = {combined_bytes:,}  "
          f"({' + '.join(str(len(archives[k])) for k in range(N_MODELS))})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
