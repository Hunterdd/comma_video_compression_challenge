"""Structured channel pruning diagnostic for HNeRV decoder.

Phase 1 only — NO fine-tuning.

For each pruning ratio p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
  1. Load model/0.bin (parse_archive -> decoder_sd, latents, meta)
  2. Collect mean-abs activations per output channel across all 600 latents
  3. Zero the bottom-p fraction of channels in each decoder block
  4. Evaluate seg/pose distortion without any training
  5. Compute compressed archive size for the pruned model
  6. Print and save the sensitivity curve

Run from the repo root:
  python submissions/my_submission/prune_diagnostic.py

Output:
  submissions/my_submission/pruning_curve.txt
"""
from __future__ import annotations

import math
import sys
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: locate the challenge root and the hnerv_muon src directory
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent.parent       # comma_video_compression_challenge/
HNERV_SRC = HERE.parent / "hnerv_muon" / "src"

sys.path.insert(0, str(CHALLENGE_ROOT))
sys.path.insert(0, str(HNERV_SRC))

# hnerv_muon src imports (codec, model, score)
from codec import parse_archive, build_archive, quantize_state_dict  # noqa: E402
from model import HNeRVDecoder                                        # noqa: E402
from score import evaluate_decoder, compute_score, total_video_bytes  # noqa: E402

# Challenge root imports (DistortionNet, video path)
from modules import DistortionNet, segnet_sd_path, posenet_sd_path    # noqa: E402
from frame_utils import camera_size                                    # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARCHIVE_PATH = HERE / "model" / "0.bin"
EVAL_SIZE = (384, 512)   # (H, W) — decoder native output, matches hnerv_muon
PRUNE_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BATCH_PAIRS = 8          # pairs per eval batch (memory-safe on CPU/MPS)


# ---------------------------------------------------------------------------
# DistortionNet loader (frozen)
# ---------------------------------------------------------------------------
def load_distortion_net(device: torch.device) -> DistortionNet:
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_channel_activations(decoder: HNeRVDecoder,
                                 latents: torch.Tensor,
                                 device: torch.device,
                                 batch_size: int = 32) -> list[torch.Tensor]:
    """Return mean-abs activation per output channel for each PixelShuffle
    block output (post-shuffle, post-sin), shape [n_blocks, out_ch].

    We register forward hooks on the PixelShuffle output (i.e., after ps())
    before the sin residual add) to get the natural channel importance signal.
    """
    decoder.eval().to(device)
    latents = latents.to(device)
    n_pairs = latents.shape[0]

    # We'll accumulate sum-abs and count per block per channel
    # Hook captures the pixel-shuffled output before sin() is applied
    # (The forward loop in HNeRVDecoder: ps(block(x)) then sin(... + identity))
    n_blocks = len(decoder.blocks)

    # sums[i]: shape (out_ch,)  — accumulated |activation|
    # counts[i]: number of (B, H, W) elements seen
    sums = [None] * n_blocks
    counts = [0] * n_blocks

    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            # out is the PixelShuffle output: (B, out_ch, H, W)
            with torch.no_grad():
                abs_mean = out.abs().mean(dim=(0, 2, 3))  # (out_ch,)
                if sums[idx] is None:
                    sums[idx] = abs_mean.cpu()
                else:
                    sums[idx] = sums[idx] + abs_mean.cpu()
                counts[idx] += 1
        return hook

    # PixelShuffle is shared (decoder.ps); we need per-block hooks.
    # Instead, hook the block Conv2d outputs *before* ps, then track ps input.
    # Simpler: re-implement the forward partially to capture post-ps values.
    # We use a wrapper approach: hook decoder.ps and track which block we're in.

    block_call_counter = [0]

    def ps_hook(module, inp, out):
        idx = block_call_counter[0] % n_blocks
        with torch.no_grad():
            abs_mean = out.abs().mean(dim=(0, 2, 3))  # (out_ch,)
            if sums[idx] is None:
                sums[idx] = abs_mean.cpu()
            else:
                sums[idx] = sums[idx] + abs_mean.cpu()
            counts[idx] += 1
        block_call_counter[0] += 1

    handles.append(decoder.ps.register_forward_hook(ps_hook))

    try:
        for start in range(0, n_pairs, batch_size):
            z = latents[start: start + batch_size]
            _ = decoder(z)
    finally:
        for h in handles:
            h.remove()

    # Normalise by number of forward calls per block
    mean_abs = []
    for i in range(n_blocks):
        if sums[i] is None:
            raise RuntimeError(f"Block {i} hook never fired")
        mean_abs.append(sums[i] / counts[i])  # (out_ch,)

    return mean_abs


# ---------------------------------------------------------------------------
# Channel zeroing (structured pruning — in-place on a deep-copied state dict)
# ---------------------------------------------------------------------------
def prune_state_dict(sd: dict, mean_abs: list[torch.Tensor],
                     prune_ratio: float,
                     channels: list[int]) -> dict:
    """Zero channels in-place on a copy of sd.

    For block i (Conv2d(in_ch, out_ch*4, 3) → PixelShuffle(2) → out_ch):
      - Rank the out_ch post-shuffle channels by mean_abs[i].
      - Bottom p% channels to prune → zero those 4 consecutive pre-shuffle
        filters (4 sub-pixel positions per output channel).
      - Also zero the matching input channels on block i+1 and skip i+1.
    """
    import copy
    sd = copy.deepcopy(sd)

    n_blocks = len(mean_abs)

    for i in range(n_blocks):
        out_ch = channels[i + 1]  # post-shuffle output channels for block i
        n_prune = max(0, int(out_ch * prune_ratio))
        if n_prune == 0:
            continue

        # Channels sorted worst-first
        sorted_idx = torch.argsort(mean_abs[i])  # ascending abs — weakest first
        prune_ch = sorted_idx[:n_prune].tolist()

        # Key for this block's Conv2d weight: blocks.{i}.weight shape (out_ch*4, in_ch, 3, 3)
        k_block = f"blocks.{i}.weight"
        if k_block not in sd:
            print(f"  [warn] key {k_block} not in state dict — skipping")
            continue

        w = sd[k_block]  # (out_ch*4, in_ch, 3, 3)
        # Each post-shuffle channel c corresponds to 4 pre-shuffle filters:
        # positions c*4, c*4+1, c*4+2, c*4+3  (PixelShuffle(2) layout)
        for c in prune_ch:
            for s in range(4):
                idx = c * 4 + s
                if idx < w.shape[0]:
                    w[idx] = 0.0
        sd[k_block] = w

        # Zero bias if present
        k_bias = f"blocks.{i}.bias"
        if k_bias in sd:
            b = sd[k_bias]
            for c in prune_ch:
                for s in range(4):
                    idx = c * 4 + s
                    if idx < b.shape[0]:
                        b[idx] = 0.0
            sd[k_bias] = b

        # Propagate to the next block's input channels (block i+1)
        if i + 1 < n_blocks:
            k_next = f"blocks.{i+1}.weight"
            if k_next in sd:
                w_next = sd[k_next]  # (out_ch_next*4, in_ch_next, 3, 3)
                for c in prune_ch:
                    if c < w_next.shape[1]:
                        w_next[:, c, :, :] = 0.0
                sd[k_next] = w_next

        # Propagate to the skip connection for block i (skip.weight or Identity)
        k_skip = f"skips.{i}.weight"
        if k_skip in sd:
            w_skip = sd[k_skip]  # (out_ch, in_ch, 1, 1)
            for c in prune_ch:
                if c < w_skip.shape[0]:
                    w_skip[c] = 0.0
            sd[k_skip] = w_skip
            k_skip_b = f"skips.{i}.bias"
            if k_skip_b in sd:
                b_skip = sd[k_skip_b]
                for c in prune_ch:
                    if c < b_skip.shape[0]:
                        b_skip[c] = 0.0
                sd[k_skip_b] = b_skip

    return sd


# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Loading archive: {ARCHIVE_PATH}")

    with open(ARCHIVE_PATH, "rb") as f:
        archive_bytes = f.read()

    decoder_sd, latents, meta = parse_archive(archive_bytes)
    n_pairs = latents.shape[0]
    latent_dim = meta.get("latent_dim", 28)
    base_channels = meta.get("base_channels", 36)
    eval_size = tuple(meta.get("eval_size", [384, 512]))

    print(f"  n_pairs={n_pairs}, latent_dim={latent_dim}, base_channels={base_channels}, "
          f"eval_size={eval_size}, original_archive={len(archive_bytes):,} bytes")

    # Load distortion net
    print("Loading DistortionNet (frozen)...")
    distortion_net = load_distortion_net(device)

    # Locate video file
    videos_dir = CHALLENGE_ROOT / "videos"
    names_file = CHALLENGE_ROOT / "public_test_video_names.txt"
    with open(names_file) as f:
        video_name = f.readline().strip()
    video_path = videos_dir / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    print(f"Video: {video_path}")

    total_vb = total_video_bytes(video_path)
    print(f"Original video bytes (rate denominator): {total_vb:,}")

    # Build baseline decoder and collect activations
    print("\nBuilding baseline decoder and collecting channel activations over all "
          f"{n_pairs} pairs...")
    decoder_base = HNeRVDecoder(latent_dim=latent_dim, base_channels=base_channels,
                                eval_size=eval_size).to(device)
    decoder_base.load_state_dict(decoder_sd)
    decoder_base.eval()
    latents_dev = latents.to(device)

    mean_abs = collect_channel_activations(decoder_base, latents_dev, device,
                                           batch_size=32)
    print(f"  Collected activations for {len(mean_abs)} blocks")
    for i, ma in enumerate(mean_abs):
        print(f"    block {i}: {ma.shape[0]} channels, "
              f"min={ma.min():.4f} max={ma.max():.4f} mean={ma.mean():.4f}")

    # Baseline eval (p=0)
    print("\n--- Baseline (p=0.00) ---")
    decoder_base.eval()
    dist_base = evaluate_decoder(decoder_base, latents_dev, distortion_net,
                                 video_path, batch_pairs=BATCH_PAIRS, device=device)

    # Baseline archive size (what we already have)
    baseline_archive_size = len(archive_bytes)
    score_base = compute_score(dist_base['seg_distortion'], dist_base['pose_distortion'],
                               baseline_archive_size, total_vb)
    print(f"  seg={dist_base['seg_distortion']:.6f}  "
          f"pose={dist_base['pose_distortion']:.6f}  "
          f"archive={baseline_archive_size:,}  score={score_base['score']:.4f}")

    # Derive the channel schedule directly from the actual model's block weights
    # (avoids off-by-one from int() truncation, e.g. int(36*0.58)=20 not 21)
    channels = [decoder_base.channels[i] for i in range(len(decoder_base.channels))]

    # Sweep
    rows = [
        ("ratio", "seg_dist", "pose_dist", "archive_bytes", "score",
         "seg_vs_base", "score_vs_base"),
        (0.00, dist_base['seg_distortion'], dist_base['pose_distortion'],
         baseline_archive_size, score_base['score'], 0.0, 0.0),
    ]

    print(f"\n{'ratio':>6}  {'seg_dist':>10}  {'pose_dist':>10}  "
          f"{'archive':>10}  {'score':>8}  {'Δseg':>9}  {'Δscore':>8}")
    print(f"{'-----':>6}  {'--------':>10}  {'---------':>10}  "
          f"{'-------':>10}  {'-----':>8}  {'----':>9}  {'------':>8}")
    print(f"{'0.00':>6}  {dist_base['seg_distortion']:>10.6f}  "
          f"{dist_base['pose_distortion']:>10.6f}  "
          f"{baseline_archive_size:>10,}  {score_base['score']:>8.4f}  "
          f"{'0.000000':>9}  {'0.0000':>8}")

    for p in PRUNE_RATIOS:
        print(f"\n--- Pruning ratio p={p:.2f} ---")

        pruned_sd = prune_state_dict(decoder_sd, mean_abs, p, channels)

        decoder_pruned = HNeRVDecoder(latent_dim=latent_dim, base_channels=base_channels,
                                      eval_size=eval_size).to(device)
        decoder_pruned.load_state_dict(pruned_sd)
        decoder_pruned.eval()

        dist = evaluate_decoder(decoder_pruned, latents_dev, distortion_net,
                                video_path, batch_pairs=BATCH_PAIRS, device=device)

        # Repack archive with pruned weights (same latents)
        pruned_archive = build_archive(
            pruned_sd, latents,
            meta_dict={"n_pairs": n_pairs, "latent_dim": latent_dim,
                       "base_channels": base_channels, "eval_size": list(eval_size)},
        )
        pruned_archive_size = len(pruned_archive)

        score_pruned = compute_score(dist['seg_distortion'], dist['pose_distortion'],
                                     pruned_archive_size, total_vb)

        delta_seg = dist['seg_distortion'] - dist_base['seg_distortion']
        delta_score = score_pruned['score'] - score_base['score']

        print(f"  seg={dist['seg_distortion']:.6f}  "
              f"pose={dist['pose_distortion']:.6f}  "
              f"archive={pruned_archive_size:,}  "
              f"score={score_pruned['score']:.4f}  "
              f"Δseg={delta_seg:+.6f}  Δscore={delta_score:+.4f}")

        rows.append((p, dist['seg_distortion'], dist['pose_distortion'],
                     pruned_archive_size, score_pruned['score'], delta_seg, delta_score))

        print(f"  {'p':>6}  {dist['seg_distortion']:>10.6f}  "
              f"{dist['pose_distortion']:>10.6f}  "
              f"{pruned_archive_size:>10,}  {score_pruned['score']:>8.4f}  "
              f"{delta_seg:>+9.6f}  {delta_score:>+8.4f}")

        del decoder_pruned

    # Write curve file
    out_path = HERE / "pruning_curve.txt"
    with open(out_path, "w") as f:
        header = rows[0]
        f.write("\t".join(str(h) for h in header) + "\n")
        for row in rows[1:]:
            f.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                              for v in row) + "\n")

    print(f"\nSensitivity curve written to: {out_path}")
    print("\nInterpretation guide:")
    print("  Δseg < +0.0003  → channel was redundant, graceful degradation")
    print("  Δseg 0.0003–0.003 → recoverable gap, consider fine-tuning at this ratio")
    print("  Δseg > 0.003   → cliff, too aggressive or model not prunable")


if __name__ == "__main__":
    main()
