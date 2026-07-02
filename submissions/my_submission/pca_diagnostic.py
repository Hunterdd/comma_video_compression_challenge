"""PCA latent compression diagnostic.

Idea: instead of storing 28 floats per frame pair, store k PCA coefficients
(k < 28) plus a shared 28×k basis. At inflate time, reconstruct 28d latents
from coefficients and decode normally.

This script:
  1. Loads model/0.bin (decoder weights + 600×28 latents)
  2. Fits PCA on the 600×28 latent matrix (using torch SVD, no sklearn)
  3. Sweeps k in [16, 17, 18, 19, 20, 22, 24, 28]
     - k=28 is the original (baseline, should reproduce exactly)
  4. For each k:
     - Project → k coefficients per pair, reconstruct → 28d approximation
     - Evaluate seg/pose distortion (original decoder weights, approx latents)
     - Estimate compressed archive size: decoder blob (same) + PCA blob (basis+mean+coefficients → brotli)
     - Print score
  5. Writes pca_curve.txt

Run from repo root:
    python submissions/my_submission/pca_diagnostic.py

Output:
    submissions/my_submission/pca_curve.txt
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import brotli
import torch
import torch.nn.functional as F
import numpy as np

HERE = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent.parent
HNERV_SRC = HERE.parent / "hnerv_muon" / "src"

sys.path.insert(0, str(CHALLENGE_ROOT))
sys.path.insert(0, str(HNERV_SRC))

from codec import parse_archive, encode_decoder, quantize_state_dict  # noqa: E402
from model import HNeRVDecoder                                        # noqa: E402
from score import evaluate_decoder, compute_score, total_video_bytes  # noqa: E402
from modules import DistortionNet, segnet_sd_path, posenet_sd_path    # noqa: E402

ARCHIVE_PATH = HERE / "model" / "0.bin"
BATCH_PAIRS = 8
K_VALUES = [16, 17, 18, 19, 20, 22, 24, 28]   # 28 = original baseline


# ---------------------------------------------------------------------------
# PCA helpers (torch SVD, no sklearn dependency)
# ---------------------------------------------------------------------------

def fit_pca(latents: torch.Tensor):
    """Fit PCA on (N, D) float tensor.  Returns (mean, basis).
    basis shape: (D, D) — columns are principal components, descending variance.
    """
    mean = latents.mean(dim=0)           # (D,)
    centered = latents - mean.unsqueeze(0)
    # economy SVD: U (N,D), S (D,), Vh (D,D)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    basis = Vh.T                         # (D, D), columns = PCs
    return mean, basis


def project(latents: torch.Tensor, mean: torch.Tensor,
            basis: torch.Tensor, k: int) -> torch.Tensor:
    """Project to k-dim PCA space. Returns coefficients (N, k)."""
    centered = latents - mean.unsqueeze(0)
    return centered @ basis[:, :k]      # (N, k)


def reconstruct(coeffs: torch.Tensor, mean: torch.Tensor,
                basis: torch.Tensor, k: int) -> torch.Tensor:
    """Reconstruct 28d latents from (N, k) coefficients."""
    return coeffs @ basis[:, :k].T + mean.unsqueeze(0)   # (N, D)


# ---------------------------------------------------------------------------
# PCA blob encoder (basis + mean + coefficients → brotli bytes)
# ---------------------------------------------------------------------------

def encode_pca_blob(mean: torch.Tensor, basis_k: torch.Tensor,
                    coeffs: torch.Tensor) -> bytes:
    """Pack PCA data into a brotli-compressed blob.

    Layout (all little-endian):
      k: uint32
      D: uint32
      mean: D × float32
      basis_k: D × k × float32   (column-major = each PC is contiguous)
      coefficients: N × k × float32

    Returns brotli-compressed bytes.
    """
    k = coeffs.shape[1]
    D = mean.shape[0]
    N = coeffs.shape[0]

    buf = io.BytesIO()
    buf.write(struct.pack("<II", k, D))
    buf.write(mean.cpu().float().numpy().tobytes())
    buf.write(basis_k.cpu().float().numpy().tobytes())   # (D, k)
    buf.write(coeffs.cpu().float().numpy().tobytes())    # (N, k)
    return brotli.compress(buf.getvalue(), quality=11)


def pca_blob_size(mean, basis_k, coeffs) -> int:
    return len(encode_pca_blob(mean, basis_k, coeffs))


# ---------------------------------------------------------------------------
# Full archive size = decoder blob (constant) + PCA latent blob
# ---------------------------------------------------------------------------

def decoder_blob_bytes(decoder_sd: dict) -> int:
    """Compressed decoder-only blob size (same for all k)."""
    from codec import encode_decoder, quantize_state_dict
    q_sd = quantize_state_dict(decoder_sd)
    return len(encode_decoder(q_sd))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading {ARCHIVE_PATH}")
    with open(ARCHIVE_PATH, "rb") as f:
        raw = f.read()
    decoder_sd, latents, meta = parse_archive(raw)
    n_pairs, latent_dim = latents.shape
    eval_size = tuple(meta.get("eval_size", [384, 512]))
    base_channels = meta.get("base_channels", 36)
    print(f"  latents: {latents.shape}  archive: {len(raw):,} bytes")

    # Load frozen eval net
    print("Loading DistortionNet...")
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for p in net.parameters():
        p.requires_grad_(False)

    # Video
    names_file = CHALLENGE_ROOT / "public_test_video_names.txt"
    video_path = CHALLENGE_ROOT / "videos" / names_file.read_text().strip().splitlines()[0]
    print(f"Video: {video_path}")
    total_vb = total_video_bytes(video_path)

    # Frozen decoder (weights never change across k sweep)
    decoder = HNeRVDecoder(latent_dim=latent_dim, base_channels=base_channels,
                           eval_size=eval_size).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    # Fixed decoder blob size
    dec_blob_bytes = decoder_blob_bytes(decoder_sd)
    print(f"  Decoder blob: {dec_blob_bytes:,} bytes (constant across all k)")

    # Fit PCA on CPU (latents are small: 600×28)
    latents_f = latents.float()
    mean_pca, basis_pca = fit_pca(latents_f)   # (28,), (28, 28)
    print(f"  PCA fit done. Explained variance ratio per component:")
    centered = latents_f - mean_pca
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var_ratio = (S**2) / (S**2).sum()
    cumvar = var_ratio.cumsum(dim=0)
    for ki in [16, 18, 20, 22, 24, 28]:
        print(f"    k={ki:2d}: cumulative variance explained = {cumvar[ki-1]:.4f}")

    # Sweep
    print(f"\n{'k':>4}  {'seg_dist':>10}  {'pose_dist':>10}  "
          f"{'recon_mse':>10}  {'archive':>10}  {'score':>8}  {'Δscore':>8}")
    print("-" * 80)

    rows = []
    base_score = None

    for k in K_VALUES:
        # Project → reconstruct
        coeffs = project(latents_f, mean_pca, basis_pca, k)    # (N, k)
        latents_approx = reconstruct(coeffs, mean_pca, basis_pca, k).to(device)

        recon_mse = F.mse_loss(latents_approx.cpu(), latents_f).item()

        # Evaluate decoder with approximated latents
        dist = evaluate_decoder(decoder, latents_approx, net,
                                video_path, batch_pairs=BATCH_PAIRS, device=device)

        # Archive size estimate: decoder blob + brotli(basis + mean + coefficients)
        basis_k = basis_pca[:, :k]                              # (28, k)
        lat_blob_bytes = pca_blob_size(mean_pca, basis_k, coeffs)
        # 4 bytes for meta (k) + 4 bytes for meta (D) already inside blob;
        # add 8 bytes for two u32 length fields (same layout as build_archive)
        archive_bytes_est = dec_blob_bytes + lat_blob_bytes + 16  # 16 = overhead headers

        score_d = compute_score(dist['seg_distortion'], dist['pose_distortion'],
                                archive_bytes_est, total_vb)
        s = score_d['score']

        if k == 28:
            base_score = s

        delta = (s - base_score) if base_score is not None else 0.0

        print(f"{k:>4}  {dist['seg_distortion']:>10.6f}  {dist['pose_distortion']:>10.6f}  "
              f"{recon_mse:>10.6f}  {archive_bytes_est:>10,}  {s:>8.4f}  {delta:>+8.4f}")

        rows.append((k, dist['seg_distortion'], dist['pose_distortion'],
                     recon_mse, archive_bytes_est, s))

    # Write curve
    out = HERE / "pca_curve.txt"
    with open(out, "w") as f:
        f.write("k\tseg_dist\tpose_dist\trecon_mse\tarchive_bytes\tscore\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"\nCurve written to: {out}")

    print("\nInterpretation guide:")
    print("  recon_mse ≈ 0, seg_dist unchanged → PCA at this k is lossless for the decoder")
    print("  seg_dist drifts significantly      → latents are not low-dimensional at this k")
    print("  archive shrinks AND seg stable     → winning trade-off")


if __name__ == "__main__":
    main()
