"""Stage-1 training: 3 HNeRV models × 200 frames each = 600 frames total.

KISS / YAGNI:
  - Single file, Stage 1 CE loss only, three models trained sequentially.
  - 600 frames → 3 models, each sees 100 pairs (frames [k*200, (k+1)*200)).
  - Progress print every 10 epochs; score eval every 100 epochs.
  - After all 3 models finish: build combined archive (sum of 3 bin files),
    compute final score with the real combined size.
  - A100 optimised: torch.compile, TF32, batch=32.

Archive layout (same codec as hnerv_muon — INT8 + brotli):
  Each model produces its own .bin (build_archive).
  Combined size = sum of 3 bin sizes (used for the rate term in scoring).

Run:
  python submissions/my_submission/train_dry.py

Outputs  submissions/my_submission/dry_run/
  model_{k}_decoder.pt   EMA decoder weights (best epoch)
  model_{k}_latents.pt   EMA latents (best epoch)
  model_{k}.bin          codec archive for model k
  train_log.txt
"""
from __future__ import annotations

import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE          = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent.parent
HNERV_SRC     = HERE.parent / "hnerv_muon" / "src"

sys.path.insert(0, str(CHALLENGE_ROOT))
sys.path.insert(0, str(HNERV_SRC))

from codec  import build_archive, parse_archive               # noqa: E402
from model  import HNeRVDecoder                               # noqa: E402
from score  import compute_score, total_video_bytes           # noqa: E402
from losses import ce_seg_loss, ema_update                    # noqa: E402
from data   import precompute_targets, get_default_video_path # noqa: E402

# Differentiable YUV patch — identical to hnerv_muon/src/data.py.
# Prevents @no_grad in frame_utils from severing pose gradients.
import frame_utils  # noqa: E402
import modules      # noqa: E402
import av           # noqa: E402
from frame_utils import yuv420_to_rgb  # noqa: E402


def _rgb_to_yuv6_diff(rgb_chw):
    H, W   = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb    = rgb_chw[..., :, :2*H2, :2*W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y  = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0., 255.)
    U  = ((B - Y) / 1.772 + 128.).clamp(0., 255.)
    V  = ((R - Y) / 1.402 + 128.).clamp(0., 255.)
    Us = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
          + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    Vs = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
          + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2],
                        Y[..., 0::2, 1::2], Y[..., 1::2, 1::2], Us, Vs], dim=-3)


frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_diff
modules.rgb_to_yuv6     = _rgb_to_yuv6_diff

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_CHANNELS    = 20          # channels = [20,20,20,15,11,10,10]
LATENT_DIM       = 28
EVAL_SIZE        = (384, 512)

N_MODELS         = 3
FRAMES_PER_MODEL = 200
PAIRS_PER_MODEL  = FRAMES_PER_MODEL // 2   # 100

EPOCHS           = 3000        # full Stage 1
BATCH_SIZE       = 32          # A100: 4× bigger than CPU default
EVAL_EVERY       = 100         # score during training
ADAMW_LR         = 1e-3
LATENT_LR_MULT   = 10.0
GRAD_CLIP        = 1.0
EMA_DECAY        = 0.999

OUT_DIR = HERE / "dry_run"


# ---------------------------------------------------------------------------
# Eval helper: evaluate a decoder against a specific 100-pair slice of the video.
# score.py's evaluate_decoder always starts from pair 0; this version skips
# pair_offset pairs so model 1 and model 2 are scored on the right frames.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_slice(decoder, latents, distortion_net, video_path,
                   pair_offset: int, device: torch.device,
                   batch_pairs: int = 16):
    """Return {'seg_distortion', 'pose_distortion'} for one 100-pair model slice."""
    decoder.eval()
    n_pairs = latents.shape[0]

    # Stream video, skip pair_offset*2 frames, collect n_pairs pairs
    container = av.open(str(video_path))
    stream    = container.decode(container.streams.video[0])

    frames_to_skip = pair_offset * 2
    for _ in range(frames_to_skip):
        next(stream)

    gt_pairs = []
    prev = None
    for frame in stream:
        f = yuv420_to_rgb(frame)
        if prev is None:
            prev = f
            continue
        gt_pairs.append(torch.stack([prev, f]))  # (2, H, W, 3) uint8
        prev = None
        if len(gt_pairs) == n_pairs:
            break
    container.close()

    seg_total = 0.0; pose_total = 0.0; count = 0
    for start in range(0, len(gt_pairs), batch_pairs):
        batch_gt = torch.stack(gt_pairs[start:start + batch_pairs]).to(device)
        B = batch_gt.shape[0]
        decoded = decoder(latents[start:start + B])         # (B,2,3,H,W)
        flat = decoded.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
        up   = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
        out  = (up.reshape(B, 2, 3, 874, 1164)
                   .permute(0, 1, 3, 4, 2)
                   .clamp(0, 255).round().to(torch.uint8))
        pose_d, seg_d = distortion_net.compute_distortion(batch_gt, out)
        seg_total  += seg_d.sum().item()
        pose_total += pose_d.sum().item()
        count += B

    return {'seg_distortion':  seg_total  / max(count, 1),
            'pose_distortion': pose_total / max(count, 1)}


# ---------------------------------------------------------------------------
# Single-model Stage-1 training loop
# ---------------------------------------------------------------------------
def train_one_model(model_idx: int,
                    seg_targets: torch.Tensor,
                    pose_targets: torch.Tensor,
                    distortion_net,
                    video_path: Path,
                    tvb: int,
                    device: torch.device,
                    log):
    """Train one HNeRVDecoder on PAIRS_PER_MODEL pairs. Returns (ema_sd, ema_lat, best_score)."""
    n_pairs = PAIRS_PER_MODEL
    pair_offset = model_idx * PAIRS_PER_MODEL

    decoder = HNeRVDecoder(latent_dim=LATENT_DIM,
                           base_channels=BASE_CHANNELS,
                           eval_size=EVAL_SIZE).to(device)
    latents    = nn.Parameter(torch.randn(n_pairs, LATENT_DIM, device=device) * 0.1)
    ema_decoder = deepcopy(decoder)
    ema_latents = latents.data.clone()

    # Compile for A100 speed (deepcopy done first so EMA model is uncompiled)
    if device.type == "cuda":
        decoder = torch.compile(decoder)

    optimizer = torch.optim.AdamW(
        [{'params': decoder.parameters(), 'lr': ADAMW_LR},
         {'params': [latents], 'lr': ADAMW_LR * LATENT_LR_MULT}],
        weight_decay=0.0,
    )
    lr_floor  = 5e-6 / ADAMW_LR
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: max(0.5 * (1 + math.cos(math.pi * ep / EPOCHS)), lr_floor),
    )

    best_score    = float('inf')
    best_ep       = 0
    best_archive  = None
    t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0; nb = 0
        perm = torch.randperm(n_pairs, device=device)

        for start in range(0, n_pairs, BATCH_SIZE):
            idx = perm[start: start + BATCH_SIZE]
            B   = len(idx)

            decoded_pair = decoder(latents[idx])          # (B,2,3,H,W) [0,255]
            flat = decoded_pair.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
            up   = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
            down = F.interpolate(up,   size=(384, 512),  mode='bilinear', align_corners=False)
            bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)

            # Straight-through rounding (STE)
            dc   = bhwc.clamp(0, 255)
            bhwc = dc + (dc.round() - dc).detach()

            posenet_in, segnet_in = distortion_net.preprocess_input(bhwc)
            seg_out  = distortion_net.segnet(segnet_in)
            pose_out = distortion_net.posenet(posenet_in)

            seg_l  = ce_seg_loss(seg_out, seg_targets[idx])
            pose_l = torch.sqrt(
                10.0 * F.mse_loss(pose_out['pose'][:, :6], pose_targets[idx]) + 1e-12)
            loss   = 100.0 * seg_l + pose_l

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + [latents], GRAD_CLIP)
            optimizer.step()
            ema_update(ema_decoder, decoder, ema_latents, latents, decay=EMA_DECAY)

            epoch_loss += loss.item(); nb += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            log(f"  [m{model_idx}] ep{epoch+1:4d}/{EPOCHS}  "
                f"loss={epoch_loss/nb:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({time.time()-t0:.0f}s)")

        if (epoch + 1) % EVAL_EVERY == 0:
            # Build archive (INT8 + brotli — same as hnerv_muon)
            archive = build_archive(
                ema_decoder.state_dict(), ema_latents.cpu(),
                meta_dict={"n_pairs": n_pairs, "latent_dim": LATENT_DIM,
                           "base_channels": BASE_CHANNELS,
                           "eval_size": list(EVAL_SIZE)},
            )
            # Score with projected 3-model size (approximate mid-training signal)
            projected = len(archive) * N_MODELS
            eval_sd, eval_lat, _ = parse_archive(archive)
            eval_dec = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).to(device)
            eval_dec.load_state_dict(eval_sd); eval_dec.eval()
            dist = evaluate_slice(eval_dec, eval_lat.to(device), distortion_net,
                                  video_path, pair_offset=pair_offset, device=device)
            del eval_dec
            if device.type == "cuda":
                torch.cuda.empty_cache()

            result = compute_score(dist['seg_distortion'], dist['pose_distortion'],
                                   projected, tvb)
            marker = " *** BEST" if result['score'] < best_score else ""
            log(f"  [m{model_idx}] >>> ep{epoch+1:4d}  score={result['score']:.4f}  "
                f"seg={dist['seg_distortion']:.5f}  pose={dist['pose_distortion']:.6f}  "
                f"1-bin={len(archive):,}B  3-proj={projected:,}B{marker}")

            if result['score'] < best_score:
                best_score   = result['score']
                best_ep      = epoch + 1
                best_archive = archive

    log(f"  [m{model_idx}] DONE  best_score={best_score:.4f} at ep{best_ep}")
    return ema_decoder.state_dict(), ema_latents.cpu(), best_score, best_archive


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    log_path = OUT_DIR / "train_log.txt"
    log_path.write_text("")

    # A100 tuning
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")

    device = (torch.device("cuda", 0) if torch.cuda.is_available() else
              torch.device("mps")     if torch.backends.mps.is_available() else
              torch.device("cpu"))

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"Device: {device}")
    log(f"Model: base_channels={BASE_CHANNELS}  channels={HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).channels}")
    log(f"3 models × {FRAMES_PER_MODEL} frames ({PAIRS_PER_MODEL} pairs) = 600 frames total")
    log(f"Epochs={EPOCHS}  batch={BATCH_SIZE}  eval_every={EVAL_EVERY}  lr={ADAMW_LR}")

    video_path = get_default_video_path()
    log(f"Video: {video_path}")

    # Precompute all targets once; slice per model
    log("\nPrecomputing SegNet/PoseNet targets for all 300 pairs...")
    distortion_net, seg_all, pose_all, _gt_half, n_pairs_full = (
        precompute_targets(video_path, device))
    log(f"  Full video pairs: {n_pairs_full}")

    tvb = total_video_bytes(video_path)
    log(f"  Total video bytes: {tvb:,}\n")

    archives   = {}   # model_idx -> archive bytes
    t_global   = time.time()

    for k in range(N_MODELS):
        lo = k * PAIRS_PER_MODEL
        hi = lo + PAIRS_PER_MODEL
        seg_k  = seg_all[lo:hi]
        pose_k = pose_all[lo:hi]

        log(f"\n{'='*70}")
        log(f"Model {k}/{ N_MODELS-1}  pairs [{lo}, {hi})  "
            f"frames [{lo*2}, {hi*2})")
        log(f"{'='*70}")

        ema_sd, ema_lat, best_score, best_archive = train_one_model(
            model_idx=k,
            seg_targets=seg_k,
            pose_targets=pose_k,
            distortion_net=distortion_net,
            video_path=video_path,
            tvb=tvb,
            device=device,
            log=log,
        )

        # Save per-model artifacts
        torch.save(ema_sd,  OUT_DIR / f"model_{k}_decoder.pt")
        torch.save(ema_lat, OUT_DIR / f"model_{k}_latents.pt")
        bin_path = OUT_DIR / f"model_{k}.bin"
        with open(bin_path, "wb") as fh:
            fh.write(best_archive)
        archives[k] = best_archive
        log(f"  Saved model_{k}.bin ({len(best_archive):,} bytes)")

    # -------------------------------------------------------------------
    # Final combined score: real 3-model archive size
    # -------------------------------------------------------------------
    log(f"\n{'='*70}")
    log("FINAL combined evaluation")
    log(f"{'='*70}")

    combined_bytes = sum(len(a) for a in archives.values())
    log(f"Combined archive: {combined_bytes:,} bytes  "
        f"(model_0={len(archives[0]):,}  "
        f"model_1={len(archives[1]):,}  "
        f"model_2={len(archives[2]):,})")

    # Evaluate each model on its own video slice and aggregate
    seg_sum = 0.0; pose_sum = 0.0
    for k in range(N_MODELS):
        eval_sd, eval_lat, _ = parse_archive(archives[k])
        eval_dec = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE).to(device)
        eval_dec.load_state_dict(eval_sd); eval_dec.eval()
        dist = evaluate_slice(eval_dec, eval_lat.to(device), distortion_net,
                              video_path, pair_offset=k * PAIRS_PER_MODEL, device=device)
        del eval_dec
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"  model_{k}: seg={dist['seg_distortion']:.5f}  "
            f"pose={dist['pose_distortion']:.6f}")
        seg_sum  += dist['seg_distortion']
        pose_sum += dist['pose_distortion']

    avg_seg  = seg_sum  / N_MODELS
    avg_pose = pose_sum / N_MODELS
    result   = compute_score(avg_seg, avg_pose, combined_bytes, tvb)

    log(f"\nFinal score : {result['score']:.4f}")
    log(f"  seg_dist  = {avg_seg:.6f}  (component: {result['seg_component']:.4f})")
    log(f"  pose_dist = {avg_pose:.6f}  (component: {result['pose_component']:.4f})")
    log(f"  rate      = {result['rate']:.6f}  (component: {result['rate_component']:.4f})")
    log(f"\nTotal wall time: {(time.time() - t_global)/3600:.2f} hr")
    log(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
