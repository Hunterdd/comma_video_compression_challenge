"""All-stage training: 3 HNeRV models × 200 frames through all 8 stages.

Stage progression (fresh cosine per stage, peak LR ~3× higher than hnerv_muon):
  S1  CE           lr=3e-3  3000 ep
  S2  Softplus     lr=3e-3  5650 ep
  S3  Smooth       lr=3e-4  1500 ep
  S4  Smooth+QAT   lr=3e-4   500 ep
  S5  L7+C1a       lr=1e-4  9000 ep  λ=0.01 σ=0.2
  S6  L7+C1a       lr=1e-4  2000 ep  λ=0.02 σ=0.2
  S7  L7+C1a       lr=1e-4  3000 ep  λ=0.02 σ=0.1
  S8  L7+C1a/Muon  lr=3e-5  5000 ep  λ=0.02 σ=0.1  muon_lr=6e-4

Checkpoint convention (mirrors hnerv_muon/src/stages/common.py):
  full_run/model_{k}/{stage_name}/
    final_decoder.pt   EMA weights at last epoch  (resume token for next stage)
    final_latents.pt   EMA latents at last epoch
    best_archive.bin   INT8+brotli archive at best-score epoch
    best_meta.json     {score, epoch, archive_bytes}

Resume: if final_decoder.pt already exists the stage is SKIPPED.
        This lets you resume from any stage boundary without re-training.

Run (from challenge root):
  python submissions/my_submission/train_full.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths — local model.py takes priority over hnerv_muon's
# ---------------------------------------------------------------------------
HERE           = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent.parent
HNERV_SRC      = HERE.parent / "hnerv_muon" / "src"

sys.path.insert(0, str(CHALLENGE_ROOT))
sys.path.insert(0, str(HNERV_SRC))
sys.path.insert(0, str(HERE))  # my_submission/model.py shadows hnerv_muon/src/model.py

from codec  import build_archive, parse_archive                # noqa: E402
from model  import HNeRVDecoder                                # noqa: E402  (→ local model.py)
from score  import compute_score, total_video_bytes            # noqa: E402
from optim  import Muon, partition_params_for_muon             # noqa: E402
from losses import (                                           # noqa: E402
    ce_seg_loss, tau_softplus_seg_loss,
    smooth_disagreement_seg_loss, l7_softplus_seg_loss,
    cat_entropy_v2, apply_qat, restore_qat, ema_update,
)
from data   import precompute_targets, get_default_video_path  # noqa: E402
import frame_utils                                             # noqa: E402
import modules                                                 # noqa: E402
import av                                                      # noqa: E402
from frame_utils import yuv420_to_rgb                         # noqa: E402


# ---------------------------------------------------------------------------
# Differentiable YUV patch
# ---------------------------------------------------------------------------
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
BASE_CHANNELS    = 28         # channels = [17,17,17,12,9,8,8]  ~51K params
STEM_DIM         = 14
LATENT_DIM       = 28
EVAL_SIZE        = (384, 512)

N_MODELS         = 3
FRAMES_PER_MODEL = 200
PAIRS_PER_MODEL  = FRAMES_PER_MODEL // 2   # 100

BATCH_SIZE       = 32    # A100
EVAL_EVERY       = 100   # epochs between mid-training score checks
EMA_DECAY        = 0.999
GRAD_CLIP        = 1.0
LATENT_LR_MULT   = 10.0
LR_FLOOR         = 5e-6  # absolute minimum LR (floor of cosine schedule)

OUT_DIR = HERE / "full_run"


# ---------------------------------------------------------------------------
# Stage definitions — all 8 stages inline, no separate files
# ---------------------------------------------------------------------------
@dataclass
class StageSpec:
    name:       str
    seg_loss:   Callable      # (logits, targets) -> scalar
    epochs:     int
    lr:         float         # peak AdamW LR (fresh cosine per stage)
    qat:        bool
    use_muon:   bool
    muon_lr:    float = 6e-4  # only used when use_muon=True
    muon_wd:    float = 5e-4
    cat_lambda: float = 0.0   # C1a entropy weight
    cat_sigma:  float = 0.2   # C1a entropy bandwidth


STAGES: list[StageSpec] = [
    StageSpec("s1_ce",
              lambda l, t: ce_seg_loss(l, t),
              epochs=3000, lr=3e-3, qat=False, use_muon=False),
    StageSpec("s2_softplus",
              lambda l, t: tau_softplus_seg_loss(l, t, tau=0.3),
              epochs=5650, lr=3e-3, qat=False, use_muon=False),
    StageSpec("s3_smooth",
              lambda l, t: smooth_disagreement_seg_loss(l, t, tau=0.3),
              epochs=1500, lr=3e-4, qat=False, use_muon=False),
    StageSpec("s4_qat",
              lambda l, t: smooth_disagreement_seg_loss(l, t, tau=0.3),
              epochs=500,  lr=3e-4, qat=True,  use_muon=False),
    StageSpec("s5_c1a",
              lambda l, t: l7_softplus_seg_loss(l, t, tau=0.3),
              epochs=9000, lr=1e-4, qat=True,  use_muon=False,
              cat_lambda=0.01, cat_sigma=0.2),
    StageSpec("s6_lambda",
              lambda l, t: l7_softplus_seg_loss(l, t, tau=0.3),
              epochs=2000, lr=1e-4, qat=True,  use_muon=False,
              cat_lambda=0.02, cat_sigma=0.2),
    StageSpec("s7_sigma",
              lambda l, t: l7_softplus_seg_loss(l, t, tau=0.3),
              epochs=3000, lr=1e-4, qat=True,  use_muon=False,
              cat_lambda=0.02, cat_sigma=0.1),
    StageSpec("s8_muon",
              lambda l, t: l7_softplus_seg_loss(l, t, tau=0.3),
              epochs=5000, lr=3e-5, qat=True,  use_muon=True,
              cat_lambda=0.02, cat_sigma=0.1),
]


# ---------------------------------------------------------------------------
# Evaluate one model on its own 100-pair video slice
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_slice(decoder, latents, distortion_net, video_path,
                   pair_offset: int, device: torch.device,
                   batch_pairs: int = 16):
    """Return {'seg_distortion', 'pose_distortion'} for one model's slice."""
    decoder.eval()
    n_pairs = latents.shape[0]

    container = av.open(str(video_path))
    stream    = container.decode(container.streams.video[0])
    for _ in range(pair_offset * 2):          # skip to this model's frames
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
        decoded  = decoder(latents[start:start + B])
        flat = decoded.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
        up   = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
        out  = (up.reshape(B, 2, 3, 874, 1164)
                   .permute(0, 1, 3, 4, 2)
                   .clamp(0, 255).round().to(torch.uint8))
        pose_d, seg_d = distortion_net.compute_distortion(batch_gt, out)
        seg_sum  += seg_d.sum().item()
        pose_sum += pose_d.sum().item()
        count    += B

    return {'seg_distortion':  seg_sum  / max(count, 1),
            'pose_distortion': pose_sum / max(count, 1)}


# ---------------------------------------------------------------------------
# Single stage training
# ---------------------------------------------------------------------------
def train_one_stage(
    model_idx:    int,
    pair_offset:  int,            # absolute pair index into full video
    stage:        StageSpec,
    resume_dir:   Optional[Path],  # None → random init (stage 1 only)
    output_dir:   Path,
    seg_targets:  torch.Tensor,    # (n_pairs,) long, pre-sliced for this model
    pose_targets: torch.Tensor,    # (n_pairs, 6) float, pre-sliced
    distortion_net,
    video_path:   Path,
    tvb:          int,
    device:       torch.device,
    log:          Callable,
):
    """Train one stage for one model. Saves final + best checkpoints."""
    n_pairs = len(seg_targets)     # actual pairs for this model (may differ from PAIRS_PER_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build decoder -----
    decoder = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE,
                           stem_dim=STEM_DIM).to(device)

    if resume_dir is None:
        # Stage 1: random init
        latents = nn.Parameter(
            torch.randn(n_pairs, LATENT_DIM, device=device) * 0.1)
        log(f"    Random init")
    else:
        dec_path = resume_dir / "final_decoder.pt"
        lat_path = resume_dir / "final_latents.pt"
        decoder.load_state_dict(torch.load(dec_path, map_location=device))
        latents = nn.Parameter(torch.load(lat_path, map_location=device))
        log(f"    Resumed from {resume_dir.name}")

    ema_decoder = deepcopy(decoder)
    ema_latents = latents.data.clone()

    # ----- Optimizer -----
    if stage.use_muon:
        muon_params, adamw_params = partition_params_for_muon(decoder)
        muon_opt = Muon(muon_params, lr=stage.muon_lr, momentum=0.95,
                        nesterov=True, ns_steps=5, weight_decay=stage.muon_wd)
        adamw_opt = torch.optim.AdamW(
            [{'params': adamw_params,  'lr': stage.lr},
             {'params': [latents],     'lr': stage.lr * LATENT_LR_MULT}],
            weight_decay=0.0)
        log(f"    Muon={sum(p.numel() for p in muon_params):,}p "
            f"AdamW={sum(p.numel() for p in adamw_params):,}p "
            f"latents={latents.numel():,}")
    else:
        muon_opt = None
        adamw_opt = torch.optim.AdamW(
            [{'params': decoder.parameters(), 'lr': stage.lr},
             {'params': [latents],            'lr': stage.lr * LATENT_LR_MULT}],
            weight_decay=0.0)

    # Fresh cosine per stage
    lr_floor_mult = max(LR_FLOOR / stage.lr, 1e-3)
    def lr_lambda(ep):
        return max(0.5 * (1 + math.cos(math.pi * ep / stage.epochs)), lr_floor_mult)
    adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)
    muon_sched  = (torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
                   if muon_opt is not None else None)

    best_score = float('inf'); best_ep = 0
    t0 = time.time()

    for epoch in range(stage.epochs):
        epoch_loss = 0.0; nb = 0
        perm = torch.randperm(n_pairs, device=device)

        for start in range(0, n_pairs, BATCH_SIZE):
            idx = perm[start: start + BATCH_SIZE]
            B   = len(idx)

            if stage.qat:
                originals = apply_qat(decoder)
            decoded_pair = decoder(latents[idx])     # (B,2,3,H,W)
            if stage.qat:
                restore_qat(decoder, originals)

            flat = decoded_pair.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
            up   = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
            down = F.interpolate(up,   size=(384, 512),  mode='bilinear', align_corners=False)
            bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)

            dc   = bhwc.clamp(0, 255)
            bhwc = dc + (dc.round() - dc).detach()  # STE rounding

            posenet_in, segnet_in = distortion_net.preprocess_input(bhwc)
            seg_out  = distortion_net.segnet(segnet_in)
            pose_out = distortion_net.posenet(posenet_in)

            seg_l  = stage.seg_loss(seg_out, seg_targets[idx])
            pose_l = torch.sqrt(10.0 * F.mse_loss(
                pose_out['pose'][:, :6], pose_targets[idx]) + 1e-12)
            loss   = 100.0 * seg_l + pose_l

            if stage.cat_lambda > 0:
                ent  = cat_entropy_v2(decoder, sigma=stage.cat_sigma,
                                      sample_size=2000, device=device)
                loss = loss + stage.cat_lambda * ent

            adamw_opt.zero_grad()
            if muon_opt is not None:
                muon_opt.zero_grad()
            loss.backward()

            # Clip AdamW params (+ latents) and Muon params separately — mirrors common.py.
            # When Muon is active, latents is already inside adamw_opt.param_groups[1]['params'],
            # so don't add it again (would double-count its gradient magnitude).
            if muon_opt is None:
                torch.nn.utils.clip_grad_norm_(
                    list(decoder.parameters()) + [latents], GRAD_CLIP)
            else:
                adamw_clip = [p for pg in adamw_opt.param_groups for p in pg['params']]
                torch.nn.utils.clip_grad_norm_(adamw_clip, GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(
                    list(muon_opt.param_groups[0]['params']), GRAD_CLIP)

            adamw_opt.step()
            if muon_opt is not None:
                muon_opt.step()

            ema_update(ema_decoder, decoder, ema_latents, latents, decay=EMA_DECAY)
            epoch_loss += loss.item(); nb += 1

        adamw_sched.step()
        if muon_sched is not None:
            muon_sched.step()

        if (epoch + 1) % 10 == 0:
            log(f"    ep{epoch+1:5d}/{stage.epochs}  loss={epoch_loss/nb:.4f}  "
                f"lr={adamw_opt.param_groups[0]['lr']:.2e}  "
                f"({time.time()-t0:.0f}s)")

        if (epoch + 1) % EVAL_EVERY == 0:
            archive = build_archive(
                ema_decoder.state_dict(), ema_latents.cpu(),
                meta_dict={"n_pairs": n_pairs, "latent_dim": LATENT_DIM,
                           "base_channels": BASE_CHANNELS, "stem_dim": STEM_DIM,
                           "eval_size": list(EVAL_SIZE)},
            )
            projected = len(archive) * N_MODELS
            eval_sd, eval_lat, _ = parse_archive(archive)
            eval_dec = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE,
                                    stem_dim=STEM_DIM).to(device)
            eval_dec.load_state_dict(eval_sd); eval_dec.eval()
            dist = evaluate_slice(eval_dec, eval_lat.to(device), distortion_net,
                                  video_path, pair_offset=pair_offset, device=device)
            del eval_dec
            if device.type == "cuda":
                torch.cuda.empty_cache()

            result = compute_score(dist['seg_distortion'], dist['pose_distortion'],
                                   projected, tvb)
            marker = " *** BEST" if result['score'] < best_score else ""
            log(f"    >>> ep{epoch+1:5d}  score={result['score']:.4f}  "
                f"seg={dist['seg_distortion']:.5f}  pose={dist['pose_distortion']:.6f}  "
                f"1-bin={len(archive):,}B{marker}")

            if result['score'] < best_score:
                best_score = result['score']; best_ep = epoch + 1
                with open(output_dir / "best_archive.bin", "wb") as fh:
                    fh.write(archive)
                with open(output_dir / "best_meta.json", "w") as fh:
                    json.dump({"stage": stage.name, "score": result['score'],
                               "seg_distortion": dist['seg_distortion'],
                               "pose_distortion": dist['pose_distortion'],
                               "archive_bytes": len(archive),
                               "epoch": epoch + 1}, fh, indent=2)

    # Always save final state for next stage
    torch.save(ema_decoder.state_dict(), output_dir / "final_decoder.pt")
    torch.save(ema_latents.cpu(),        output_dir / "final_latents.pt")
    log(f"    DONE  best_score={best_score:.4f} at ep{best_ep}")
    return best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    log_path = OUT_DIR / "train_log.txt"
    # Append (don't overwrite) so resume doesn't erase earlier stage logs
    with open(log_path, "a") as fh:
        fh.write(f"\n{'='*70}\nRun started {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.fp32_precision       = 'tf32'
    torch.backends.cudnn.benchmark = True

    device = (torch.device("cuda", 0) if torch.cuda.is_available() else
              torch.device("mps")     if torch.backends.mps.is_available() else
              torch.device("cpu"))

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    log(f"Device : {device}")
    log(f"Model  : base_channels={BASE_CHANNELS}  stem_dim={STEM_DIM}  "
        f"params={sum(p.numel() for p in HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE, STEM_DIM).parameters()):,}")
    log(f"Setup  : {N_MODELS} models  batch={BATCH_SIZE}  eval_every={EVAL_EVERY}  (pairs split dynamically)")
    log(f"Stages : {[s.name for s in STAGES]}")

    video_path = get_default_video_path()
    log(f"Video  : {video_path}")

    log("\nPrecomputing SegNet/PoseNet targets...")
    distortion_net, seg_all, pose_all, _, n_pairs_full = (
        precompute_targets(video_path, device))
    log(f"  Full video pairs: {n_pairs_full} ({n_pairs_full*2} frames)")
    tvb = total_video_bytes(video_path)
    log(f"  Total video bytes: {tvb:,}\n")

    # Distribute pairs evenly; last model absorbs any remainder
    base = n_pairs_full // N_MODELS
    rem  = n_pairs_full % N_MODELS
    pairs_by_model  = [base + (1 if i < rem else 0) for i in range(N_MODELS)]
    offsets_by_model = [sum(pairs_by_model[:i]) for i in range(N_MODELS)]
    log(f"  Pairs per model : {pairs_by_model}  (sum={sum(pairs_by_model)})")
    log(f"  Pair offsets    : {offsets_by_model}")

    t_global = time.time()

    for k in range(N_MODELS):
        lo = offsets_by_model[k]
        hi = lo + pairs_by_model[k]
        seg_k  = seg_all[lo:hi]
        pose_k = pose_all[lo:hi]
        model_dir = OUT_DIR / f"model_{k}"
        model_dir.mkdir(exist_ok=True)

        log(f"\n{'='*70}")
        log(f"MODEL {k}/{N_MODELS-1}  pairs [{lo},{hi})  n={pairs_by_model[k]}  frames [{lo*2},{hi*2})")
        log(f"{'='*70}")

        for s_idx, stage in enumerate(STAGES):
            stage_dir  = model_dir / stage.name
            done_flag  = stage_dir / "final_decoder.pt"

            log(f"\n  [{stage.name}]  epochs={stage.epochs}  lr={stage.lr}  "
                f"qat={stage.qat}  muon={stage.use_muon}  "
                f"λ={stage.cat_lambda}  σ={stage.cat_sigma}")

            if done_flag.exists():
                log(f"  SKIPPED (checkpoint exists at {stage_dir.relative_to(OUT_DIR)})")
                continue

            # Resume from previous stage's final state (or random init for s1)
            resume_dir = (model_dir / STAGES[s_idx - 1].name) if s_idx > 0 else None
            if resume_dir is not None and not (resume_dir / "final_decoder.pt").exists():
                log(f"  ERROR: previous stage {resume_dir.name} has no final_decoder.pt — "
                    f"run stages in order.")
                raise RuntimeError(f"Missing resume checkpoint: {resume_dir}")

            train_one_stage(
                model_idx=k,
                pair_offset=offsets_by_model[k],
                stage=stage,
                resume_dir=resume_dir,
                output_dir=stage_dir,
                seg_targets=seg_k,
                pose_targets=pose_k,
                distortion_net=distortion_net,
                video_path=video_path,
                tvb=tvb,
                device=device,
                log=log,
            )

    # -------------------------------------------------------------------
    # Final combined score using last stage's best_archive.bin per model
    # -------------------------------------------------------------------
    log(f"\n{'='*70}")
    log("FINAL COMBINED SCORE (Stage 8 archives)")
    log(f"{'='*70}")

    final_stage_name = STAGES[-1].name
    archives = {}
    for k in range(N_MODELS):
        p = OUT_DIR / f"model_{k}" / final_stage_name / "best_archive.bin"
        if p.exists():
            archives[k] = p.read_bytes()
        else:
            log(f"  WARNING: model_{k}/{final_stage_name}/best_archive.bin not found")

    if len(archives) == N_MODELS:
        combined_bytes = sum(len(a) for a in archives.values())
        log(f"  Combined: {combined_bytes:,}B  "
            + "  ".join(f"m{k}={len(archives[k]):,}B" for k in range(N_MODELS)))

        seg_sum = 0.0; pose_sum = 0.0
        for k in range(N_MODELS):
            eval_sd, eval_lat, _ = parse_archive(archives[k])
            eval_dec = HNeRVDecoder(LATENT_DIM, BASE_CHANNELS, EVAL_SIZE,
                                    stem_dim=STEM_DIM).to(device)
            eval_dec.load_state_dict(eval_sd); eval_dec.eval()
            dist = evaluate_slice(eval_dec, eval_lat.to(device), distortion_net,
                                  video_path, pair_offset=offsets_by_model[k], device=device)
            del eval_dec
            if device.type == "cuda":
                torch.cuda.empty_cache()
            seg_sum  += dist['seg_distortion']
            pose_sum += dist['pose_distortion']
            log(f"  model_{k}: seg={dist['seg_distortion']:.5f}  "
                f"pose={dist['pose_distortion']:.6f}")

        avg_seg  = seg_sum  / N_MODELS
        avg_pose = pose_sum / N_MODELS
        result   = compute_score(avg_seg, avg_pose, combined_bytes, tvb)
        log(f"\n  FINAL SCORE: {result['score']:.4f}")
        log(f"    seg={avg_seg:.6f}  pose={avg_pose:.6f}")
        log(f"    rate={result.get('rate', combined_bytes/tvb):.6f}")

    log(f"\nTotal wall time: {(time.time() - t_global)/3600:.2f} hr")
    log(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
