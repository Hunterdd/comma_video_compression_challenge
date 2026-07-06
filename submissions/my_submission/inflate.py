#!/usr/bin/env python
"""Inflate our multi-model 0.bin → 0.raw (all 600 frames).

0.bin format:
    4B  N  (uint32 LE) — number of sub-archives (always 3)
    4B×N   size of each sub-archive (uint32 LE)
    then N sub-archives concatenated in pair order (model 0 → 1 → 2)

Each sub-archive is a standard HNeRV codec archive (INT8 + brotli) covering
100 frame-pairs (200 frames).  inflate decodes them in order and writes all
600 frames as contiguous uint8 RGB (N, H, W, 3) — no header — to dst.raw.

Called by inflate.sh as:
    python -m submissions.my_submission.inflate <src.bin> <dst.raw>
"""
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hnerv_muon" / "src")) # codec, score, etc.
sys.path.insert(0, str(HERE))                               # local model.py (stem_dim) — must be last insert to be first in path

from model import HNeRVDecoder, CompactTINCHNeRV   # noqa: E402  (→ my_submission/model.py)
from codec import parse_archive  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164  # required by the eval harness


# ---------------------------------------------------------------------------
# Container packing / unpacking
# ---------------------------------------------------------------------------

def pack_archives(archives: list[bytes]) -> bytes:
    """Pack N sub-archives into a single 0.bin blob."""
    n = len(archives)
    header = struct.pack(f"<I{n}I", n, *[len(a) for a in archives])
    return header + b"".join(archives)


def unpack_archives(data: bytes) -> list[bytes]:
    """Unpack a 0.bin blob into N sub-archive byte strings."""
    n = struct.unpack("<I", data[:4])[0]
    sizes = struct.unpack(f"<{n}I", data[4: 4 + 4 * n])
    offset = 4 + 4 * n
    out = []
    for sz in sizes:
        out.append(data[offset: offset + sz])
        offset += sz
    return out


# ---------------------------------------------------------------------------
# Inflate one sub-archive → raw frames (numpy bytes)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _decode_subarchive(archive_bytes: bytes, device: torch.device) -> bytes:
    decoder_sd, latents, meta = parse_archive(archive_bytes)

    model_type = meta.get("model_type", "HNeRV")  # default for legacy archives
    if model_type == "CompactTINCHNeRV":
        decoder = CompactTINCHNeRV(
            latent_dim=meta["latent_dim"],
            base_channels=meta["base_channels"],
            eval_size=tuple(meta["eval_size"]),
        ).to(device)
        decoder.load_state_dict(decoder_sd)
        decoder.eval()
        latents = latents.to(device)
        n_chunks = meta.get("n_chunks", 4)
        pairs_per_chunk = meta["n_pairs"] // n_chunks
        chunks = []
        for chunk_id in range(n_chunks):
            chunk_start = chunk_id * pairs_per_chunk
            chunk_end = min((chunk_id + 1) * pairs_per_chunk, meta["n_pairs"])
            if chunk_id == n_chunks - 1:
                chunk_end = meta["n_pairs"]
            chunk_id_tensor = torch.tensor([chunk_id], device=device)
            for i in range(chunk_start, chunk_end, 16):
                j = min(i + 16, chunk_end)
                B = j - i
                decoded = decoder(latents[i:j], chunk_id_tensor)  # (B,2,3,H,W)
                flat = decoded.reshape(B * 2, 3, meta["eval_size"][0], meta["eval_size"][1])
                up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                                   mode="bicubic", align_corners=False)
                frames = (up.clamp(0, 255)
                            .permute(0, 2, 3, 1)
                            .round().to(torch.uint8).cpu().numpy())
                chunks.append(frames.tobytes())
        return b"".join(chunks)

    # Original HNeRV fallback (no chunking)
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
        stem_dim=meta.get("stem_dim", 14),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]

    chunks = []
    for i in range(0, n_pairs, 16):
        j = min(i + 16, n_pairs)
        B = j - i
        decoded = decoder(latents[i:j])                      # (B, 2, 3, eval_h, eval_w)
        flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
        up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                           mode="bicubic", align_corners=False)
        frames = (up.clamp(0, 255)
                    .permute(0, 2, 3, 1)                     # (B*2, H, W, 3)
                    .round().to(torch.uint8).cpu().numpy())
        chunks.append(frames.tobytes())

    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def inflate(src_bin: str, dst_raw: str):
    raw = Path(src_bin).read_bytes()
    sub_archives = unpack_archives(raw)
    print(f"  {len(sub_archives)} sub-archives, sizes: "
          f"{[len(a) for a in sub_archives]}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_frames = 0

    with open(dst_raw, "wb") as fout:
        for k, arch in enumerate(sub_archives):
            frame_bytes = _decode_subarchive(arch, device)
            fout.write(frame_bytes)
            n = len(frame_bytes) // (CAMERA_H * CAMERA_W * 3)
            total_frames += n
            print(f"  model_{k}: {n} frames written", flush=True)

    print(f"saved {total_frames} frames → {dst_raw}")
    return total_frames


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.my_submission.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
