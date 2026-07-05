#!/usr/bin/env python
"""Pack 3 trained model archives into a single 0.bin for submission.

Reads:
    full_run/model_0/<stage>/best_archive.bin
    full_run/model_1/<stage>/best_archive.bin
    full_run/model_2/<stage>/best_archive.bin

Writes:
    0.bin  (in the current working directory, then compress.sh zips it)

0.bin format (see inflate.py for the matching unpack):
    4B  N=3         (uint32 LE)
    4B×3  sizes     (uint32 LE each)
    then 3 sub-archives concatenated in model order (0 → 1 → 2)

Usage:
    python compress.py [stage_name]   (default: s8_muon)
"""
import struct
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
FULL_RUN = HERE / "full_run"
N_MODELS = 3


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "s8_muon"

    archives = []
    for k in range(N_MODELS):
        p = FULL_RUN / f"model_{k}" / stage / "best_archive.bin"
        if not p.exists():
            sys.exit(f"ERROR: {p} not found — run train_full.py first")
        data = p.read_bytes()
        archives.append(data)
        print(f"  model_{k}: {len(data):,} bytes  ({p.relative_to(HERE)})")

    n = len(archives)
    header  = struct.pack(f"<I{n}I", n, *[len(a) for a in archives])
    packed  = header + b"".join(archives)
    out     = Path("0.bin")
    out.write_bytes(packed)

    total = sum(len(a) for a in archives)
    print(f"Wrote 0.bin  ({total:,}B data + {len(header)}B header = {len(packed):,}B total)")


if __name__ == "__main__":
    main()
