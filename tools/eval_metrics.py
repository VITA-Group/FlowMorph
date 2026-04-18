"""Perceptual / fidelity metrics used in Table 1 of the paper.

Computes LPIPS_sum, FID_mean, and PPL_sum over a directory of
interpolation sequences, matching the protocol in Section 4. This is a
thin wrapper: we defer the heavy lifting to the same libraries used by
FreeMorph and IMPUS so the numbers are directly comparable.

Dependencies: `pip install lpips torchmetrics[image] cleanfid`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List

import numpy as np


def _lazy_imports():
    try:
        import torch
        import lpips
        from cleanfid import fid
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Evaluation dependencies missing. Install with: "
            "`pip install lpips torchmetrics[image] cleanfid`. "
            f"Underlying error: {e}"
        )
    return torch, lpips, fid, Image


def _open(path, Image):
    import torch
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def lpips_sum(frames: List[str]) -> float:
    """Sum of LPIPS distances between successive frames."""
    torch, lpips, _, Image = _lazy_imports()
    loss_fn = lpips.LPIPS(net="alex").cuda() if torch.cuda.is_available() else lpips.LPIPS(net="alex")

    total = 0.0
    prev = None
    for p in frames:
        cur = _open(p, Image).to(next(loss_fn.parameters()).device)
        if prev is not None:
            total += float(loss_fn(prev, cur).detach().cpu().item())
        prev = cur
    return total


def ppl_sum(frames: List[str]) -> float:
    """Perceptual Path Length proxy: LPIPS between widely-separated frames."""
    n = len(frames)
    if n < 2:
        return float("nan")
    # Match the FreeMorph protocol: sum squared LPIPS distances scaled by n.
    torch, lpips, _, Image = _lazy_imports()
    loss_fn = lpips.LPIPS(net="alex").cuda() if torch.cuda.is_available() else lpips.LPIPS(net="alex")

    total = 0.0
    prev = _open(frames[0], Image).to(next(loss_fn.parameters()).device)
    for p in frames[1:]:
        cur = _open(p, Image).to(next(loss_fn.parameters()).device)
        d = float(loss_fn(prev, cur).detach().cpu().item())
        total += (d ** 2) * (n - 1)
        prev = cur
    return total


def fid_mean(real_dir: str, fake_dir: str) -> float:
    """Clean-FID between two directories (numbers from Table 1 use this)."""
    _, _, fid, _ = _lazy_imports()
    return float(fid.compute_fid(real_dir, fake_dir))


def run_directory(seq_dir: Path, frame_glob: str) -> dict:
    frames = sorted(glob.glob(str(seq_dir / frame_glob)))
    return {
        "seq_dir": str(seq_dir),
        "n_frames": len(frames),
        "lpips_sum": lpips_sum(frames) if frames else float("nan"),
        "ppl_sum": ppl_sum(frames) if frames else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="FlowMorph quantitative metrics")
    parser.add_argument("path", type=str, help="Directory of frames or parent of sequence dirs")
    parser.add_argument("--pattern", type=str, default="frame_*.png")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--real-dir", type=str, default=None,
                        help="Directory of real images for FID_mean computation")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.path)
    results: list[dict] = []
    if args.recursive:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            results.append(run_directory(sub, args.pattern))
    else:
        results.append(run_directory(root, args.pattern))

    if args.real_dir:
        fid_score = fid_mean(args.real_dir, str(root))
        print(f"FID_mean vs {args.real_dir}: {fid_score:.4f}")

    lp = [r["lpips_sum"] for r in results if not np.isnan(r["lpips_sum"])]
    pp = [r["ppl_sum"] for r in results if not np.isnan(r["ppl_sum"])]
    print("\nPer-sequence:")
    for r in results:
        print(f"  {r['seq_dir']}: LPIPS_sum={r['lpips_sum']:.3f}, PPL_sum={r['ppl_sum']:.3f}")
    if lp:
        print(f"\nAvg LPIPS_sum = {np.mean(lp):.3f}")
    if pp:
        print(f"Avg PPL_sum   = {np.mean(pp):.3f}")

    if args.json_out:
        with open(args.json_out, "w") as fp:
            json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
