"""Landmark-based geometry-preservation evaluation.

For each interpolation sequence, we extract 478 MediaPipe FaceMesh
landmarks per frame and compute the average per-frame displacement between
consecutive frames. Used for the Flow-Optimizer landmark-tracking
evaluation on depth-aligned face pairs (Table X in the paper: FlowMorph
85.33 vs RF-Inversion 123.74 vs DiffMorpher 153.22).

Usage:
    python tools/eval_landmarks.py path/to/frames --pattern 'frame_*.png'
    python tools/eval_landmarks.py path/to/seq_dir --recursive
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np


def _load_mediapipe():
    try:
        import mediapipe as mp  # noqa: F401
        return mp
    except ImportError:  # pragma: no cover
        sys.exit("`mediapipe` is required for eval_landmarks.py; `pip install mediapipe`.")


def extract_landmarks(image_path: str, face_mesh) -> np.ndarray | None:
    import cv2  # local import keeps the metric optional

    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    landmarks = res.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)


def sequence_displacement(frames: List[str]) -> float:
    """Sum of per-frame mean displacement (in pixels) across the sequence.

    Frames without a detected face are skipped; if fewer than 2 frames
    contain faces, NaN is returned so bad sequences are flagged.
    """
    mp = _load_mediapipe()
    total = 0.0
    prev = None
    counted = 0

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.3) as fm:
        for p in frames:
            pts = extract_landmarks(p, fm)
            if pts is None:
                continue
            if prev is not None:
                total += float(np.linalg.norm(pts - prev, axis=1).mean())
                counted += 1
            prev = pts
    if counted == 0:
        return float("nan")
    return total


def main():
    parser = argparse.ArgumentParser(description="Sum-of-landmark-displacement evaluator")
    parser.add_argument("path", type=str, help="Directory with frames, or parent directory if --recursive")
    parser.add_argument("--pattern", type=str, default="frame_*.png")
    parser.add_argument("--recursive", action="store_true",
                        help="Compute for every immediate subdirectory")
    args = parser.parse_args()

    def _one(seq_dir: Path) -> float:
        files = sorted(glob.glob(str(seq_dir / args.pattern)))
        if not files:
            print(f"  (no frames) {seq_dir}")
            return float("nan")
        return sequence_displacement(files)

    root = Path(args.path)
    if args.recursive:
        scores = {}
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            scores[sub.name] = _one(sub)
        print()
        print(f"{'sequence':60s}  sum_displacement")
        for k, v in scores.items():
            print(f"{k:60s}  {v:.4f}")
        vs = [v for v in scores.values() if not np.isnan(v)]
        if vs:
            print(f"{'AVERAGE':60s}  {np.mean(vs):.4f}")
    else:
        print(f"sum_displacement = {_one(root):.4f}")


if __name__ == "__main__":
    main()
