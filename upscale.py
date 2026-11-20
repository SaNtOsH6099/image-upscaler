#!/usr/bin/env python3
"""Upscale an image 2x with Real-ESRGAN x2plus (ONNX, CPU, tiled).

Usage: upscale.py INPUT OUTPUT [--tile 512] [--threads 4]
Input is expected as RGB JPEG. Output is JPEG quality 95.
Tiling keeps memory low: each tile is processed with a halo so seams
do not show. The model file lives in models/ next to this script.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "realesrgan-x2plus.onnx",
)


def run_tiled(sess, x, tile=512, halo=16):
    _, _, H, W = x.shape
    out = np.zeros((1, 3, H * 2, W * 2), dtype=np.float32)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            hy0 = max(y0 - halo, 0)
            hx0 = max(x0 - halo, 0)
            hy1 = min(y1 + halo, H)
            hx1 = min(x1 + halo, W)
            tin = x[:, :, hy0:hy1, hx0:hx1]
            tout = sess.run(["output"], {"input": tin})[0]
            oy0 = (y0 - hy0) * 2
            ox0 = (x0 - hx0) * 2
            oh = (y1 - y0) * 2
            ow = (x1 - x0) * 2
            out[:, :, y0 * 2:y1 * 2, x0 * 2:x1 * 2] = tout[
                :, :, oy0:oy0 + oh, ox0:ox0 + ow
            ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = np.transpose(arr, (2, 0, 1))[None, :, :, :]

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        MODEL, so, providers=["CPUExecutionProvider"]
    )

    t0 = time.time()
    out = run_tiled(sess, x, tile=args.tile)
    dt = time.time() - t0

    o = np.clip(np.transpose(out[0], (1, 2, 0)), 0, 1)
    Image.fromarray((o * 255.0).round().astype(np.uint8)).save(
        args.output, quality=95
    )
    print(f"{args.input} {img.size} -> {Image.open(args.output).size} in {dt:.1f}s")


if __name__ == "__main__":
    main()
