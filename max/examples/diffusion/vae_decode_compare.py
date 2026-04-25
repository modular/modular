#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Compare Wan VAE decode output between cuDNN and Mojo conv paths.

The Wan VAE normally routes its 3D/2D convolutions through cuDNN on NVIDIA
GPUs. Setting ``MAX_WAN_VAE_DISABLE_CUDNN=1`` re-routes every conv through
MAX's native Mojo kernels. This script runs the VAE decoder twice on the
same deterministic latent (once per path) and reports pixel-level
differences. Any regression introduced into the Mojo conv path will show
up as a larger-than-expected max diff, a jump in NaN fraction, or a jump
in the fraction of pixels saturating to -1 (which post-processes to black).

Usage:

.. code-block:: bash

    # Run the full comparison (default shape/model):
    ./bazelw run //max/examples/diffusion:vae_decode_compare

    # Custom latent shape and seed:
    ./bazelw run //max/examples/diffusion:vae_decode_compare -- \\
        --shape 1 16 5 30 52 --seed 42

    # Re-use saved outputs and only re-run the comparison:
    ./bazelw run //max/examples/diffusion:vae_decode_compare -- --compare-only

The comparison prints: per-path value ranges, abs-diff statistics
(max/mean/std/percentiles), per-frame max diff, black-pixel and NaN
fractions per path, and a pass/warn/fail verdict based on max abs diff.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("vae_decode_compare")


# T2V-A14B (and Wan2.1) has the standard z_dim=16 / base_dim=96 /
# decoder_base_dim=None config that MAX's AutoencoderKLWanModel supports.
# TI2V-5B uses a split base_dim=160 encoder / decoder_base_dim=256 decoder
# which MAX does not currently wire through, so it fails weight-shape
# validation at load time. Stick to a standard-config model here.
_DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
# (B, z_dim, T, H, W). T=5 exercises first-frame + rest-frame cached decode
# (4 cached frames after the first); H/W are chosen small so the whole
# comparison runs in a few seconds on B200 without blowing memory.
_DEFAULT_SHAPE = [1, 16, 5, 30, 52]


def _run_decode(
    model_repo: str,
    seed: int,
    latent_shape: tuple[int, ...],
    output_path: Path,
    encoding: str,
) -> None:
    """Decode a deterministic latent through the Wan VAE and save to .npz.

    Which conv path is exercised is decided by the ``MAX_WAN_VAE_DISABLE_CUDNN``
    env var at VAE-graph-compile time, so this worker is meant to be
    launched once per setting from the orchestrator.
    """
    # Late imports so ``--compare-only`` doesn't pay the MAX import cost.
    from typing import cast

    from huggingface_hub import snapshot_download
    from max.driver import CPU, DeviceSpec, load_devices
    from max.engine import InferenceSession
    from max.graph.weights import load_weights
    from max.pipelines.architectures.autoencoders.autoencoder_kl_wan import (
        AutoencoderKLWanModel,
        _buffer_to_numpy_f32,
        _numpy_f32_to_buffer,
    )
    from max.pipelines.lib import SupportedEncoding

    env_flag = os.environ.get("MAX_WAN_VAE_DISABLE_CUDNN", "").lower()
    path_label = "mojo" if env_flag in ("1", "true") else "cudnn"

    logger.info("[%s] Downloading VAE from %s", path_label, model_repo)
    snapshot = snapshot_download(repo_id=model_repo, allow_patterns=["vae/*"])
    vae_dir = Path(snapshot) / "vae"
    config_path = vae_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"VAE config.json not found at {config_path}")
    with open(config_path) as f:
        config_dict = json.load(f)

    weight_paths = sorted(vae_dir.glob("*.safetensors"))
    if not weight_paths:
        raise RuntimeError(
            f"No .safetensors files under {vae_dir}. "
            "The VAE subfolder may not have been downloaded."
        )
    weights = load_weights(weight_paths)

    # Pre-flight: warn on configs that MAX's Wan VAE code path doesn't
    # support, so we fail with a pointer instead of a cryptic weight
    # shape mismatch from load_state_dict 200 lines later.
    if config_dict.get("decoder_base_dim") is not None:
        raise RuntimeError(
            f"Model {model_repo!r} has decoder_base_dim="
            f"{config_dict['decoder_base_dim']} which MAX's "
            "AutoencoderKLWanModel does not currently wire into the "
            "decoder (it always uses base_dim). Pick a VAE with a "
            "standard config (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers)."
        )
    cfg_z_dim = config_dict.get("z_dim", 16)
    if latent_shape[1] != cfg_z_dim:
        raise RuntimeError(
            f"Latent z-dim {latent_shape[1]} (from --shape) does not "
            f"match VAE config z_dim={cfg_z_dim}. Re-run with "
            f"--shape {latent_shape[0]} {cfg_z_dim} "
            f"{' '.join(str(s) for s in latent_shape[2:])}."
        )

    devices = load_devices([DeviceSpec.accelerator()])
    device = devices[0]
    session = InferenceSession(devices=devices)

    logger.info(
        "[%s] Building AutoencoderKLWanModel (path decided by env: "
        "MAX_WAN_VAE_DISABLE_CUDNN=%r)",
        path_label,
        env_flag or "(unset)",
    )
    encoding_literal = cast("SupportedEncoding", encoding)
    vae = AutoencoderKLWanModel(
        config=config_dict,
        encoding=encoding_literal,
        devices=devices,
        weights=weights,
        session=session,
    )

    # Deterministic latent in f32 (matches the pipeline's pre-decode dtype).
    rng = np.random.RandomState(seed)
    latent_np = rng.randn(*latent_shape).astype(np.float32)
    target_dtype = vae.config.dtype
    latent_buf = _numpy_f32_to_buffer(latent_np, target_dtype, device)

    logger.info(
        "[%s] Decoding latent of shape %s (seed=%d)",
        path_label,
        tuple(latent_shape),
        seed,
    )
    # Warm up once so the timed run doesn't include any first-launch
    # kernel JIT/autotune cost that the MAX engine may do lazily.
    _ = vae.decode_5d(latent_buf)
    device.synchronize()

    # Execution-only wall time: time decode_5d + device.synchronize()
    # so we block on the GPU before taking the end timestamp. Model
    # download, weight load, and graph compile all ran earlier and are
    # excluded. The CPU-side copy (_buffer_to_numpy_f32) is also
    # excluded so this number reflects GPU conv/norm/attention work
    # only.
    t0 = time.perf_counter()
    decoded_buf = vae.decode_5d(latent_buf)
    device.synchronize()
    decode_s = time.perf_counter() - t0
    print(
        f"[{path_label}] decode_wall_time: {decode_s * 1000:.2f} ms"
        f" ({decode_s:.3f} s)"
    )
    logger.info(
        "[%s] Decode wall time (GPU only): %.3f s", path_label, decode_s
    )

    decoded_np = _buffer_to_numpy_f32(decoded_buf, CPU())

    np.savez(
        output_path,
        decoded=decoded_np,
        latent=latent_np,
        seed=np.asarray(seed),
        shape=np.asarray(latent_shape),
        path=np.asarray([path_label]),
        model=np.asarray([model_repo]),
    )
    print(
        f"[{path_label}] shape={decoded_np.shape} "
        f"range=[{decoded_np.min():.4f}, {decoded_np.max():.4f}] "
        f"mean={decoded_np.mean():.4f} std={decoded_np.std():.4f}"
    )
    nan_frac = float(np.isnan(decoded_np).mean())
    inf_frac = float(np.isinf(decoded_np).mean())
    if nan_frac > 0 or inf_frac > 0:
        print(
            f"[{path_label}] WARNING: nan_frac={nan_frac:.6f} "
            f"inf_frac={inf_frac:.6f}"
        )
    print(f"[{path_label}] Saved to {output_path}")


def _summarize(name: str, arr: np.ndarray) -> None:
    finite = arr[np.isfinite(arr)]
    nan_frac = float(np.isnan(arr).mean())
    inf_frac = float(np.isinf(arr).mean())
    print(
        f"  {name}: shape={arr.shape} "
        f"range=[{finite.min():.6f}, {finite.max():.6f}] "
        f"mean={finite.mean():.6f} std={finite.std():.6f}"
    )
    if nan_frac > 0 or inf_frac > 0:
        print(f"    nan_frac={nan_frac:.6f} inf_frac={inf_frac:.6f}")


def _compare(cudnn_path: Path, mojo_path: Path) -> int:
    """Print diff statistics. Return 0 if PASS, 1 if WARN, 2 if FAIL."""
    cudnn = np.load(cudnn_path, allow_pickle=False)
    mojo = np.load(mojo_path, allow_pickle=False)

    cudnn_pixels = cudnn["decoded"].astype(np.float32)
    mojo_pixels = mojo["decoded"].astype(np.float32)

    if cudnn_pixels.shape != mojo_pixels.shape:
        print(
            f"ERROR: shape mismatch cudnn={cudnn_pixels.shape} "
            f"mojo={mojo_pixels.shape}",
            file=sys.stderr,
        )
        return 2

    # Sanity check: same latent was used.
    if not np.array_equal(cudnn["latent"], mojo["latent"]):
        print(
            "ERROR: input latents differ between cuDNN and Mojo runs; "
            "re-run both with the same --seed and --shape.",
            file=sys.stderr,
        )
        return 2

    diff = cudnn_pixels - mojo_pixels
    finite_mask = np.isfinite(diff)
    abs_diff = np.abs(diff)
    finite_abs_diff = abs_diff[finite_mask]

    print("=" * 70)
    print("Wan VAE decode comparison: cuDNN vs Mojo")
    print("=" * 70)
    print(f"Model:       {cudnn['model'][0]}")
    print(
        f"Latent:      shape={tuple(cudnn['shape'])} seed={int(cudnn['seed'])}"
    )
    print()
    print("Per-path decoded tensor statistics:")
    _summarize("cuDNN", cudnn_pixels)
    _summarize("Mojo ", mojo_pixels)
    print()

    print("Absolute difference (cuDNN - Mojo):")
    print(
        f"  max={finite_abs_diff.max():.6f} "
        f"mean={finite_abs_diff.mean():.6f} "
        f"std={finite_abs_diff.std():.6f}"
    )
    print(
        f"  p50={np.percentile(finite_abs_diff, 50):.6f} "
        f"p90={np.percentile(finite_abs_diff, 90):.6f} "
        f"p99={np.percentile(finite_abs_diff, 99):.6f} "
        f"p99.9={np.percentile(finite_abs_diff, 99.9):.6f}"
    )
    print()

    if cudnn_pixels.ndim == 5:
        _, _, t, _, _ = cudnn_pixels.shape
        print(f"Per-frame max abs diff (T={t}):")
        for ti in range(t):
            frame_diff = abs_diff[:, :, ti]
            fmax = float(np.nanmax(frame_diff))
            fmean = float(np.nanmean(frame_diff))
            print(f"  frame {ti:2d}: max={fmax:.6f} mean={fmean:.6f}")
        print()

    # Black-frame diagnostic: after post-processing, x < -0.99 clips to
    # pixel 0 (black). If the Mojo path has a regression that drives
    # outputs negative, the mojo fraction will spike relative to cuDNN.
    clip_lo_cudnn = float((cudnn_pixels < -0.99).mean())
    clip_lo_mojo = float((mojo_pixels < -0.99).mean())
    clip_hi_cudnn = float((cudnn_pixels > 0.99).mean())
    clip_hi_mojo = float((mojo_pixels > 0.99).mean())
    print("Clamp-boundary diagnostic (post-VAE values are clipped to [-1, 1]):")
    print(
        f"  fraction <-0.99 (maps to black): "
        f"cudnn={clip_lo_cudnn:.6f} mojo={clip_lo_mojo:.6f}"
    )
    print(
        f"  fraction > 0.99 (maps to white): "
        f"cudnn={clip_hi_cudnn:.6f} mojo={clip_hi_mojo:.6f}"
    )
    print()

    # Verdict. Thresholds are empirical; tune if needed. bf16 accumulator
    # drift on WAN shapes is typically <0.05 end-to-end per my kernel
    # bench (max |im2col - cuDNN| = 16.0 on raw sums of ~thousands, which
    # collapses to <0.05 after the clamp to [-1, 1]).
    max_diff = float(finite_abs_diff.max())
    if max_diff < 0.05:
        verdict = "PASS"
        rc = 0
    elif max_diff < 0.5:
        verdict = "WARN"
        rc = 1
    else:
        verdict = "FAIL"
        rc = 2

    nan_delta = abs(
        float(np.isnan(cudnn_pixels).mean())
        - float(np.isnan(mojo_pixels).mean())
    )
    if nan_delta > 1e-6:
        verdict = "FAIL"
        rc = 2

    print(f"VERDICT: {verdict} (max abs diff={max_diff:.6f})")
    return rc


def _spawn_worker(cmd_args: list[str], env_override: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_override)
    # Make sure the worker doesn't try to re-orchestrate.
    for var in ("MAX_WAN_VAE_DISABLE_CUDNN",):
        if var not in env_override and var in env:
            env.pop(var)
    subprocess.run(cmd_args, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Wan VAE decode between cuDNN and Mojo paths."
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="HuggingFace repo hosting the Wan VAE (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic random latent (default: %(default)s).",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        default=_DEFAULT_SHAPE,
        metavar=("B", "Z", "T", "H", "W"),
        help="Latent shape (default: %(default)s).",
    )
    parser.add_argument(
        "--encoding",
        default="bfloat16",
        help="Model encoding (default: %(default)s).",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/vae_decode_compare",
        help="Directory for intermediate .npz files (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-cudnn",
        action="store_true",
        help="Don't re-run cuDNN decode if an output file is already present.",
    )
    parser.add_argument(
        "--skip-mojo",
        action="store_true",
        help="Don't re-run Mojo decode if an output file is already present.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip decoding; only compare previously saved outputs.",
    )
    # Hidden subprocess worker mode.
    parser.add_argument("--_inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_output", type=Path, default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args._inner:
        if args._output is None:
            parser.error("--_inner requires --_output")
        _run_decode(
            args.model,
            args.seed,
            tuple(args.shape),
            args._output,
            args.encoding,
        )
        return

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cudnn_path = work_dir / "cudnn.npz"
    mojo_path = work_dir / "mojo.npz"

    worker_cmd_base = [
        sys.executable,
        __file__,
        "--_inner",
        "--model",
        args.model,
        "--seed",
        str(args.seed),
        "--shape",
        *[str(s) for s in args.shape],
        "--encoding",
        args.encoding,
    ]

    if not args.compare_only:
        run_cudnn = not (args.skip_cudnn and cudnn_path.exists())
        run_mojo = not (args.skip_mojo and mojo_path.exists())

        if run_cudnn:
            logger.info(
                "Spawning cuDNN decode (MAX_WAN_VAE_DISABLE_CUDNN unset)"
            )
            _spawn_worker(
                worker_cmd_base + ["--_output", str(cudnn_path)],
                env_override={},
            )
        else:
            logger.info("Reusing existing cuDNN output at %s", cudnn_path)

        if run_mojo:
            logger.info("Spawning Mojo decode (MAX_WAN_VAE_DISABLE_CUDNN=1)")
            _spawn_worker(
                worker_cmd_base + ["--_output", str(mojo_path)],
                env_override={"MAX_WAN_VAE_DISABLE_CUDNN": "1"},
            )
        else:
            logger.info("Reusing existing Mojo output at %s", mojo_path)

    missing = [p for p in (cudnn_path, mojo_path) if not p.exists()]
    if missing:
        print(
            f"ERROR: required output(s) missing: {missing}. "
            "Run without --compare-only first.",
            file=sys.stderr,
        )
        sys.exit(1)

    rc = _compare(cudnn_path, mojo_path)
    sys.exit(rc)


if __name__ == "__main__":
    main()
