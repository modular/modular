#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""Wan video generation full metric: benchmarks all model variants via MAX.

Runs all Wan model variants (2.2/2.1, T2V/I2V) at 720p with
multiple resolutions to verify symbolic seq_len recompilation behavior.

Usage:
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 \
    ./bazelw run //max/examples/diffusion:full_metric

    # Specific model only
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 \
    ./bazelw run //max/examples/diffusion:full_metric -- \
        --model wan2.2-t2v-a14b
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Track child processes so we can kill them on exit
_child_procs: list[subprocess.Popen[str]] = []


def _cleanup_children() -> None:
    for p in _child_procs:
        try:
            p.kill()
        except OSError:
            pass


atexit.register(_cleanup_children)


def _signal_handler(*_: object) -> None:
    _cleanup_children()
    sys.exit(1)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("full_metric")

NUM_STEPS = 40

T2V_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)
I2V_PROMPT = "A cat surfing on a wave"
I2V_IMAGE_URL = (
    "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B"
    "/resolve/main/examples/i2v_input.JPG"
)
NEGATIVE_PROMPT = "low quality"

RESOLUTIONS: list[dict[str, str | int]] = [
    {"height": 720, "width": 1280, "num_frames": 81, "label": "1280x720"},
    {"height": 1280, "width": 720, "num_frames": 81, "label": "720x1280"},
]


@dataclass
class ModelConfig:
    name: str
    repo_id: str
    mode: str
    guidance_scale: float
    guidance_scale_2: float | None


MODELS: dict[str, ModelConfig] = {
    "wan2.2-t2v-a14b": ModelConfig(
        name="wan2.2-t2v-a14b",
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        mode="t2v",
        guidance_scale=4.0,
        guidance_scale_2=3.0,
    ),
    "wan2.2-i2v-a14b": ModelConfig(
        name="wan2.2-i2v-a14b",
        repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        mode="i2v",
        guidance_scale=4.0,
        guidance_scale_2=3.0,
    ),
    "wan2.1-t2v-14b": ModelConfig(
        name="wan2.1-t2v-14b",
        repo_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        mode="t2v",
        guidance_scale=5.0,
        guidance_scale_2=None,
    ),
    "wan2.1-i2v-14b": ModelConfig(
        name="wan2.1-i2v-14b",
        repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        mode="i2v",
        guidance_scale=5.0,
        guidance_scale_2=None,
    ),
}


@dataclass
class TimingResult:
    model: str
    label: str
    e2e_seconds: float
    components: dict[str, float] = field(default_factory=dict)


def _parse_profiling(output: str) -> dict[str, float]:
    components: dict[str, float] = {}
    in_methods = False
    for line in output.splitlines():
        if "Method Timings:" in line:
            in_methods = True
            continue
        if in_methods and "===" in line:
            break
        if in_methods:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    total_ms = float(parts[-2])
                    float(parts[-1])  # validate avg
                    int(parts[-3])  # validate calls
                    name = " ".join(parts[:-3])
                    components[name] = total_ms / 1000.0
                except (ValueError, IndexError):
                    pass
    return components


def run(models: list[ModelConfig], output_dir: Path) -> list[TimingResult]:
    results: list[TimingResult] = []
    bazel_target = "//max/examples/diffusion:simple_offline_video_generation"

    # Build --resolutions specs
    res_specs = [
        f"{res['width']}x{res['height']}x{res['num_frames']}"
        for res in RESOLUTIONS
    ]

    for model_idx, model in enumerate(models):
        tag = model.name
        log.info(
            "(%d/%d) %s — %d resolutions",
            model_idx + 1, len(models), tag, len(RESOLUTIONS),
        )
        t0 = time.perf_counter()

        prompt = I2V_PROMPT if model.mode == "i2v" else T2V_PROMPT
        video_path = output_dir / f"{model.name}.mp4"
        env = os.environ.copy()
        chunk_pct = env.get(
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", ""
        )
        env_prefix: list[str] = []
        if chunk_pct:
            env_prefix = [
                "env",
                f"MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT={chunk_pct}",
            ]
        cmd = [
            *env_prefix,
            "./bazelw",
            "run",
            bazel_target,
            "--",
            "--model",
            model.repo_id,
            "--prompt",
            prompt,
            "--negative-prompt",
            NEGATIVE_PROMPT,
            "--resolutions",
            *res_specs,
            "--num-inference-steps",
            str(NUM_STEPS),
            "--guidance-scale",
            str(model.guidance_scale),
            "--output",
            str(video_path),
        ]
        if model.guidance_scale_2 is not None:
            cmd += ["--guidance-scale-2", str(model.guidance_scale_2)]
        if model.mode == "i2v":
            cmd += ["--input-image", I2V_IMAGE_URL]


        # Stream stdout+stderr in real-time (tqdm visible)
        output_file = output_dir / f"{model.name}.log"
        with open(output_file, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(Path(__file__).resolve().parents[3]),
            )
            _child_procs.append(proc)
            full_lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                full_lines.append(line)
                # Filter bazel noise from display, but keep for parsing
                if not line.startswith(("INFO:", "Loading:", "Analyzing:")):
                    sys.stdout.write(line)
                    sys.stdout.flush()
            proc.wait()
        elapsed = time.perf_counter() - t0
        full = "".join(full_lines)

        if proc.returncode != 0:
            log.error("%s FAILED (%.0fs)", tag, elapsed)
            for res in RESOLUTIONS:
                results.append(
                    TimingResult(model.name, str(res["label"]), -1.0)
                )
            continue

        # Parse profiling — covers all resolutions in one report
        components = _parse_profiling(full)
        e2e = components.pop("E2E execute", components.pop("E2E", -1.0))
        log.info("%s — E2E %.1fs (total %.0fs)", tag, e2e, elapsed)
        for res in RESOLUTIONS:
            results.append(
                TimingResult(model.name, str(res["label"]), e2e, components)
            )
    return results


def _gpu_name() -> str:
    try:
        return (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        return "unknown GPU"


def print_summary(all_results: list[TimingResult]) -> None:
    gpu = _gpu_name()
    model_names = list(dict.fromkeys(r.model for r in all_results))

    print(f"\n{'=' * 60}")
    print(f"  Wan Full Metric — {gpu}, {NUM_STEPS} steps")
    print(f"{'=' * 60}\n")

    hdr = f"{'Model':<22} {'Resolution':<12} {'E2E (s)':>10}"
    print(hdr)
    print("-" * len(hdr))

    by_key: dict[tuple[str, str], TimingResult] = {}
    for r in all_results:
        by_key[(r.model, r.label)] = r

    for model_name in model_names:
        e2e_vals: list[float] = []
        for res in RESOLUTIONS:
            label = str(res["label"])
            result = by_key.get((model_name, label))
            e2e = result.e2e_seconds if result else -1
            e2e_str = f"{e2e:>10.1f}" if e2e > 0 else f"{'FAIL':>10}"
            print(f"{model_name:<22} {label:<12} {e2e_str}")
            if e2e > 0:
                e2e_vals.append(e2e)
        if e2e_vals:
            avg = sum(e2e_vals) / len(e2e_vals)
            print(f"{model_name:<22} {'avg':<12} {avg:>10.1f}")
        print()

    print(f"{'=' * 60}")
    print("  Component Breakdown (seconds)")
    print(f"{'=' * 60}\n")

    for model_name in model_names:
        # Per-resolution breakdown
        for res in RESOLUTIONS:
            label = str(res["label"])
            result = by_key.get((model_name, label))
            if not result or not result.components:
                continue
            print(f"  {model_name} / {label}:")
            for comp, secs in sorted(result.components.items()):
                print(f"    {comp:<30} {secs:>10.3f}s")
            print()

        # Average across resolutions
        comp_totals: dict[str, list[float]] = {}
        for res in RESOLUTIONS:
            label = str(res["label"])
            result = by_key.get((model_name, label))
            if not result or not result.components:
                continue
            for comp, secs in result.components.items():
                comp_totals.setdefault(comp, []).append(secs)
        if comp_totals:
            print(f"  {model_name} / avg:")
            for comp in sorted(comp_totals):
                vals = comp_totals[comp]
                print(f"    {comp:<30} {sum(vals)/len(vals):>10.3f}s")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wan video generation full metric (MAX)"
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help=f"Model(s) to benchmark. Choices: {', '.join(MODELS)}. "
        "Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/wan_full_metric",
    )
    args = parser.parse_args()

    selected = [MODELS[m] for m in (args.model or MODELS.keys())]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = run(selected, output_dir)
    print_summary(all_results)


if __name__ == "__main__":
    main()
