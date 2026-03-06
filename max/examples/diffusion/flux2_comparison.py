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

"""FLUX.2 performance comparison: diffusers (PyTorch) vs MAX.

Runs FLUX.2 image generation with both backends, performs warmup runs,
benchmarks with split preprocessing/execution timings, and prints a
side-by-side summary. When --vary-inputs is set, each iteration uses a
different (height, width, num_inference_steps) tuple to stress test
eager-mode recompilation.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Varying-input configurations for recompilation stress testing.
# Each tuple is (height, width, num_inference_steps).
VARIED_CONFIGS: list[tuple[int, int, int]] = [
    (1024, 1024, 50),
    (768, 1360, 40),
    (1360, 768, 30),
    (512, 512, 50),
    (1024, 768, 28),
    (768, 768, 50),
    (1360, 1024, 40),
    (512, 1024, 30),
    (1024, 512, 28),
    (768, 1024, 50),
]


@dataclass
class TimingResult:
    """Collected timings for a single backend."""

    preprocess_durations: list[float] = field(default_factory=list)
    execute_durations: list[float] = field(default_factory=list)
    total_durations: list[float] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FLUX.2 performance: diffusers vs MAX."
    )
    parser.add_argument(
        "--prompt",
        default="dog dancing near the sun",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps (used when --vary-inputs is off).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height (used when --vary-inputs is off).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width (used when --vary-inputs is off).",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=2,
        help="Number of warmup runs (not timed) per backend.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of timed iterations per backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-diffusers",
        action="store_true",
        help="Skip the diffusers (PyTorch) run.",
    )
    parser.add_argument(
        "--skip-max",
        action="store_true",
        help="Skip the MAX run.",
    )
    parser.add_argument(
        "--vary-inputs",
        action="store_true",
        help=(
            "Vary (height, width, num_inference_steps) across iterations to "
            "stress test eager-mode recompilation."
        ),
    )
    return parser.parse_args(argv)


def _iter_configs(
    args: argparse.Namespace, count: int
) -> list[tuple[int, int, int]]:
    """Return (height, width, steps) tuples for *count* iterations."""
    if args.vary_inputs:
        return [VARIED_CONFIGS[i % len(VARIED_CONFIGS)] for i in range(count)]
    return [(args.height, args.width, args.num_inference_steps)] * count


def _load_diffusers_pipeline() -> Any:
    """Load FLUX.2 pipeline via diffusers with optimized attention."""
    import torch
    from diffusers import Flux2Pipeline

    repo_id = "black-forest-labs/FLUX.2-dev"
    pipe = Flux2Pipeline.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.set_attention_backend("native")
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder = torch.compile(
            pipe.text_encoder, mode="max-autotune", fullgraph=True
        )
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2 = torch.compile(
            pipe.text_encoder_2, mode="max-autotune", fullgraph=True
        )
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = torch.compile(pipe.vae, mode="max-autotune", fullgraph=True)
    return pipe


def run_diffusers(args: argparse.Namespace) -> TimingResult:
    """Benchmark FLUX.2 through diffusers. Returns split timings.

    Preprocessing is measured as text encoding (encode_prompt). Execution
    is the remainder: latent prep, denoising loop, and VAE decode.
    """
    import torch

    print("\n=== Diffusers (PyTorch) backend ===")
    print("Loading pipeline...")
    pipe = _load_diffusers_pipeline()

    # Warm up using the same split path (encode_prompt + `pipe(prompt_embeds=)`)
    # that we use for timed iterations. If we warmed up with `pipe(prompt=...)`
    # instead, then `torch.compile` would recompile when it first sees the
    # `prompt_embeds` call signature during timed runs, adding ~50s of overhead.
    warmup_configs = _iter_configs(args, args.num_warmups)
    for i, (h, w, steps) in enumerate(warmup_configs):
        print(f"  warmup {i + 1}/{args.num_warmups} ({w}x{h}, {steps} steps)")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=args.prompt, device=pipe._execution_device
        )
        pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=steps,
            guidance_scale=args.guidance_scale,
            height=h,
            width=w,
            generator=generator,
        )
        torch.cuda.synchronize()

    result = TimingResult()
    timed_configs = _iter_configs(args, args.num_iterations)
    for i, (h, w, steps) in enumerate(timed_configs):
        print(f"  iter {i + 1}/{args.num_iterations} ({w}x{h}, {steps} steps)")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        # Preprocessing: text encoding
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=args.prompt,
            device=pipe._execution_device,
        )
        torch.cuda.synchronize()
        t_preprocess = time.perf_counter() - t0

        # Execution: latent prep + denoising loop + VAE decode
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=steps,
            guidance_scale=args.guidance_scale,
            height=h,
            width=w,
            generator=generator,
        )
        torch.cuda.synchronize()
        t_execute = time.perf_counter() - t1

        result.preprocess_durations.append(t_preprocess)
        result.execute_durations.append(t_execute)
        result.total_durations.append(t_preprocess + t_execute)

    # Free GPU memory before MAX runs.
    import gc

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return result


def _load_max_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load FLUX.2 pipeline via MAX."""
    from typing import cast

    from max.driver import DeviceSpec
    from max.interfaces import PipelineTask
    from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
    from max.pipelines.core import PixelContext
    from max.pipelines.lib import PixelGenerationTokenizer
    from max.pipelines.lib.interfaces import DiffusionPipeline
    from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
    from max.pipelines.lib.pipeline_variants.pixel_generation import (
        PixelGenerationPipeline,
    )

    model_id = "black-forest-labs/FLUX.2-dev"

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=model_id,
            device_specs=[DeviceSpec.accelerator()],
        ),
        runtime=PipelineRuntimeConfig(prefer_module_v3=True),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        prefer_module_v3=config.runtime.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, "No FLUX.2 architecture found in MAX registry."

    diffusers_config = config.model.diffusers_config
    max_length = None
    if (
        diffusers_config is not None
        and (comps := diffusers_config.get("components"))
        and comps.get("tokenizer") is not None
    ):
        max_length = 512  # Flux2-specific override

    tokenizer = PixelGenerationTokenizer(
        model_path=model_id,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=max_length,
    )

    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )
    return pipeline, tokenizer, config


async def _build_max_inputs(
    args: argparse.Namespace,
    tokenizer: Any,
    height: int,
    width: int,
    steps: int,
) -> tuple[Any, Any]:
    """Build MAX pipeline inputs for a single generation."""
    from max.interfaces import PixelGenerationInputs, RequestID
    from max.interfaces.provider_options import (
        ImageProviderOptions,
        ProviderOptions,
    )
    from max.interfaces.request import OpenResponsesRequest
    from max.interfaces.request.open_responses import OpenResponsesRequestBody
    from max.pipelines.core import PixelContext

    body = OpenResponsesRequestBody(
        model="black-forest-labs/FLUX.2-dev",
        input=args.prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                height=height,
                width=width,
                steps=steps,
                guidance_scale=args.guidance_scale,
            )
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )
    return inputs, context


def run_max(args: argparse.Namespace) -> TimingResult:
    """Benchmark FLUX.2 through MAX with split preprocess/execute timings."""
    print("\n=== MAX backend ===")
    print("Loading pipeline...")
    pipeline, tokenizer, _config = _load_max_pipeline(args)

    warmup_configs = _iter_configs(args, args.num_warmups)
    for i, (h, w, steps) in enumerate(warmup_configs):
        print(f"  warmup {i + 1}/{args.num_warmups} ({w}x{h}, {steps} steps)")
        inputs, _ = asyncio.run(_build_max_inputs(args, tokenizer, h, w, steps))
        pipeline.execute(inputs)

    result = TimingResult()
    timed_configs = _iter_configs(args, args.num_iterations)
    for i, (h, w, steps) in enumerate(timed_configs):
        print(f"  iter {i + 1}/{args.num_iterations} ({w}x{h}, {steps} steps)")

        # Preprocessing: tokenization + context + input building
        t0 = time.perf_counter()
        inputs, _ = asyncio.run(_build_max_inputs(args, tokenizer, h, w, steps))
        t_preprocess = time.perf_counter() - t0

        # Model execution
        t1 = time.perf_counter()
        pipeline.execute(inputs)
        t_execute = time.perf_counter() - t1

        result.preprocess_durations.append(t_preprocess)
        result.execute_durations.append(t_execute)
        result.total_durations.append(t_preprocess + t_execute)

    return result


def _print_summary(label: str, result: TimingResult) -> None:
    """Print timing summary for one backend."""
    n = len(result.total_durations)
    print(f"  {label}:")
    print(f"    iterations : {n}")

    if result.preprocess_durations:
        mean_pp = statistics.mean(result.preprocess_durations)
        std_pp = statistics.stdev(result.preprocess_durations) if n > 1 else 0.0
        print(f"    preprocess : {mean_pp:8.2f} s  (std {std_pp:.2f} s)")

    mean_ex = statistics.mean(result.execute_durations)
    std_ex = statistics.stdev(result.execute_durations) if n > 1 else 0.0
    print(f"    execute    : {mean_ex:8.2f} s  (std {std_ex:.2f} s)")

    mean_tot = statistics.mean(result.total_durations)
    std_tot = statistics.stdev(result.total_durations) if n > 1 else 0.0
    mn = min(result.total_durations)
    mx = max(result.total_durations)
    print(f"    total      : {mean_tot:8.2f} s  (std {std_tot:.2f} s)")
    print(f"    min        : {mn:8.2f} s")
    print(f"    max        : {mx:8.2f} s")


def _print_per_iteration(
    label: str, result: TimingResult, configs: list[tuple[int, int, int]]
) -> None:
    """Print per-iteration breakdown when --vary-inputs is used."""
    print(f"\n  {label} per-iteration breakdown:")
    print(
        f"    {'Config':>15s}  {'Preprocess':>10s}"
        f"  {'Execute':>10s}  {'Total':>10s}"
    )
    for i, (h, w, steps) in enumerate(configs):
        pp = (
            result.preprocess_durations[i]
            if result.preprocess_durations
            else 0.0
        )
        ex = result.execute_durations[i]
        tot = result.total_durations[i]
        print(
            f"    {w:4d}x{h:<4d} {steps:2d}s"
            f"  {pp:10.2f}  {ex:10.2f}  {tot:10.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    diffusers_result: TimingResult | None = None
    max_result: TimingResult | None = None

    if not args.skip_diffusers:
        try:
            diffusers_result = run_diffusers(args)
        except Exception as e:
            print(f"ERROR running diffusers: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    if not args.skip_max:
        try:
            max_result = run_max(args)
        except Exception as e:
            print(f"ERROR running MAX: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    # Summary header
    print("\n" + "=" * 60)
    print("FLUX.2 Performance Comparison")
    print("=" * 60)
    print(f"  prompt           : {args.prompt!r}")
    if args.vary_inputs:
        print("  inputs           : varied (recompilation stress test)")
    else:
        print(f"  resolution       : {args.width}x{args.height}")
        print(f"  inference steps  : {args.num_inference_steps}")
    print(f"  guidance scale   : {args.guidance_scale}")
    print(f"  warmup runs      : {args.num_warmups}")
    print()

    timed_configs = _iter_configs(args, args.num_iterations)

    if diffusers_result:
        _print_summary("Diffusers (PyTorch)", diffusers_result)
    if max_result:
        _print_summary("MAX", max_result)

    if args.vary_inputs:
        if diffusers_result:
            _print_per_iteration(
                "Diffusers (PyTorch)", diffusers_result, timed_configs
            )
        if max_result:
            _print_per_iteration("MAX", max_result, timed_configs)

    if diffusers_result and max_result:
        d_mean = statistics.mean(diffusers_result.total_durations)
        m_mean = statistics.mean(max_result.total_durations)
        speedup = d_mean / m_mean if m_mean > 0 else float("inf")
        print()
        print(f"  Speedup (MAX vs Diffusers): {speedup:.2f}x")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
