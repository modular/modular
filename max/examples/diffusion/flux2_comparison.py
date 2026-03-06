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
benchmarks with microbench, and prints a side-by-side timing summary.

Usage (from a venv with diffusers, torch, microbench, kernels, and max installed):

    python3 max/examples/diffusion/flux2_comparison.py \
        --prompt "dog dancing near the sun" \
        --num-inference-steps 50 \
        --num-warmups 2 \
        --num-iterations 3
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from typing import Any

from microbench import MicroBench


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
        help="Number of denoising steps.",
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
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width in pixels.",
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
    return parser.parse_args(argv)


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


def _run_diffusers_once(pipe: Any, args: argparse.Namespace) -> Any:
    """Run a single diffusers inference pass and return the image."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )
    return result.images[0]


def run_diffusers(args: argparse.Namespace) -> list[float]:
    """Benchmark FLUX.2 through diffusers. Returns list of durations (s)."""
    print("\n=== Diffusers (PyTorch) backend ===")
    print("Loading pipeline...")
    pipe = _load_diffusers_pipeline()

    for i in range(args.num_warmups):
        print(f"  warmup {i + 1}/{args.num_warmups}")
        _run_diffusers_once(pipe, args)

    bench = MicroBench(iterations=args.num_iterations)

    @bench
    def diffusers_generate() -> None:
        _run_diffusers_once(pipe, args)

    diffusers_generate()
    results = bench.get_results()
    durations = _extract_durations(results)

    # Free GPU memory before MAX runs
    import gc

    import torch

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return durations


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
        # FLUX.2 is only available as a V3 module currently, even though it
        # doesn't follow the typical V3 naming conventions, so this flag is
        # currently a no-op. I expect it to be used in the future.
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
    args: argparse.Namespace, tokenizer: Any, config: Any
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
                height=args.height,
                width=args.width,
                steps=args.num_inference_steps,
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


def run_max(args: argparse.Namespace) -> list[float]:
    """Benchmark FLUX.2 through MAX. Returns list of durations (s)."""
    print("\n=== MAX backend ===")
    print("Loading pipeline...")
    pipeline, tokenizer, config = _load_max_pipeline(args)

    for i in range(args.num_warmups):
        print(f"  warmup {i + 1}/{args.num_warmups}")
        inputs, _ = asyncio.run(_build_max_inputs(args, tokenizer, config))
        pipeline.execute(inputs)

    # Build inputs once outside the timed loop
    # TODO: Figure out if we want to build inside or outside the loop. The
    # difference was neglibable between runs in my testing (29.40s vs 29.46s).
    inputs, _ = asyncio.run(_build_max_inputs(args, tokenizer, config))

    bench = MicroBench(iterations=args.num_iterations)

    @bench
    def max_generate() -> None:
        pipeline.execute(inputs)

    max_generate()
    results = bench.get_results()
    durations = _extract_durations(results)
    return durations


def _extract_durations(results: Any) -> list[float]:
    """Extract run_durations from microbench results (DataFrame or StringIO)."""
    import pandas as pd

    if isinstance(results, pd.DataFrame):
        durations: list[float] = []
        for cell in results["run_durations"]:
            if isinstance(cell, list):
                durations.extend(cell)
            else:
                durations.append(float(cell))
        return durations

    # Fallback for StringIO
    import json

    results.seek(0)
    durations = []
    for line in results:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        durations.extend(data.get("run_durations", []))
    return durations


def _print_summary(
    label: str,
    durations: list[float],
) -> None:
    """Print timing summary for one backend."""
    n = len(durations)
    mean = statistics.mean(durations)
    std = statistics.stdev(durations) if n > 1 else 0.0
    mn = min(durations)
    mx = max(durations)
    print(f"  {label}:")
    print(f"    iterations : {n}")
    print(f"    mean       : {mean:8.2f} s")
    print(f"    std        : {std:8.2f} s")
    print(f"    min        : {mn:8.2f} s")
    print(f"    max        : {mx:8.2f} s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    diffusers_durations: list[float] = []
    max_durations: list[float] = []

    if not args.skip_diffusers:
        try:
            diffusers_durations = run_diffusers(args)
        except Exception as e:
            print(f"ERROR running diffusers: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    if not args.skip_max:
        try:
            max_durations = run_max(args)
        except Exception as e:
            print(f"ERROR running MAX: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("FLUX.2 Performance Comparison")
    print("=" * 50)
    print(f"  prompt           : {args.prompt!r}")
    print(f"  resolution       : {args.width}x{args.height}")
    print(f"  inference steps  : {args.num_inference_steps}")
    print(f"  guidance scale   : {args.guidance_scale}")
    print(f"  warmup runs      : {args.num_warmups}")
    print()

    if diffusers_durations:
        _print_summary("Diffusers (PyTorch)", diffusers_durations)
    if max_durations:
        _print_summary("MAX", max_durations)

    if diffusers_durations and max_durations:
        d_mean = statistics.mean(diffusers_durations)
        m_mean = statistics.mean(max_durations)
        speedup = d_mean / m_mean if m_mean > 0 else float("inf")
        print()
        print(f"  Speedup (MAX vs Diffusers): {speedup:.2f}x")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
