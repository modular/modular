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
side-by-side summary. Supports both text-to-image and image-to-image modes.
When `--vary-inputs` is set, each iteration uses a different
`(height, width, num_inference_steps)` tuple. When `--vary-prompts` is set,
each iteration cycles through a fixed set of prompts with different sequence
lengths.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np
import torch
from diffusers import Flux2Pipeline
from max.driver import DeviceSpec
from max.interfaces import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    InputImageContent,
    InputTextContent,
    OpenResponsesRequestBody,
    OutputImageContent,
    UserMessage,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from PIL import Image

# Varying-input configurations for recompilation stress testing.
# Each tuple is `(height, width, num_inference_steps)`.
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
    (256, 256, 15),
]

# Fixed prompts of varying sequence lengths for text-encoder stress testing.
# Ordered roughly short → long so iteration logs are easy to follow.
VARIED_PROMPTS: list[str] = [
    (
        "Black cat hiding behind a watermelon slice, professional studio"
        " shot, bright red and turquoise background with summer mystery vibe"
    ),
    (
        "Dog wrapped in white towel after bath, photographed with direct"
        " flash and high exposure, fur wet details sharply visible, editorial"
        " raw portrait, cinematic harsh flash lighting, intimate humorous"
        " documentary style"
    ),
    (
        "A small red propeller plane banking sharply between massive jungle"
        " trees in a bright anime style, with midday sun illuminating lush"
        " green foliage and waterfalls cascading in the background."
    ),
    (
        "A businessman in a charcoal grey suit resting his arms on a bamboo"
        " railing at a secluded beach in the Philippines, cigarette glowing"
        " between his lips, illustrated in a vintage Japanese woodblock print"
        " style with soft pastel tones, calm turquoise waters, and a hazy"
        " afternoon sky."
    ),
    (
        "A group of baby penguins in a trampoline park, having the time of"
        " their lives, 80s vintage photo"
    ),
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
    parser.add_argument(
        "--enable-fbc",
        action="store_true",
        help="Enable first-block caching (FBC) for the MAX diffusion pipeline.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.08,
        help=(
            "Residual threshold for step-cache early stopping in the "
            "denoising loop (default: 0.08)."
        ),
    )
    parser.add_argument(
        "--vary-prompts",
        action="store_true",
        help=(
            "Cycle through fixed prompts of varying sequence lengths across "
            "iterations to stress text-encoder recompilation/performance."
        ),
    )
    parser.add_argument(
        "--image-to-image",
        action="store_true",
        help=(
            "Run image-to-image mode instead of text-to-image. Requires "
            "--input-image or generates a synthetic input image."
        ),
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help=(
            "Path to an input image file for image-to-image mode. "
            "If --image-to-image is set but no path is provided, a "
            "synthetic gradient image is generated."
        ),
    )
    parser.add_argument(
        "--taylorseer",
        action="store_true",
        help="Enable TaylorSeer cache optimization for the MAX pipeline.",
    )
    parser.add_argument(
        "--taylorseer-cache-interval",
        type=int,
        default=None,
        help="Steps between full computations for TaylorSeer (model default if unset).",
    )
    parser.add_argument(
        "--taylorseer-warmup-steps",
        type=int,
        default=None,
        help="Warmup steps for TaylorSeer factor gathering (model default if unset).",
    )
    parser.add_argument(
        "--taylorseer-max-order",
        type=int,
        default=None,
        choices=[1, 2],
        help="Taylor expansion order: 1=linear, 2=quadratic (model default if unset).",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving generated images to disk.",
    )
    return parser.parse_args(argv)


def _iter_configs(
    args: argparse.Namespace, count: int
) -> list[tuple[int, int, int]]:
    """Return (height, width, steps) tuples for *count* iterations."""
    if args.vary_inputs:
        return [VARIED_CONFIGS[i % len(VARIED_CONFIGS)] for i in range(count)]
    return [(args.height, args.width, args.num_inference_steps)] * count


def _iter_prompts(args: argparse.Namespace, count: int) -> list[str]:
    """Return prompt strings for *count* iterations."""
    if args.vary_prompts:
        return [VARIED_PROMPTS[i % len(VARIED_PROMPTS)] for i in range(count)]
    return [args.prompt] * count


def _warmup_configs_and_prompts(
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Return warmup (config, prompt) lists that cover every unique timed combo.

    When `--vary-inputs` or `--vary-prompts` is set, torch.compile recompiles for
    each new input shape / sequence length. The warmup set must cover every
    unique `(config, prompt)` pair that timed iterations will use, otherwise the
    first timed occurrence eats a recompilation penalty.

    Structure: one compilation pass that covers each unique combo once (this
    absorbs `torch.compile` tracing + autotuning), then `num_warmups` additional
    steady-state passes that cycle through the combos evenly. This keeps the
    effective warmup count fair across backends, since both see exactly
    `num_warmups` post-compilation iterations.
    """
    timed_configs = _iter_configs(args, args.num_iterations)
    timed_prompts = _iter_prompts(args, args.num_iterations)

    # Deduplicate while preserving order.
    seen: set[tuple[tuple[int, int, int], str]] = set()
    unique_configs: list[tuple[int, int, int]] = []
    unique_prompts: list[str] = []
    for cfg, prompt in zip(timed_configs, timed_prompts, strict=False):
        key = (cfg, prompt)
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
            unique_prompts.append(prompt)

    # Compilation pass (1x each unique combo) + `num_warmups` steady-state
    # passes.
    n_unique = len(unique_configs)
    total = n_unique + args.num_warmups
    warmup_configs = [unique_configs[i % n_unique] for i in range(total)]
    warmup_prompts = [unique_prompts[i % n_unique] for i in range(total)]
    return warmup_configs, warmup_prompts


def _print_gpu_info() -> None:
    """Print GPU name, VRAM, and CUDA build version."""
    if not torch.cuda.is_available():
        print("  GPU: not available (CPU mode)")
        return
    name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_memory / (1024**3)
    cuda_version = torch.version.cuda or "N/A"
    print(f"  GPU            : {name}")
    print(f"  VRAM           : {total_gb:.1f} GB")
    print(f"  CUDA (build)   : {cuda_version}")


def _truncate_prompt(prompt: str, max_len: int = 40) -> str:
    """Return a display-friendly truncated prompt."""
    if len(prompt) <= max_len:
        return prompt
    return prompt[: max_len - 1] + "…"


def _load_input_image(args: argparse.Namespace) -> Image.Image:
    """Load or generate an input image for image-to-image mode.

    If `--input-image` is provided, loads it from disk. Otherwise,
    generates a synthetic gradient image at the requested resolution.
    """
    if args.input_image is not None:
        img = Image.open(args.input_image).convert("RGB")
        print(
            f"  Loaded input image: {args.input_image} ({img.width}x{img.height})"
        )
        return img

    # Generate a synthetic gradient image.
    w, h = args.width, args.height
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]  # R
    arr[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]  # G
    arr[:, :, 2] = 128  # B
    img = Image.fromarray(arr)
    print(f"  Generated synthetic input image ({w}x{h})")
    return img


def _image_to_data_uri(img: Image.Image) -> str:
    """Encode a PIL image as a JPEG base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _images_to_jpeg_base64(images: list[Any]) -> list[str]:
    """Convert PIL images to JPEG-encoded base64 strings.

    This mirrors the post-processing that MAX performs internally
    (numpy → PIL → JPEG → base64) so that timing comparisons are fair.
    """
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        result.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return result


def _build_image_filename(
    backend: str,
    width: int,
    height: int,
    steps: int,
    iteration: int,
    args: argparse.Namespace,
) -> str:
    """Build a descriptive image filename encoding all relevant settings."""
    parts = [
        f"output_{backend}_bf16_seed{args.seed}_{width}x{height}_{steps}steps"
    ]
    if args.enable_fbc:
        parts.append(f"fbc_thresh{args.residual_threshold}")
    if args.taylorseer:
        interval = args.taylorseer_cache_interval or "default"
        warmup = args.taylorseer_warmup_steps or "default"
        order = args.taylorseer_max_order or "default"
        parts.append(f"taylorseer_i{interval}_w{warmup}_o{order}")
    parts.append(f"iter{iteration}")
    return "_".join(parts) + ".png"


def _load_diffusers_pipeline() -> Any:
    """Load FLUX.2 pipeline via diffusers with optimized attention."""
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


def run_diffusers(
    args: argparse.Namespace,
    input_image: Image.Image | None = None,
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark FLUX.2 through diffusers. Returns split timings.

    Preprocessing is measured as text encoding (encode_prompt). Execution
    is the remainder: latent prep, denoising loop, VAE decode, and
    JPEG + base64 encoding (to match MAX's post-processing).

    When *input_image* is provided the pipeline runs in image-to-image
    mode, passing the image (and omitting explicit height/width so that
    the output matches the input dimensions).
    """
    mode = "image-to-image" if input_image is not None else "text-to-image"
    print(f"\n=== Diffusers (PyTorch) backend ({mode}) ===")
    print("Loading pipeline...")
    pipe = _load_diffusers_pipeline()

    def _make_pipe_kwargs(
        prompt_embeds: Any,
        h: int,
        w: int,
        steps: int,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        """Build kwargs dict for the diffusers pipeline call."""
        kwargs: dict[str, Any] = {
            "prompt_embeds": prompt_embeds,
            "num_inference_steps": steps,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
        }
        if input_image is not None:
            kwargs["image"] = input_image
        else:
            kwargs["height"] = h
            kwargs["width"] = w
        return kwargs

    # Warm up using the same split path (encode_prompt + `pipe(prompt_embeds=)`)
    # that we use for timed iterations. If we warmed up with `pipe(prompt=...)`
    # instead, then `torch.compile` would recompile when it first sees the
    # `prompt_embeds` call signature during timed runs, adding ~50s of overhead.
    warmup_configs, warmup_prompts = _warmup_configs_and_prompts(args)
    n_warmups = len(warmup_configs)
    for i, ((h, w, steps), prompt) in enumerate(
        zip(warmup_configs, warmup_prompts, strict=False)
    ):
        print(
            f"  warmup {i + 1}/{n_warmups} ({w}x{h}, {steps} steps,"
            f" prompt={_truncate_prompt(prompt)!r})"
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt, device=pipe._execution_device
        )
        output = pipe(
            **_make_pipe_kwargs(prompt_embeds, h, w, steps, generator)
        )
        _images_to_jpeg_base64(output.images)
        torch.cuda.synchronize()

    result = TimingResult()
    timed_configs = _iter_configs(args, args.num_iterations)
    timed_prompts = _iter_prompts(args, args.num_iterations)

    for i, ((h, w, steps), prompt) in enumerate(
        zip(timed_configs, timed_prompts, strict=False)
    ):
        print(
            f"  iter {i + 1}/{args.num_iterations} ({w}x{h}, {steps} steps,"
            f" prompt={_truncate_prompt(prompt)!r})"
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        # Preprocessing: text encoding
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=pipe._execution_device,
        )
        torch.cuda.synchronize()
        t_preprocess = time.perf_counter() - t0

        # Execution: latent prep + denoising loop + VAE decode + JPEG encode
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        output = pipe(
            **_make_pipe_kwargs(prompt_embeds, h, w, steps, generator)
        )
        _images_to_jpeg_base64(output.images)
        torch.cuda.synchronize()
        t_execute = time.perf_counter() - t1

        result.preprocess_durations.append(t_preprocess)
        result.execute_durations.append(t_execute)
        result.total_durations.append(t_preprocess + t_execute)

        # Save image (outside timing)
        if output_dir is not None and output.images:
            fname = _build_image_filename("torch", w, h, steps, i, args)
            output.images[0].save(os.path.join(output_dir, fname))
            print(f"    saved: {fname}")

    # Free GPU memory before MAX runs.
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return result


def _load_max_pipeline(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load FLUX.2 pipeline via MAX."""
    model_id = "black-forest-labs/FLUX.2-dev"

    config = PipelineConfig(
        model=MAXModelConfig(
            model_path=model_id,
            device_specs=[DeviceSpec.accelerator()],
        ),
        runtime=PipelineRuntimeConfig(
            prefer_module_v3=True,
            enable_fbc=args.enable_fbc,
        ),
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

    cache_config = DenoisingCacheConfig(
        first_block_caching=args.enable_fbc,
        residual_threshold=args.residual_threshold if args.enable_fbc else None,
        taylorseer=args.taylorseer,
        taylorseer_cache_interval=args.taylorseer_cache_interval,
        taylorseer_warmup_steps=args.taylorseer_warmup_steps,
        taylorseer_max_order=args.taylorseer_max_order,
    )

    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
        cache_config=cache_config,
    )
    return pipeline, tokenizer, config


async def _build_max_inputs(
    args: argparse.Namespace,
    tokenizer: Any,
    prompt: str,
    height: int,
    width: int,
    steps: int,
    input_image_data_uri: str | None = None,
) -> tuple[Any, Any]:
    """Build MAX pipeline inputs for a single generation.

    When input_image_data_uri is provided the request is constructed with
    a `UserMessage` containing both text and image content, triggering the
    image-to-image code path.
    """
    if input_image_data_uri is not None:
        # Image-to-image: wrap prompt + image in a UserMessage.
        request_input: Any = [
            UserMessage(
                role="user",
                content=[
                    InputTextContent(type="input_text", text=prompt),
                    InputImageContent(
                        type="input_image",
                        image_url=input_image_data_uri,
                    ),
                ],
            )
        ]
    else:
        request_input = prompt

    body = OpenResponsesRequestBody(
        model="black-forest-labs/FLUX.2-dev",
        input=request_input,
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


def run_max(
    args: argparse.Namespace,
    input_image: Image.Image | None = None,
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark FLUX.2 through MAX with split preprocess/execute timings.

    When `input_image` is provided the pipeline runs in image-to-image
    mode.
    """
    mode = "image-to-image" if input_image is not None else "text-to-image"
    print(f"\n=== MAX backend ({mode}) ===")
    print("Loading pipeline...")
    pipeline, tokenizer, _config = _load_max_pipeline(args)

    # Pre-encode the input image once so it can be reused across iterations.
    input_image_data_uri: str | None = None
    if input_image is not None:
        input_image_data_uri = _image_to_data_uri(input_image)

    warmup_configs, warmup_prompts = _warmup_configs_and_prompts(args)
    n_warmups = len(warmup_configs)
    for i, ((h, w, steps), prompt) in enumerate(
        zip(warmup_configs, warmup_prompts, strict=False)
    ):
        print(
            f"  warmup {i + 1}/{n_warmups} ({w}x{h}, {steps} steps,"
            f" prompt={_truncate_prompt(prompt)!r})"
        )
        inputs, _ = asyncio.run(
            _build_max_inputs(
                args, tokenizer, prompt, h, w, steps, input_image_data_uri
            )
        )
        pipeline.execute(inputs)

    result = TimingResult()
    timed_configs = _iter_configs(args, args.num_iterations)
    timed_prompts = _iter_prompts(args, args.num_iterations)

    for i, ((h, w, steps), prompt) in enumerate(
        zip(timed_configs, timed_prompts, strict=False)
    ):
        print(
            f"  iter {i + 1}/{args.num_iterations} ({w}x{h}, {steps} steps,"
            f" prompt={_truncate_prompt(prompt)!r})"
        )

        # Preprocessing: tokenization + context + input building
        t0 = time.perf_counter()
        inputs, context = asyncio.run(
            _build_max_inputs(
                args, tokenizer, prompt, h, w, steps, input_image_data_uri
            )
        )
        t_preprocess = time.perf_counter() - t0

        # Model execution
        t1 = time.perf_counter()
        outputs = pipeline.execute(inputs)
        t_execute = time.perf_counter() - t1

        result.preprocess_durations.append(t_preprocess)
        result.execute_durations.append(t_execute)
        result.total_durations.append(t_preprocess + t_execute)

        # Save image (outside timing)
        if output_dir is not None:
            output = outputs[context.request_id]
            output = asyncio.run(tokenizer.postprocess(output))
            if output.output:
                for img_content in output.output:
                    if (
                        isinstance(img_content, OutputImageContent)
                        and img_content.image_data
                    ):
                        image_bytes = base64.b64decode(img_content.image_data)
                        img = Image.open(io.BytesIO(image_bytes))
                        fname = _build_image_filename(
                            "max", w, h, steps, i, args
                        )
                        img.save(os.path.join(output_dir, fname))
                        print(f"    saved: {fname}")
                        break  # Save only the first image per iteration

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
    label: str,
    result: TimingResult,
    configs: list[tuple[int, int, int]],
    prompts: list[str],
) -> None:
    """Print per-iteration breakdown when `--vary-inputs`/`--vary-prompts` is used."""
    print(f"\n  {label} per-iteration breakdown:")
    print(
        f"    {'Config':>15s}  {'Prompt':>42s}"
        f"  {'Preprocess':>10s}  {'Execute':>10s}  {'Total':>10s}"
    )
    for i, ((h, w, steps), prompt) in enumerate(
        zip(configs, prompts, strict=False)
    ):
        pp = (
            result.preprocess_durations[i]
            if result.preprocess_durations
            else 0.0
        )
        ex = result.execute_durations[i]
        tot = result.total_durations[i]
        print(
            f"    {w:4d}x{h:<4d} {steps:2d}s"
            f"  {_truncate_prompt(prompt, 42):>42s}"
            f"  {pp:10.2f}  {ex:10.2f}  {tot:10.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Auto-enable image-to-image mode when --input-image is provided.
    if args.input_image and not args.image_to_image:
        args.image_to_image = True

    # Load the input image once if running in image-to-image mode.
    input_image: Image.Image | None = None
    if args.image_to_image:
        input_image = _load_input_image(args)

    # Create output directory for saved images (unless --no-output).
    output_dir: str | None = None
    if not args.no_output:
        output_dir = "flux_comparison"
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving images to: {output_dir}/")

    diffusers_result: TimingResult | None = None
    max_result: TimingResult | None = None

    if not args.skip_diffusers:
        try:
            diffusers_result = run_diffusers(args, input_image, output_dir)
        except Exception as e:
            print(f"ERROR running diffusers: {e}", file=sys.stderr)
            traceback.print_exc()

    if not args.skip_max:
        try:
            max_result = run_max(args, input_image, output_dir)
        except Exception as e:
            print(f"ERROR running MAX: {e}", file=sys.stderr)
            traceback.print_exc()

    # Summary header
    mode = "Image-to-Image" if args.image_to_image else "Text-to-Image"
    print("\n" + "=" * 60)
    print(f"FLUX.2 {mode} Performance Comparison — {datetime.now():%Y-%m-%d}")
    print("=" * 60)
    _print_gpu_info()
    print(f"  mode             : {mode}")
    if args.image_to_image:
        src = args.input_image or "synthetic gradient"
        print(f"  input image      : {src}")
    if args.vary_prompts:
        print("  prompts          : varied (seq-length stress test)")
    else:
        print(f"  prompt           : {args.prompt!r}")
    if args.vary_inputs:
        print("  inputs           : varied (recompilation stress test)")
    else:
        print(f"  resolution       : {args.width}x{args.height}")
        print(f"  inference steps  : {args.num_inference_steps}")
    print(f"  guidance scale   : {args.guidance_scale}")
    print(f"  enable FBC       : {args.enable_fbc}")
    print(f"  residual thresh  : {args.residual_threshold}")
    print(f"  taylorseer       : {args.taylorseer}")
    if args.taylorseer:
        print(
            f"  ts interval      : {args.taylorseer_cache_interval or 'model-default'}"
        )
        print(
            f"  ts warmup        : {args.taylorseer_warmup_steps or 'model-default'}"
        )
        print(
            f"  ts max order     : {args.taylorseer_max_order or 'model-default'}"
        )
    print(f"  warmup runs      : {args.num_warmups}")
    print()
    print("  Torch config:")
    print("    mode           : torch.compile (max-autotune)")
    print("    dtype          : BF16")
    print()
    print("  MAX config:")
    print("    dtype          : BF16")
    caching_parts: list[str] = []
    if args.enable_fbc:
        caching_parts.append(
            f"first block cache (threshold: {args.residual_threshold})"
        )
    print(f"    caching        : {', '.join(caching_parts) or 'none'}")
    print()

    timed_configs = _iter_configs(args, args.num_iterations)
    timed_prompts = _iter_prompts(args, args.num_iterations)

    if diffusers_result:
        _print_summary("Diffusers (PyTorch)", diffusers_result)
    if max_result:
        _print_summary("MAX", max_result)

    if args.vary_inputs or args.vary_prompts:
        if diffusers_result:
            _print_per_iteration(
                "Diffusers (PyTorch)",
                diffusers_result,
                timed_configs,
                timed_prompts,
            )
        if max_result:
            _print_per_iteration(
                "MAX", max_result, timed_configs, timed_prompts
            )

    if diffusers_result and max_result:
        d_mean = statistics.mean(diffusers_result.total_durations)
        m_mean = statistics.mean(max_result.total_durations)
        speedup = d_mean / m_mean if m_mean > 0 else float("inf")
        per_iter_speedups = [
            d / m if m > 0 else float("inf")
            for d, m in zip(
                diffusers_result.total_durations,
                max_result.total_durations,
                strict=True,
            )
        ]
        min_speedup = min(per_iter_speedups)
        max_speedup = max(per_iter_speedups)
        print()
        print("  Speedup (MAX vs Diffusers):")
        print(f"    avg: {speedup:.2f}x")
        print(f"    min: {min_speedup:.2f}x")
        print(f"    max: {max_speedup:.2f}x")

    print()
    return 0


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    raise SystemExit(main())
