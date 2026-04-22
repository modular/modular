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

"""Wan T2V performance comparison: diffusers (PyTorch) vs MAX.

Runs Wan text-to-video generation with both backends, performs warmup runs,
benchmarks with split preprocessing/execution timings, and prints a
side-by-side summary.  Supports Wan 2.1/2.2 T2V models (including MoE
variants) via the `--model` argument.  Also includes 1-frame (text-to-image)
configs in the iteration set.  Each iteration uses a different
`(height, width, num_frames, num_inference_steps)` tuple and cycles through
a fixed set of prompts with different sequence lengths to stress test
eager-mode recompilation and text-encoder performance.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import json
import logging
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from max.driver import DeviceSpec
from max.interfaces import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
    VideoProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    OpenResponsesRequestBody,
    OutputImageContent,
)
from max.pipelines import PIPELINE_REGISTRY, MAXModelConfig, PipelineConfig
from max.pipelines.architectures.wan.context import WanContext
from max.pipelines.architectures.wan.tokenizer import WanTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from PIL import Image

# Varying-input configurations for systematic memory/performance testing.
# Each tuple is `(height, width, num_frames, num_inference_steps)`.
# Ordered from largest to smallest estimated memory footprint so the
# biggest allocation lands first while the memory manager has maximum
# headroom, allowing more configurations to succeed before OOM.
#
# B200 memory budget (192 GB HBM3e, ~185 GB usable):
#   - DiT weights (14B, bf16) x 2 experts (MoE dual-load): ~56 GB
#   - Text encoder (UMT5):      ~9.4 GB
#   - VAE weights:               ~0.5 GB
#   - Framework/compile overhead: ~6 GB
#   - Workspace (shared per expert): scales with seq_len, x 2 experts
#   - Scheduler latent state:     scales with latent volume (modest)
#   - VAE decode peak:            scales with pixel volume
#
# With combined-blocks compilation (1 workspace per expert, not 40):
#   480p/49f   → ~74 GB   (seq_len ~20k)
#   480p/81f   → ~76 GB   (seq_len ~33k)
#   720p/49f   → ~77 GB   (seq_len ~47k)
#   720p/81f   → ~80 GB   (seq_len ~76k)
#
# 1-frame image generation configs (height, width, num_frames, steps).
# Run these first to establish a text-to-image baseline.
IMAGE_CONFIGS: list[tuple[int, int, int, int]] = [
    (1152, 2048, 1, 20),
    (1536, 2048, 1, 20),
    (2048, 2048, 1, 20),
]

VARIED_CONFIGS: list[tuple[int, int, int, int]] = [
    # --- Image configs (1 frame) — run first ---
    *IMAGE_CONFIGS,
    # --- Tier 2: Medium 720p (80-90 GB) ---
    (720, 1280, 81, 50),  # ~80 GB — standard 720p, 81 frames
    (720, 1280, 49, 30),  # ~77 GB — 720p short clip
    # --- Tier 1: Small / fast (well under 80 GB) ---
    (480, 832, 81, 50),  # ~76 GB — standard 480p
    (480, 832, 49, 30),  # ~74 GB — baseline sanity check
]

# Prompts following the Wan2.1 recommended formula:
#   Subject (description) + Scene (environment) + Motion (action/speed)
#   + Camera Language (shot/angle/movement) + Atmosphere + Style
# Each prompt is ~80-100 words with a single coherent scene, explicit
# motion description, and camera direction — the patterns that Wan
# responds to best.  Varying lengths also stress the text encoder.
VARIED_PROMPTS: list[str] = [
    # 1 — Short/simple: animal + nature + subtle motion + static shot
    (
        "A golden retriever with wet fur runs joyfully through shallow"
        " ocean waves on a sandy beach at sunset. Water splashes around"
        " its paws with each stride. The camera holds a low-angle"
        " tracking shot following the dog from the side. Warm golden"
        " hour lighting, cinematic color grading, photorealistic style."
    ),
    # 2 — Medium: human subject + urban scene + walking motion + tracking
    (
        "A young woman in a red raincoat walks slowly through a"
        " rain-soaked Tokyo street at night. Neon signs in Japanese"
        " reflect off the wet pavement, casting pink and blue light"
        " across her face. The camera follows her from behind in a"
        " smooth tracking shot. Raindrops are visible falling through"
        " the neon glow. Moody, atmospheric, cyberpunk aesthetic,"
        " shallow depth of field."
    ),
    # 3 — Nature + transformation + slow motion
    (
        "A single white lotus flower slowly blooms in a still dark pond,"
        " its petals unfurling one by one over the course of the clip."
        " Morning mist drifts across the water surface. The camera"
        " holds a close-up macro shot, gradually pulling back to reveal"
        " the surrounding lily pads. Soft diffused lighting, serene"
        " and meditative atmosphere, nature documentary style."
    ),
    # 4 — Aerial/drone + landscape + large motion
    (
        "An aerial drone shot slowly flies over a winding turquoise"
        " river cutting through a dense tropical rainforest. Flocks"
        " of white birds scatter from the treetops as the camera"
        " passes overhead. The river reflects the golden light of late"
        " afternoon. The camera moves forward steadily, tilting down"
        " to follow the river's curve. Lush green canopy, warm"
        " cinematic lighting, epic documentary style, 4K detail."
    ),
    # 5 — Sci-fi + interior + floating motion + camera drift
    (
        "An astronaut in a white spacesuit floats weightlessly inside a"
        " dimly lit space station module. Loose tools and droplets of"
        " water drift slowly around her. Sunlight streams through a"
        " circular window, casting a sharp beam across the cabin and"
        " illuminating dust particles in the air. The camera drifts"
        " gently forward toward the astronaut. Quiet, contemplative"
        " atmosphere, photorealistic, soft volumetric lighting."
    ),
    # 6 — Food/product + macro + subtle motion + static
    (
        "A steaming cup of black coffee sits on a rustic wooden table"
        " in a cozy morning kitchen. Wisps of steam rise and curl"
        " slowly from the dark liquid surface. Warm sunlight streams"
        " through a nearby window, casting long soft shadows across"
        " the table. The camera holds a static close-up, shallow"
        " depth of field with a bokeh background of kitchen shelves."
        " Warm, inviting atmosphere, photorealistic still life style."
    ),
    # 7 — Fantasy/stylized + dynamic motion + orbital camera
    (
        "Two anthropomorphic cats wearing colorful boxing gloves and"
        " protective headgear exchange rapid punches on a small"
        " spotlighted boxing ring. One cat dodges with a quick lean"
        " while the other throws a wide hook. The camera orbits slowly"
        " around the ring at eye level. Dramatic overhead stage"
        " lighting with visible light beams, energetic and playful"
        " atmosphere, Pixar-style 3D animation, vibrant saturated"
        " colors."
    ),
    # 8 — Longest: complex scene + multiple details + cinematic
    (
        "A sprawling medieval castle perched on a misty hilltop at the"
        " break of dawn. Thick fog rolls through the valley below,"
        " slowly revealing the stone walls and towers as the morning"
        " light intensifies. A flock of birds takes flight from the"
        " tallest tower. The camera begins with a wide establishing"
        " shot, then slowly pushes forward through the mist toward"
        " the castle gate. Cold blue morning light gradually warming"
        " to golden tones, epic and majestic atmosphere, cinematic"
        " aerial photography style, high detail."
    ),
]


@dataclass
class GPUMemorySnapshot:
    """A point-in-time snapshot of GPU memory usage."""

    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float

    @staticmethod
    def capture() -> GPUMemorySnapshot | None:
        """Capture current GPU memory state. Returns None if unavailable."""
        if not torch.cuda.is_available():
            return None
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            props = torch.cuda.get_device_properties(0)
            total = props.total_mem / (1024**3)
            free = total - reserved
            return GPUMemorySnapshot(
                allocated_gb=allocated,
                reserved_gb=reserved,
                free_gb=free,
                total_gb=total,
            )
        except Exception:
            return None

    @staticmethod
    def capture_nvidia_smi() -> GPUMemorySnapshot | None:
        """Capture GPU memory via nvidia-smi (works even without PyTorch
        holding the memory)."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            parts = out.strip().split(",")
            used_mb = float(parts[0].strip())
            free_mb = float(parts[1].strip())
            total_mb = float(parts[2].strip())
            return GPUMemorySnapshot(
                allocated_gb=used_mb / 1024,
                reserved_gb=used_mb / 1024,
                free_gb=free_mb / 1024,
                total_gb=total_mb / 1024,
            )
        except Exception:
            return None

    def __str__(self) -> str:
        return (
            f"alloc={self.allocated_gb:.1f}GB"
            f" reserved={self.reserved_gb:.1f}GB"
            f" free={self.free_gb:.1f}GB"
            f" total={self.total_gb:.1f}GB"
        )


def _compute_theoretical_memory(
    height: int,
    width: int,
    num_frames: int,
    *,
    dual_moe: bool = True,
) -> dict[str, float | str]:
    """Compute theoretical memory estimates for a given video config.

    Returns a dict of component names to estimated GB.

    With ``dual_moe=True`` (default), accounts for both high-noise and
    low-noise expert transformers being resident on GPU simultaneously,
    each with their own compiled model and shared workspace.
    """
    # VAE compression.
    vae_scale_s = 8
    vae_scale_t = 4
    latent_h = 2 * (height // (vae_scale_s * 2))
    latent_w = 2 * (width // (vae_scale_s * 2))
    adjusted = max(1, num_frames)
    if adjusted > 1:
        remainder = (adjusted - 1) % vae_scale_t
        if remainder != 0:
            adjusted += vae_scale_t - remainder
    latent_frames = (adjusted - 1) // vae_scale_t + 1

    # Patch embedding: patch_size = (1, 2, 2).
    seq_len = latent_frames * (latent_h // 2) * (latent_w // 2)

    # Latent tensor sizes.
    latent_elements = 1 * 16 * latent_frames * latent_h * latent_w
    latent_f32_gb = latent_elements * 4 / (1024**3)

    # DiT weights: 14B params x 2 bytes (bf16) ~ 28.1 GB per expert.
    num_experts = 2 if dual_moe else 1
    dit_weights_gb = 28.1 * num_experts

    # Workspace: with combined-blocks compilation, each expert has one
    # shared workspace.  Workspace ≈ peak single-block intermediates.
    # hidden_state [1, seq_len, 5120] bf16 + FFN [1, seq_len, 13824] bf16
    # + Q/K/V for self-attention (flash attn, ~2x hidden for Q+output).
    hidden_gb = seq_len * 5120 * 2 / (1024**3)
    ffn_gb = seq_len * 13824 * 2 / (1024**3)
    attn_gb = seq_len * 5120 * 2 * 2 / (1024**3)
    workspace_per_expert_gb = hidden_gb + ffn_gb + attn_gb
    workspace_gb = workspace_per_expert_gb * num_experts

    # Scheduler state: 5 copies of latent in f32.
    scheduler_gb = 5 * latent_f32_gb

    # VAE decode: output tensor [1, 3, frames, H, W] f32.
    vae_output_gb = 1 * 3 * num_frames * height * width * 4 / (1024**3)

    # RoPE: [seq_len, 128] f32 x 2 (cos + sin).
    rope_gb = 2 * seq_len * 128 * 4 / (1024**3)

    return {
        "dit_weights_bf16": dit_weights_gb,
        "text_encoder_bf16": 9.4,
        "vae_weights": 0.5,
        "workspace_peak": workspace_gb,
        "scheduler_latent_state": scheduler_gb,
        "vae_decode_output": vae_output_gb,
        "rope_embeddings": rope_gb,
        "framework_overhead": 6.0,
        "seq_len": float(seq_len),
        "latent_shape": f"[1,16,{latent_frames},{latent_h},{latent_w}]",
    }


@dataclass
class BenchmarkRequest:
    """All parameters for a single benchmark iteration."""

    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    prompt: str
    negative_prompt: str = ""

    @property
    def config_key(self) -> tuple[int, int, int, int]:
        return (
            self.height,
            self.width,
            self.num_frames,
            self.num_inference_steps,
        )


def _build_requests() -> list[BenchmarkRequest]:
    """Build one request per prompt, cycling through VARIED_CONFIGS."""
    return [
        BenchmarkRequest(
            height=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][0],
            width=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][1],
            num_frames=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][2],
            num_inference_steps=VARIED_CONFIGS[i % len(VARIED_CONFIGS)][3],
            prompt=VARIED_PROMPTS[i],
        )
        for i in range(len(VARIED_PROMPTS))
    ]


@dataclass
class TimingResult:
    """Collected timings for a single backend."""

    preprocess_durations: list[float] = field(default_factory=list)
    execute_durations: list[float] = field(default_factory=list)
    total_durations: list[float] = field(default_factory=list)
    peak_memory_gb: list[float] = field(default_factory=list)
    errors: list[tuple[int, str, str]] = field(default_factory=list)
    """(iteration_index, config_desc, error_message) for failed iterations."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Wan T2V performance: diffusers vs MAX."
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help=(
            "HuggingFace model ID to use (e.g. "
            "'Wan-AI/Wan2.1-T2V-14B-Diffusers', "
            "'Wan-AI/Wan2.2-T2V-A14B-Diffusers')."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=None,
        help=(
            "Secondary guidance scale for low-noise expert (MoE models). "
            "When set, the pipeline uses dual guidance scales."
        ),
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup passes over all requests (not timed) per backend.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help=(
            "Number of timed iterations to run (1 to number of prompts). "
            "Each iteration uses a different prompt/config pair. "
            "Defaults to the full prompt list."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, a random seed is generated.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Enable torch.compile on the diffusers pipeline components.",
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
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Override the number of frames for all requests. "
            "If not set, uses the per-config defaults."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override the height for all requests.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override the width for all requests.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the number of inference steps for all requests.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt for all requests.",
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        action="append",
        default=None,
        help="Path(s) to model weight files. Can be specified multiple times.",
    )
    parser.add_argument(
        "--quantization-encoding",
        type=str,
        default=None,
        choices=["float32", "bfloat16"],
        help="Weight encoding type.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip saving generated video frames to disk.",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Save output as MP4 video (requires imageio[ffmpeg]).",
    )
    parser.add_argument(
        "--model-override",
        type=str,
        action="append",
        default=None,
        help=(
            "Per-component overrides in 'component.field=value' format. "
            "Repeatable. Example: "
            "'transformer.quantization_encoding=bfloat16'."
        ),
    )
    parser.add_argument(
        "--single-prompt",
        type=str,
        default=None,
        help=(
            "Run a single prompt instead of the varied prompt set. "
            "Useful for quick testing."
        ),
    )
    return parser.parse_args(argv)


def _build_warmup_requests(
    requests: list[BenchmarkRequest],
    num_warmups: int,
) -> list[BenchmarkRequest]:
    """Return warmup requests: the full request list repeated *num_warmups* times."""
    return requests * num_warmups


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


def _save_frames(
    frames: np.ndarray,
    output_dir: str,
    prefix: str,
    max_saved: int = 8,
) -> list[str]:
    """Save a subset of video frames as PNG images.

    Args:
        frames: Video frames as numpy array, shape (T, H, W, C) or (T, C, H, W).
        output_dir: Directory to save frames in.
        prefix: Filename prefix for saved frames.
        max_saved: Maximum number of frames to save (evenly spaced).

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    if frames.ndim == 5:
        # (B, C, T, H, W) -> take first batch element -> (C, T, H, W)
        frames = frames[0]
    # Check last dim first: if it's 1/3, the data is already (T, H, W, C).
    # Only then fall back to interpreting shape[0] as the channel dim (which
    # would otherwise mis-trigger on (N=1, H, W, C) 1-frame outputs).
    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        pass  # already (T, H, W, C)
    elif frames.ndim == 4 and frames.shape[0] in (1, 3):
        # (C, T, H, W) -> (T, H, W, C)
        frames = np.transpose(frames, (1, 2, 3, 0))
    elif frames.ndim == 4:
        # (T, C, H, W) -> (T, H, W, C)
        frames = np.transpose(frames, (0, 2, 3, 1))

    num_frames = frames.shape[0]
    indices = np.linspace(
        0, num_frames - 1, min(max_saved, num_frames), dtype=int
    )
    saved = []
    for idx in indices:
        frame = frames[idx]
        # Normalize to [0, 255] if needed.
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        elif frame.dtype == np.float16:
            frame = np.clip(frame.astype(np.float32) * 255.0, 0, 255).astype(
                np.uint8
            )
        # Handle single-channel by squeezing.
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(-1)
        img = Image.fromarray(frame)
        fname = f"{prefix}_frame{idx:04d}.png"
        path = os.path.join(output_dir, fname)
        img.save(path)
        saved.append(fname)
    return saved


def _save_video(
    frames: np.ndarray,
    output_dir: str,
    prefix: str,
    fps: int = 16,
) -> str:
    """Save video frames as an MP4 file using PyAV.

    Args:
        frames: Video frames, shape (T, H, W, C) or (T, C, H, W) uint8/float.
        output_dir: Directory to save the video in.
        prefix: Filename prefix.
        fps: Frames per second for the output video.

    Returns:
        The saved filename.
    """
    import av

    os.makedirs(output_dir, exist_ok=True)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
        pass  # already (T, H, W, C)
    elif frames.ndim == 4 and frames.shape[0] in (1, 3):
        frames = np.transpose(frames, (1, 2, 3, 0))
    elif frames.ndim == 4:
        frames = np.transpose(frames, (0, 2, 3, 1))

    # Normalize to uint8 if needed.
    if frames.dtype in (np.float32, np.float64):
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    elif frames.dtype == np.float16:
        frames = np.clip(frames.astype(np.float32) * 255.0, 0, 255).astype(
            np.uint8
        )

    fname = f"{prefix}.mp4"
    path = os.path.join(output_dir, fname)

    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    for frame_data in frames:
        video_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return fname


def _get_vae_dtype(args: argparse.Namespace) -> torch.dtype:
    """Determine VAE dtype from --model-override flags.

    Defaults to bf16 for apples-to-apples comparison with MAX.  If the user
    passes ``--model-override vae.quantization_encoding=float32``, the VAE
    falls back to fp32.
    """
    if args.model_override:
        for override in args.model_override:
            if override.startswith("vae.quantization_encoding="):
                encoding = override.split("=", 1)[1].lower()
                if encoding in ("float32", "fp32"):
                    return torch.float32
    return torch.bfloat16


def _load_diffusers_pipeline(model_id: str, args: argparse.Namespace) -> Any:
    """Load Wan pipeline via diffusers with optimized attention."""
    import diffusers
    from diffusers import AutoencoderKLWan

    vae_dtype = _get_vae_dtype(args)

    # Load VAE separately so we control its dtype, ensuring an
    # apples-to-apples comparison with the MAX backend.
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype
    )
    pipe = diffusers.WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda")

    if args.torch_compile:
        # Compile key components for performance.
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder = torch.compile(
                pipe.text_encoder,
                mode="max-autotune",
                fullgraph=False,
            )
        # Use max-autotune-no-cudagraphs instead of max-autotune for the
        # transformer.  max-autotune enables CUDA graphs whose output
        # buffers get reused across replays, but the diffusers denoising
        # loop reads the previous output after the next transformer call,
        # hitting a stale-tensor error.  This mode keeps Triton kernel
        # autotuning while disabling CUDA graph capture.
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae = torch.compile(
                pipe.vae, mode="max-autotune", fullgraph=True
            )
    return pipe


def _video_to_numpy(output: Any) -> np.ndarray:
    """Extract numpy video frames from diffusers output.

    Diffusers WanPipeline returns output.frames as a list of lists of
    PIL Images, or sometimes as a tensor. Normalize to (T, H, W, C) float32.
    """
    frames = output.frames
    if isinstance(frames, torch.Tensor):
        return frames.cpu().float().numpy()
    if isinstance(frames, np.ndarray):
        return frames.astype(np.float32)
    # frames is list[list[PIL.Image]]
    if isinstance(frames, list) and isinstance(frames[0], list):
        pil_frames = frames[0]  # First batch element.
        return np.stack(
            [np.array(f).astype(np.float32) / 255.0 for f in pil_frames],
            axis=0,
        )
    if isinstance(frames, list) and isinstance(frames[0], Image.Image):
        return np.stack(
            [np.array(f).astype(np.float32) / 255.0 for f in frames],
            axis=0,
        )
    raise TypeError(f"Unexpected diffusers output type: {type(frames)}")


def run_diffusers(
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark Wan T2V through diffusers. Returns split timings."""
    print("\n=== Diffusers (PyTorch) backend (text-to-video) ===", flush=True)
    print(f"Loading pipeline ({args.model})...", flush=True)
    pipe = _load_diffusers_pipeline(args.model, args)

    warmup_requests = _build_warmup_requests(requests, args.num_warmups)
    for i, req in enumerate(warmup_requests):
        print(
            f"  warmup {i + 1}/{len(warmup_requests)}"
            f" ({req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        torch.cuda.synchronize()

    result = TimingResult()
    n = len(requests)
    for i, req in enumerate(requests):
        config_desc = (
            f"{req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps"
        )
        theory = _compute_theoretical_memory(
            req.height, req.width, req.num_frames
        )
        print(
            f"  iter {i + 1}/{n}"
            f" ({config_desc},"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        print(
            f"    theoretical: seq_len={int(theory['seq_len']):,}"
            f"  latent={theory['latent_shape']}"
            f"  est_total="
            f"{sum(v for k, v in theory.items() if isinstance(v, float) and k not in ('seq_len',)):.1f}GB",
            flush=True,
        )

        mem_before = GPUMemorySnapshot.capture()
        if mem_before:
            print(f"    gpu before : {mem_before}", flush=True)

        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        try:
            # Full pipeline timing (diffusers doesn't easily split preprocess/execute).
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            torch.cuda.synchronize()
            t_total = time.perf_counter() - t0

            peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
            mem_after = GPUMemorySnapshot.capture()
            print(f"    gpu after  : {mem_after}")
            print(f"    peak alloc : {peak_gb:.1f} GB")

            result.preprocess_durations.append(0.0)
            result.execute_durations.append(t_total)
            result.total_durations.append(t_total)
            result.peak_memory_gb.append(peak_gb)

            if output_dir is not None:
                try:
                    frames = _video_to_numpy(output)
                    prefix = (
                        f"iter{i}_torch_{req.width}x{req.height}"
                        f"_{req.num_frames}f_{req.num_inference_steps}s"
                    )
                    if args.save_mp4 and req.num_frames > 1:
                        fname = _save_video(frames, output_dir, prefix)
                        print(f"    saved video: {fname}")
                    else:
                        saved = _save_frames(frames, output_dir, prefix)
                        print(
                            f"    saved {len(saved)} frames:"
                            f" {saved[0]} ... {saved[-1]}"
                        )
                except Exception as e:
                    print(f"    WARNING: Failed to save output: {e}")
        except Exception as e:
            error_msg = str(e)
            mem_at_error = GPUMemorySnapshot.capture()
            print(f"    ERROR: {error_msg}")
            if mem_at_error:
                print(f"    gpu at error: {mem_at_error}")
            print(f"    traceback:\n{traceback.format_exc()}")
            result.errors.append((i, config_desc, error_msg))
            # Clear GPU memory and continue to next iteration.
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # Free GPU memory before MAX runs.
    del pipe
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    return result


def _load_max_pipeline(
    args: argparse.Namespace,
) -> tuple[PixelGenerationPipeline[WanContext], WanTokenizer, PipelineConfig]:
    """Load Wan pipeline via MAX."""
    model_id = args.model

    manifest = ModelManifest.from_model_path(
        model_id,
        device_specs=[DeviceSpec.accelerator()],
    )
    if args.weight_path:
        manifest = manifest.with_override(
            "transformer",
            weight_path=[Path(p) for p in args.weight_path],
        )
    if args.quantization_encoding:
        manifest = manifest.with_override(
            "transformer",
            quantization_encoding=args.quantization_encoding,
        )

    # Apply flexible per-component overrides from --model-override.
    if args.model_override:
        from pydantic import TypeAdapter

        for override in args.model_override:
            dot_pos = override.find(".")
            if dot_pos < 1:
                raise ValueError(
                    f"Invalid --model-override format: {override!r}. "
                    f"Expected 'component.field=value'."
                )
            eq_pos = override.find("=", dot_pos)
            if eq_pos < dot_pos + 2:
                raise ValueError(
                    f"Invalid --model-override format: {override!r}. "
                    f"Expected 'component.field=value'."
                )
            component = override[:dot_pos]
            field_name = override[dot_pos + 1 : eq_pos]
            raw_value = override[eq_pos + 1 :]

            if field_name not in MAXModelConfig.model_fields:
                raise ValueError(
                    f"Unknown MAXModelConfig field: {field_name!r}. "
                    f"Valid fields: "
                    f"{sorted(MAXModelConfig.model_fields.keys())}"
                )
            if component not in manifest:
                raise ValueError(
                    f"Component {component!r} not found in manifest. "
                    f"Available: {list(manifest.keys())}"
                )

            field_info = MAXModelConfig.model_fields[field_name]
            adapter: TypeAdapter[Any] = TypeAdapter(field_info.annotation)
            try:
                parsed = json.loads(raw_value)
            except (json.JSONDecodeError, ValueError):
                parsed = raw_value
            value = adapter.validate_python(parsed)
            manifest = manifest.with_override(component, **{field_name: value})

    config = PipelineConfig(
        models=manifest,
        runtime=PipelineRuntimeConfig(),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.models.main_architecture_name,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, (
        f"No architecture found in MAX registry for {model_id}."
    )

    tokenizer = WanTokenizer(
        model_path=model_id,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=512,
    )

    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[WanContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )
    return pipeline, tokenizer, config


async def _build_max_inputs(
    args: argparse.Namespace,
    tokenizer: WanTokenizer,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    negative_prompt: str = "",
) -> tuple[Any, Any]:
    """Build MAX pipeline inputs for a single video generation."""
    body = OpenResponsesRequestBody(
        model=args.model,
        input=prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                height=height,
                width=width,
                steps=steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=negative_prompt or None,
            ),
            video=VideoProviderOptions(
                negative_prompt=negative_prompt or None,
                height=height,
                width=width,
                num_frames=num_frames,
                steps=steps,
                guidance_scale_2=args.guidance_scale_2,
            ),
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[WanContext](
        batch={context.request_id: context}
    )
    return inputs, context


def run_max(
    args: argparse.Namespace,
    requests: list[BenchmarkRequest],
    output_dir: str | None = None,
) -> TimingResult:
    """Benchmark Wan T2V through MAX with split preprocess/execute timings."""
    print("\n=== MAX backend (text-to-video) ===", flush=True)
    print(f"Loading pipeline ({args.model})...", flush=True)
    pipeline, tokenizer, _config = _load_max_pipeline(args)

    warmup_requests = _build_warmup_requests(requests, args.num_warmups)
    for i, req in enumerate(warmup_requests):
        print(
            f"  warmup {i + 1}/{len(warmup_requests)}"
            f" ({req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps,"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        inputs, _ = asyncio.run(
            _build_max_inputs(
                args,
                tokenizer,
                req.prompt,
                req.height,
                req.width,
                req.num_frames,
                req.num_inference_steps,
                req.negative_prompt,
            )
        )
        pipeline.execute(inputs)

    result = TimingResult()
    n = len(requests)
    for i, req in enumerate(requests):
        config_desc = (
            f"{req.width}x{req.height}, {req.num_frames}f,"
            f" {req.num_inference_steps} steps"
        )
        theory = _compute_theoretical_memory(
            req.height, req.width, req.num_frames
        )
        print(
            f"  iter {i + 1}/{n}"
            f" ({config_desc},"
            f" prompt={_truncate_prompt(req.prompt)!r})",
            flush=True,
        )
        print(
            f"    theoretical: seq_len={int(theory['seq_len']):,}"
            f"  latent={theory['latent_shape']}"
            f"  est_total="
            f"{sum(v for k, v in theory.items() if isinstance(v, float) and k not in ('seq_len',)):.1f}GB",
            flush=True,
        )

        mem_before = GPUMemorySnapshot.capture_nvidia_smi()
        if mem_before:
            print(f"    gpu before : {mem_before}", flush=True)

        try:
            # Preprocessing: tokenization + context + input building
            t0 = time.perf_counter()
            inputs, context = asyncio.run(
                _build_max_inputs(
                    args,
                    tokenizer,
                    req.prompt,
                    req.height,
                    req.width,
                    req.num_frames,
                    req.num_inference_steps,
                    req.negative_prompt,
                )
            )
            t_preprocess = time.perf_counter() - t0

            # Model execution
            t1 = time.perf_counter()
            outputs = pipeline.execute(inputs)
            t_execute = time.perf_counter() - t1

            mem_after = GPUMemorySnapshot.capture_nvidia_smi()
            if mem_after:
                print(f"    gpu after  : {mem_after}")
                result.peak_memory_gb.append(mem_after.allocated_gb)
            else:
                result.peak_memory_gb.append(0.0)

            result.preprocess_durations.append(t_preprocess)
            result.execute_durations.append(t_execute)
            result.total_durations.append(t_preprocess + t_execute)

            # Save output frames (outside timing)
            if output_dir is not None:
                output = outputs[context.request_id]
                output = asyncio.run(tokenizer.postprocess(output))
                if output.output:
                    frames = []
                    for img_content in output.output:
                        if (
                            isinstance(img_content, OutputImageContent)
                            and img_content.image_data
                        ):
                            image_bytes = base64.b64decode(
                                img_content.image_data
                            )
                            img = Image.open(io.BytesIO(image_bytes))
                            frames.append(np.array(img))
                    if frames:
                        prefix = (
                            f"iter{i}_max_{req.width}x{req.height}"
                            f"_{req.num_frames}f_{req.num_inference_steps}s"
                        )
                        stacked = np.stack(frames)
                        if args.save_mp4:
                            fname = _save_video(stacked, output_dir, prefix)
                            print(f"    saved video: {fname}")
                        else:
                            saved = _save_frames(stacked, output_dir, prefix)
                            print(
                                f"    saved {len(saved)} frames:"
                                f" {saved[0]} ... {saved[-1]}"
                            )
        except Exception as e:
            error_msg = str(e)
            mem_at_error = GPUMemorySnapshot.capture_nvidia_smi()
            print(f"    ERROR: {error_msg}")
            if mem_at_error:
                print(f"    gpu at error: {mem_at_error}")
            print(f"    traceback:\n{traceback.format_exc()}")
            result.errors.append((i, config_desc, error_msg))
            gc.collect()
            continue

    return result


def _print_summary(label: str, result: TimingResult) -> None:
    """Print timing summary for one backend."""
    n = len(result.total_durations)
    print(f"  {label}:")
    print(f"    iterations : {n} succeeded, {len(result.errors)} failed")

    if n == 0:
        if result.errors:
            print("    All iterations failed:")
            for idx, desc, msg in result.errors:
                print(f"      iter {idx}: [{desc}] {msg}")
        return

    if result.preprocess_durations and any(
        d > 0 for d in result.preprocess_durations
    ):
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

    if result.peak_memory_gb:
        peak = max(result.peak_memory_gb)
        print(f"    peak GPU   : {peak:8.1f} GB")

    if result.errors:
        print(f"    FAILED iterations ({len(result.errors)}):")
        for idx, desc, msg in result.errors:
            print(f"      iter {idx}: [{desc}] {msg}")


def _print_per_iteration(
    label: str,
    result: TimingResult,
    requests: list[BenchmarkRequest],
) -> None:
    """Print per-iteration breakdown with config/prompt details."""
    print(f"\n  {label} per-iteration breakdown:")

    # Build a set of failed iteration indices for quick lookup.
    failed_indices = {idx for idx, _, _ in result.errors}

    header = (
        f"    {'Config':>22s}"
        f"  {'Prompt':>42s}"
        f"  {'Preprocess':>10s}  {'Execute':>10s}  {'Total':>10s}"
        f"  {'Peak GPU':>8s}  {'Status':>8s}"
    )
    print(header)

    success_idx = 0
    for i, req in enumerate(requests):
        config_str = (
            f"{req.width:4d}x{req.height:<4d} {req.num_frames:3d}f"
            f" {req.num_inference_steps:2d}s"
        )
        if i in failed_indices:
            error_msg = next(msg for idx, _, msg in result.errors if idx == i)
            short_err = (
                error_msg[:30] + "…" if len(error_msg) > 30 else error_msg
            )
            line = (
                f"    {config_str:>22s}"
                f"  {_truncate_prompt(req.prompt, 42):>42s}"
                f"  {'---':>10s}  {'---':>10s}  {'---':>10s}"
                f"  {'---':>8s}  FAILED"
            )
            print(line)
            print(f"      error: {short_err}")
        else:
            pp = (
                result.preprocess_durations[success_idx]
                if result.preprocess_durations
                else 0.0
            )
            ex = result.execute_durations[success_idx]
            tot = result.total_durations[success_idx]
            mem = (
                f"{result.peak_memory_gb[success_idx]:.1f}GB"
                if result.peak_memory_gb
                else "N/A"
            )
            line = (
                f"    {config_str:>22s}"
                f"  {_truncate_prompt(req.prompt, 42):>42s}"
                f"  {pp:10.2f}  {ex:10.2f}  {tot:10.2f}"
                f"  {mem:>8s}  OK"
            )
            print(line)
            success_idx += 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    args = parse_args(argv)

    # Resolve seed: generate a random one if not explicitly provided.
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    print(f"  seed             : {args.seed}")
    print("  (reproduce with: --seed", args.seed, ")")

    # Build the benchmark request list.
    if args.single_prompt:
        cfg = VARIED_CONFIGS[0]
        requests = [
            BenchmarkRequest(
                height=args.height or cfg[0],
                width=args.width or cfg[1],
                num_frames=args.num_frames or cfg[2],
                num_inference_steps=args.num_inference_steps or cfg[3],
                prompt=args.single_prompt,
                negative_prompt=args.negative_prompt,
            )
        ]
    else:
        requests = _build_requests()
        # Apply per-request overrides if specified.
        if (
            args.height
            or args.width
            or args.num_frames
            or args.num_inference_steps
        ):
            requests = [
                BenchmarkRequest(
                    height=args.height or req.height,
                    width=args.width or req.width,
                    num_frames=args.num_frames or req.num_frames,
                    num_inference_steps=args.num_inference_steps
                    or req.num_inference_steps,
                    prompt=req.prompt,
                    negative_prompt=args.negative_prompt,
                )
                for req in requests
            ]

    # Limit the number of timed iterations if requested.
    if args.num_iterations is not None:
        if args.num_iterations < 1 or args.num_iterations > len(requests):
            raise ValueError(
                f"--num-iterations must be between 1 and {len(requests)}"
                f" (number of prompts), got {args.num_iterations}"
            )
        requests = requests[: args.num_iterations]

    # Create output directory for saved frames (unless --no-output).
    output_dir: str | None = None
    if not args.no_output:
        output_dir = os.path.abspath("wan_comparison")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        print(f"  Saving frames to: {output_dir}/")

    # Write a prompts.txt manifest.
    if output_dir is not None:
        prompts_path = os.path.join(output_dir, "prompts.txt")
        with open(prompts_path, "w") as f:
            f.write(f"seed: {args.seed}\n")
            f.write(f"reproduce with: --seed {args.seed}\n\n")
            for i, req in enumerate(requests):
                f.write(
                    f"iter {i}: {req.width}x{req.height},"
                    f" {req.num_frames} frames,"
                    f" {req.num_inference_steps} steps\n"
                )
                f.write(f"  prompt: {req.prompt}\n\n")
        print(f"  Saved prompt manifest: {prompts_path}")

    diffusers_result: TimingResult | None = None
    max_result: TimingResult | None = None

    if not args.skip_diffusers:
        try:
            diffusers_result = run_diffusers(args, requests, output_dir)
        except Exception as e:
            print(f"ERROR running diffusers: {e}", file=sys.stderr)
            traceback.print_exc()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if not args.skip_max:
        try:
            max_result = run_max(args, requests, output_dir)
        except Exception as e:
            print(f"ERROR running MAX: {e}", file=sys.stderr)
            traceback.print_exc()

    # Summary header
    print("\n" + "=" * 60)
    model_name = (
        args.model.rsplit("/", 1)[-1] if "/" in args.model else args.model
    )
    print(
        f"{model_name} Text-to-Video Performance Comparison"
        f" — {datetime.now():%Y-%m-%d}"
    )
    print("=" * 60)
    _print_gpu_info()
    print(f"  model            : {args.model}")
    print(f"  seed             : {args.seed}")
    print("  mode             : Text-to-Video")
    print(f"  guidance scale   : {args.guidance_scale}")
    if args.guidance_scale_2 is not None:
        print(f"  guidance scale 2 : {args.guidance_scale_2}")
    print(f"  warmup runs      : {args.num_warmups}")
    print(f"  timed iterations : {len(requests)}")
    print()
    print("  Torch config:")
    compile_mode = (
        "torch.compile (max-autotune)"
        if args.torch_compile
        else "eager (no compile)"
    )
    print(f"    mode           : {compile_mode}")
    print("    dtype          : BF16")
    print()
    print("  MAX config:")
    max_dtype = (
        args.quantization_encoding.upper()
        if args.quantization_encoding
        else "BF16"
    )
    print(f"    dtype          : {max_dtype}")
    print()

    # Print theoretical memory estimates for all configs.
    print("  Theoretical memory estimates per config:")
    print(
        f"    {'Config':>24s}  {'Seq Len':>10s}  {'Latent Shape':>24s}"
        f"  {'Workspace':>9s}  {'Sched St':>9s}  {'VAE Dec':>8s}"
        f"  {'Est Total':>10s}"
    )
    for req in requests:
        t = _compute_theoretical_memory(req.height, req.width, req.num_frames)
        total_est = sum(
            v for k, v in t.items() if isinstance(v, float) and k != "seq_len"
        )
        cfg_label = (
            f"{req.width}x{req.height} {req.num_frames}f"
            f" {req.num_inference_steps}s"
        )
        print(
            f"    {cfg_label:>24s}"
            f"  {int(t['seq_len']):>10,}"
            f"  {t['latent_shape']:>24s}"
            f"  {t['workspace_peak']:>8.1f}G"
            f"  {t['scheduler_latent_state']:>8.2f}G"
            f"  {t['vae_decode_output']:>7.02f}G"
            f"  {total_est:>9.1f}G"
        )
    print()

    if diffusers_result:
        _print_summary("Diffusers (PyTorch)", diffusers_result)
    if max_result:
        _print_summary("MAX", max_result)

    if diffusers_result:
        _print_per_iteration(
            "Diffusers (PyTorch)",
            diffusers_result,
            requests,
        )
    if max_result:
        _print_per_iteration("MAX", max_result, requests)

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
