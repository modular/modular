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

"""Simple offline video generation example using diffusion models.

Usage:
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 \
    ./bazelw run //max/examples/diffusion:simple_offline_video_generation -- \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --prompt "A cat playing piano" \
        --output output.mp4

Note: MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 is required
for 720p+ resolutions with symbolic seq_len block graphs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import time

# Suppress noisy HTTP/download logs
import warnings
from typing import Any, cast

for _logger_name in ("httpx", "huggingface_hub", "hf_xet", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import numpy as np
import PIL.Image
from max.driver import DeviceSpec
from max.examples.diffusion.profiler import profile_execute
from max.interfaces import (
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.interfaces.generation import GenerationOutput
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
from max.pipelines.core import PixelContext
from max.pipelines.lib import (
    PixelGenerationTokenizer,
    load_video_frames,
    save_video,
)
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)

logging.basicConfig(
    level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
)

WAN_TEMPORAL_FRAME_STRIDE = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos with a diffusion model.",
    )
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for video generation."
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, distorted, deformed, ugly, bad, poor, worst quality",
        help="Negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video height in pixels. Auto-computed from input image if omitted.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video width in pixels. Auto-computed from input image if omitted.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=(
            "Number of video frames to generate. "
            "Defaults to 77 for WanAnimatePipeline (one segment) "
            "or 81 for Wan T2V/I2V."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=3.0,
        help="Secondary guidance scale for low-noise expert (MoE models).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video filename.",
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="Frames per second."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length of tokenizer.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of iterations to run (for benchmarking).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Run a small warmup (480x832, 17 frames, 2 steps) before timed runs.",
    )
    parser.add_argument(
        "--lora-repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID for LoRA weights.",
    )
    parser.add_argument(
        "--lora-subfolder",
        type=str,
        default=None,
        help="Subfolder in the LoRA repo containing safetensors files.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA strength multiplier for primary transformer.",
    )
    parser.add_argument(
        "--lora-scale-2",
        type=float,
        default=None,
        help="LoRA strength for transformer_2. Defaults to --lora-scale.",
    )
    parser.add_argument(
        "--lora-weight-name",
        type=str,
        default=None,
        help="Single LoRA safetensors filename (for non-MoE models).",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Path or URL to input image for I2V (image-to-video) generation.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=None,
        help="Multiple WxHxF specs (e.g. 1280x720x81 720x1280x81). "
        "Runs each resolution sequentially in the same process for "
        "recompilation testing. Overrides --width/--height/--num-frames.",
    )
    parser.add_argument(
        "--initial-noise",
        type=str,
        default=None,
        help="Path to .npy file with pre-generated initial noise (for parity testing).",
    )
    parser.add_argument(
        "--animate-mode",
        choices=["animate", "replace"],
        default="animate",
        help="Wan-Animate mode: 'animate' (motion transfer) or 'replace' (character replacement).",
    )
    parser.add_argument(
        "--pose-video",
        type=str,
        default=None,
        help="Path to preprocessed pose video for Wan-Animate.",
    )
    parser.add_argument(
        "--face-video",
        type=str,
        default=None,
        help="Path to preprocessed face video for Wan-Animate.",
    )
    parser.add_argument(
        "--background-video",
        type=str,
        default=None,
        help="Path to background video for Wan-Animate replace mode.",
    )
    parser.add_argument(
        "--mask-video",
        type=str,
        default=None,
        help="Path to mask video for Wan-Animate replace mode.",
    )
    parser.add_argument(
        "--segment-frame-length",
        type=int,
        default=77,
        help="Number of frames per segment for Wan-Animate.",
    )
    parser.add_argument(
        "--prev-segment-conditioning-frames",
        type=int,
        default=1,
        help="Number of overlap frames between segments for Wan-Animate.",
    )
    parser.add_argument(
        "--quantization-encoding",
        type=str,
        default=None,
        help="Override quantization encoding (e.g. 'bfloat16').",
    )
    parser.add_argument(
        "--secondary-max-length",
        type=int,
        default=None,
        help="Maximum length of secondary tokenizer.",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup runs before profiling or timed execution.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Profile timings of the pipeline.",
    )

    args = parser.parse_args(argv)
    assert args.prompt, "Prompt must be a non-empty string."

    if args.pose_video or args.face_video:
        assert args.pose_video and args.face_video, (
            "--pose-video and --face-video must be provided together."
        )
    if args.animate_mode == "replace":
        assert args.background_video and args.mask_video, (
            "--background-video and --mask-video are required in replace mode."
        )
    else:
        assert args.background_video is None and args.mask_video is None, (
            "--background-video and --mask-video are only supported in replace mode."
        )

    # --resolutions overrides --height/--width/--num-frames;
    # skip auto-compute and validation here (handled in generate_video).
    if not args.resolutions:
        if args.height is None or args.width is None:
            if args.input_image:
                from PIL import Image

                if args.input_image.startswith(("http://", "https://")):
                    import io
                    import urllib.request

                    with urllib.request.urlopen(args.input_image) as _resp:
                        img = Image.open(io.BytesIO(_resp.read()))
                else:
                    img = Image.open(args.input_image)
                aspect_ratio = img.height / img.width
                max_area = 720 * 1280
                mod_value = 16
                h = (
                    round(np.sqrt(max_area * aspect_ratio))
                    // mod_value
                    * mod_value
                )
                w = (
                    round(np.sqrt(max_area / aspect_ratio))
                    // mod_value
                    * mod_value
                )
                if args.height is None:
                    args.height = h
                if args.width is None:
                    args.width = w
                print(
                    f"Auto-computed resolution: {args.width}x{args.height}"
                    f" (from {img.size[0]}x{img.size[1]})"
                )
            else:
                if args.height is None:
                    args.height = 720
                if args.width is None:
                    args.width = 1280

    assert args.height > 0, "Height must be positive."
    assert args.width > 0, "Width must be positive."
    if args.num_frames is not None:
        assert args.num_frames > 0, "num-frames must be positive."
    assert args.num_inference_steps > 0, "num-inference-steps must be positive."
    assert args.guidance_scale > 0.0, "guidance-scale must be positive."
    return args


def _normalize_wan_num_frames(num_frames: int, *, phase: str) -> int:
    """Round Wan frame counts up to the nearest valid 1 + 4k size."""
    if num_frames <= 1:
        return 1

    remainder = (num_frames - 1) % WAN_TEMPORAL_FRAME_STRIDE
    if remainder == 0:
        return num_frames

    adjusted_num_frames = num_frames + WAN_TEMPORAL_FRAME_STRIDE - remainder
    print(
        "WanPipeline adjusted "
        f"{phase} num_frames from {num_frames} to {adjusted_num_frames}; "
        "Wan VAE temporal decode is stable on frame counts of the form 1 + 4k."
    )
    return adjusted_num_frames


def _video_frames_from_output(output: GenerationOutput) -> list[np.ndarray]:
    """Decode per-frame OutputImageContent payloads into RGB uint8 frames."""
    frames: list[np.ndarray] = []
    for item in output.output:
        if not isinstance(item, OutputImageContent):
            raise TypeError(
                f"Expected OutputImageContent, got {type(item).__name__}."
            )
        if item.image_data is None:
            raise ValueError("Expected inline image_data for video frame.")
        frame_bytes = base64.b64decode(item.image_data)
        with PIL.Image.open(io.BytesIO(frame_bytes)) as image:
            frames.append(np.array(image.convert("RGB"), dtype=np.uint8))
    return frames


def _load_videos(
    args: argparse.Namespace,
) -> tuple[
    list[PIL.Image.Image],
    list[PIL.Image.Image],
    list[PIL.Image.Image] | None,
    list[PIL.Image.Image] | None,
]:
    """Load Wan-Animate media as frame lists trimmed to effective_num_frames."""
    pose_video = load_video_frames(args.pose_video)
    effective_num_frames = min(len(pose_video), args.num_frames)
    pose_video = pose_video[:effective_num_frames]
    face_video = load_video_frames(args.face_video)[:effective_num_frames]
    background_video = (
        load_video_frames(args.background_video)[:effective_num_frames]
        if args.background_video
        else None
    )
    mask_video = (
        load_video_frames(args.mask_video)[:effective_num_frames]
        if args.mask_video
        else None
    )

    return pose_video, face_video, background_video, mask_video


def _build_request_body(
    args: argparse.Namespace,
    prompt: str | None = None,
    *,
    negative_prompt: str | None = None,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    guidance_scale_2: float | None = None,
) -> OpenResponsesRequestBody:
    prompt = args.prompt if prompt is None else prompt
    negative_prompt = (
        args.negative_prompt if negative_prompt is None else negative_prompt
    )
    height = args.height if height is None else height
    width = args.width if width is None else width
    num_frames = args.num_frames if num_frames is None else num_frames
    num_inference_steps = (
        args.num_inference_steps
        if num_inference_steps is None
        else num_inference_steps
    )
    guidance_scale = (
        args.guidance_scale if guidance_scale is None else guidance_scale
    )
    guidance_scale_2 = (
        getattr(args, "guidance_scale_2", None)
        if guidance_scale_2 is None
        else guidance_scale_2
    )
    return OpenResponsesRequestBody(
        model=args.model,
        input=prompt,
        seed=args.seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
            ),
            video=VideoProviderOptions(
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                steps=num_inference_steps,
                num_frames=num_frames,
                frames_per_second=getattr(args, "fps", 16),
                guidance_scale_2=guidance_scale_2,
            ),
        ),
    )


def _load_pipeline(
    args: argparse.Namespace,
) -> tuple[
    PixelGenerationTokenizer, PixelGenerationPipeline[PixelContext], Any
]:
    """Load tokenizer and pipeline from args."""
    print(f"Loading model: {args.model}")

    model_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "device_specs": [DeviceSpec.accelerator()],
    }
    if getattr(args, "quantization_encoding", None):
        model_kwargs["quantization_encoding"] = args.quantization_encoding
    config = PipelineConfig(
        model=MAXModelConfig(**model_kwargs),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, "No matching diffusion architecture found."

    # TI2V auto-routing: if resolved as WanPipeline but --input-image provided,
    # switch to WanI2VPipeline for image conditioning support.
    if arch.name == "WanPipeline" and getattr(args, "input_image", None):
        i2v_arch = PIPELINE_REGISTRY.architectures.get(
            "WanImageToVideoPipeline"
        )
        if i2v_arch is not None:
            print(
                "I2V auto-routing: WanPipeline -> WanI2VPipeline"
                " (--input-image provided)"
            )
            arch = i2v_arch

    # Wan-Animate defaults to 30 fps (matching diffusers/official config)
    if arch.name == "WanAnimatePipeline" and args.fps == 16:
        args.fps = 30
        print("Wan-Animate: auto-set fps=30 (use --fps to override)")

    # Inject LoRA config into diffusers_config if CLI args provided
    diffusers_config = config.model.diffusers_config
    if getattr(args, "lora_repo_id", None) and diffusers_config is not None:
        lora_cfg: dict[str, Any] = {
            "repo_id": args.lora_repo_id,
            "subfolder": getattr(args, "lora_subfolder", "") or "",
            "scale": getattr(args, "lora_scale", 1.0),
            "scale_2": (
                args.lora_scale_2
                if getattr(args, "lora_scale_2", None) is not None
                else getattr(args, "lora_scale", 1.0)
            ),
        }
        if getattr(args, "lora_weight_name", None):
            lora_cfg["filenames"] = [args.lora_weight_name]
        diffusers_config["lora"] = lora_cfg

    # Tokenizer setup
    max_length = args.max_length
    secondary_max_length = args.secondary_max_length
    has_tokenizer_2 = False
    if (
        max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and components_config.get("tokenizer", None) is not None
    ):
        max_length = components_config["tokenizer"]["config_dict"].get(
            "model_max_length", None
        )
        if arch.name in (
            "WanPipeline",
            "WanImageToVideoPipeline",
            "WanAnimatePipeline",
        ):
            max_length = 512
        print(f"Using max length: {max_length} for tokenizer")

    if (
        diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and components_config.get("tokenizer_2", None) is not None
    ):
        has_tokenizer_2 = True
        if secondary_max_length is None:
            secondary_max_length = components_config["tokenizer_2"][
                "config_dict"
            ].get("model_max_length", None)
            print(
                "Using secondary max length: "
                f"{secondary_max_length} for tokenizer_2"
            )

    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=max_length,
        subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
        secondary_max_length=secondary_max_length if has_tokenizer_2 else None,
    )

    # Pipeline setup.
    assert issubclass(arch.pipeline_model, DiffusionPipeline), (
        f"Architecture does not implement DiffusionPipeline: "
        f"{arch.pipeline_model}"
    )
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    print("Initialization complete.")
    return tokenizer, pipeline, arch


def _parse_resolution(spec: str) -> tuple[int, int, int]:
    """Parse 'WxHxF' or 'WxH' spec."""
    parts = spec.split("x")
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), 81
    raise ValueError(f"Invalid resolution spec: {spec}. Use WxHxF or WxH.")


async def generate_video(args: argparse.Namespace) -> None:
    tokenizer, pipeline, arch = _load_pipeline(args)
    is_animate = arch.name == "WanAnimatePipeline"

    # Resolve arch-specific frame-count defaults now that arch is known.
    # WanAnimate defaults to 77 (one segment); Wan T2V/I2V defaults to 81.
    if args.num_frames is None:
        args.num_frames = 77 if is_animate else 81

    # Build list of (width, height, num_frames) to run
    if args.resolutions:
        res_list = [_parse_resolution(r) for r in args.resolutions]
    else:
        res_list = [(args.width, args.height, args.num_frames)]

    # Optional warmup with small resolution
    if args.warmup:
        import time as _time

        print("Warmup (480x832, 17 frames, 2 steps)...")
        saved = (
            args.width,
            args.height,
            args.num_frames,
            args.num_inference_steps,
        )
        args.width, args.height, args.num_frames = 832, 480, 17
        args.num_inference_steps = 2
        t0 = _time.perf_counter()
        await _run_single(args, tokenizer, pipeline, arch, 0, 1)
        print(f"Warmup done: {_time.perf_counter() - t0:.1f}s")
        args.width, args.height, args.num_frames, args.num_inference_steps = (
            saved
        )

    for res_idx, (w, h, nf) in enumerate(res_list):
        args.width, args.height, args.num_frames = w, h, nf
        print(
            f"\n--- Resolution {res_idx + 1}/{len(res_list)}: {w}x{h}, {nf} frames ---"
        )
        await _run_single(
            args, tokenizer, pipeline, arch, res_idx, len(res_list)
        )


async def _run_single(
    args: argparse.Namespace,
    tokenizer: PixelGenerationTokenizer,
    pipeline: PixelGenerationPipeline,  # type: ignore[type-arg]
    arch: Any,
    res_idx: int,
    total_res: int,
) -> None:
    is_animate = arch.name == "WanAnimatePipeline"

    # Derive effective frame count
    effective_num_frames = args.num_frames
    if arch.name in ("WanPipeline", "WanImageToVideoPipeline"):
        effective_num_frames = _normalize_wan_num_frames(
            args.num_frames, phase="main"
        )

    input_image = None
    if args.input_image:
        from PIL import Image

        image_src = args.input_image
        if image_src.startswith(("http://", "https://")):
            import io
            import urllib.request

            with urllib.request.urlopen(image_src) as resp:
                input_image = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            input_image = Image.open(image_src).convert("RGB")
        print(
            f"Input image: {image_src}"
            f" ({input_image.size[0]}x{input_image.size[1]})"
        )
    pose_video: list[PIL.Image.Image] | None = None
    face_video: list[PIL.Image.Image] | None = None
    background_video: list[PIL.Image.Image] | None = None
    mask_video: list[PIL.Image.Image] | None = None
    if is_animate:
        if not args.pose_video or not args.face_video:
            raise ValueError(
                "Wan-Animate requires --pose-video and --face-video."
            )
        if input_image is None:
            raise ValueError("Wan-Animate requires --input-image.")
        (
            pose_video,
            face_video,
            background_video,
            mask_video,
        ) = _load_videos(args)

        print(
            f"Wan-Animate: {len(pose_video)} pose frames, "
            f"animate_mode={args.animate_mode}, "
            f"segment_len={args.segment_frame_length}"
        )

    print(f"Generating video for prompt: '{args.prompt}'")
    print(
        f"Parameters: {args.height}x{args.width}, {effective_num_frames} frames, "
        f"steps={args.num_inference_steps}, guidance={args.guidance_scale}"
    )

    body = _build_request_body(
        args, args.prompt, num_frames=effective_num_frames
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)

    context = await tokenizer.new_context(
        request,
        input_image=input_image,
        pose_video=pose_video,
        face_video=face_video,
        background_video=background_video,
        mask_video=mask_video,
        animate_mode=args.animate_mode,
        segment_frame_length=args.segment_frame_length,
        prev_segment_conditioning_frames=(
            args.prev_segment_conditioning_frames
        ),
    )

    # Override initial noise if provided (for parity testing)
    if args.initial_noise:
        noise = np.load(args.initial_noise).astype(np.float32)
        context.latents = noise
        print(f"Loaded initial noise from {args.initial_noise}: {noise.shape}")

    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )

    if args.num_warmups > 0:
        body_warmup = _build_request_body(
            args, args.prompt, num_frames=effective_num_frames
        )
        request_warmup = OpenResponsesRequest(
            request_id=RequestID(), body=body_warmup
        )
        context_warmup = await tokenizer.new_context(
            request_warmup,
            input_image=input_image,
            pose_video=pose_video,
            face_video=face_video,
            background_video=background_video,
            mask_video=mask_video,
            animate_mode=args.animate_mode,
            segment_frame_length=args.segment_frame_length,
            prev_segment_conditioning_frames=(
                args.prev_segment_conditioning_frames
            ),
        )
        inputs_warmup = PixelGenerationInputs[PixelContext](
            batch={context_warmup.request_id: context_warmup}
        )
        for i in range(args.num_warmups):
            print(f"Running warmup {i + 1} of {args.num_warmups}")
            pipeline.execute(inputs_warmup)
        print("Warmup complete")

    output: GenerationOutput | None = None
    if args.profile_timings:
        with profile_execute(pipeline, patch_tensor_ops=True) as prof:
            for i in range(args.num_iterations):
                if args.num_iterations > 1:
                    print(f"Running inference {i + 1} of {args.num_iterations}")
                output = pipeline.execute(inputs)[context.request_id]
                output = await tokenizer.postprocess(output)
                if not isinstance(output, GenerationOutput):
                    raise TypeError(
                        "Expected GenerationOutput from PixelGenerationPipeline.execute() and tokenizer.postprocess()."
                    )
        prof.report(unit="ms")
    else:
        t0 = time.perf_counter()
        for i in range(args.num_iterations):
            if args.num_iterations > 1:
                print(f"Running inference {i + 1} of {args.num_iterations}")
            output = pipeline.execute(inputs)[context.request_id]
            output = await tokenizer.postprocess(output)
            if not isinstance(output, GenerationOutput):
                raise TypeError(
                    "Expected GenerationOutput from PixelGenerationPipeline.execute() and tokenizer.postprocess()."
                )
        t1 = time.perf_counter()
        print(f"Timing: execute={t1 - t0:.2f}s")

    if output is None:
        raise RuntimeError("No generation output produced.")

    print("Generation complete!")
    frames = _video_frames_from_output(output)

    if not frames:
        print("ERROR: No frames generated")
        return

    # Save output.
    output_path = args.output
    if total_res > 1:
        base, ext = os.path.splitext(output_path)
        output_path = (
            f"{base}_{args.width}x{args.height}x{args.num_frames}{ext}"
        )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {len(frames)} frames as video to {output_path}")
    save_video(frames, output_path, args.fps)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(generate_video(args))
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)
    raise SystemExit(main())
