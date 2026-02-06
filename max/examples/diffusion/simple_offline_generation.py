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

"""Simple offline pixel generation example using diffusion models.

This module demonstrates end-to-end pixel generation using:
- PixelGenerationRequest: Create generation requests with prompts
- PixelGenerationTokenizer: Tokenize prompts and prepare model context
- PixelGenerationPipeline: Execute the diffusion model to generate pixels

Usage:
    ./bazelw run //max/examples/diffusion:simple_offline_generation -- \
        --model black-forest-labs/FLUX.2-dev \
        --prompt "A cat in a garden"
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import cast

import numpy as np
import numpy.typing as npt
from max.driver import DeviceSpec
from max.examples.diffusion.profiler import profile_execute
from max.interfaces import (
    PipelineTask,
    PixelGenerationInputs,
    PixelGenerationRequest,
    RequestID,
)
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.core import PixelContext
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)
from PIL import Image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate images with a diffusion model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Identifier of the model to use for generation (e.g., black-forest-labs/FLUX.2-dev).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the image to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt to guide what NOT to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of generated image in pixels. None uses model's native resolution.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. More steps = higher quality but slower.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier-free guidance. Set to 1.0 to disable CFG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output filename for the generated image.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum length of tokenizer",
    )
    parser.add_argument(
        "--secondary-max-length",
        type=int,
        default=None,
        help="Maximum length of secondary tokenizer",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Input image for image-to-image generation.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Profile timings of the pipeline.",
    )

    args = parser.parse_args(argv)

    # Validate arguments
    assert args.prompt, "Prompt must be a non-empty string."
    if args.height is not None:
        assert args.height > 0, "Height must be a positive integer."
    if args.width is not None:
        assert args.width > 0, "Width must be a positive integer."
    assert args.num_inference_steps > 0, (
        "num-inference-steps must be a positive integer."
    )
    assert args.guidance_scale > 0.0, "guidance-scale must be positive."

    return args


def save_image(pixel_data: np.ndarray, output_path: str) -> None:
    """Save generated pixel data as an image file.

    Args:
        pixel_data: Numpy array of shape (H, W, C) with values in [0, 1]
        output_path: Path where the image should be saved
    """
    try:
        # Convert from float [0, 1] to uint8 [0, 255]
        pixel_data = (pixel_data * 255).clip(0, 255).astype(np.uint8)

        # Create and save image
        image = Image.fromarray(pixel_data)
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    except ImportError:
        print("WARNING: PIL not available, saving as numpy array instead")
        np.save(output_path.replace(".png", ".npy"), pixel_data)
        print(f"Pixel data saved to: {output_path.replace('.png', '.npy')}")


def load_image(image_path: str | None) -> npt.NDArray[np.uint8] | None:
    """Load an image from a file."""
    if image_path is None:
        return None
    return np.array(Image.open(image_path), dtype=np.uint8)


async def generate_image(args: argparse.Namespace) -> None:
    """Main generation logic.

    Args:
        args: Parsed command-line arguments
    """
    print(f"Loading model: {args.model}")

    # Step 1: Initialize pipeline configuration
    config = PipelineConfig(
        model_path=args.model,
        device_specs=[DeviceSpec.accelerator()],
        use_legacy_module=False,
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.model.huggingface_weight_repo,
        use_legacy_module=config.use_legacy_module,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None, (
        "No matching diffusion architecture found for the provided model."
    )

    # Step 2: Initialize the tokenizer
    # The tokenizer handles prompt encoding and context preparation
    has_tokenizer_2 = False
    diffusers_config = config.model.diffusers_config
    max_length = args.max_length
    secondary_max_length = args.secondary_max_length
    if (
        max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and (components_config.get("tokenizer", None) is not None)
    ):
        max_length = components_config["tokenizer"]["config_dict"].get(
            "model_max_length", None
        )
        if arch.name == "Flux2Pipeline":
            max_length = 512
        print(f"Using max length: {max_length} for tokenizer")

    if (
        secondary_max_length is None
        and diffusers_config is not None
        and (components_config := diffusers_config.get("components", None))
        and (components_config.get("tokenizer_2", None) is not None)
    ):
        has_tokenizer_2 = True
        secondary_max_length = components_config["tokenizer_2"][
            "config_dict"
        ].get("model_max_length", None)
        print(
            f"Using secondary max length: {secondary_max_length} for tokenizer_2"
        )

    tokenizer = PixelGenerationTokenizer(
        model_path=args.model,
        pipeline_config=config,
        subfolder="tokenizer",  # Tokenizer is in a subfolder for diffusion models
        max_length=max_length,
        subfolder_2="tokenizer_2" if has_tokenizer_2 else None,
        secondary_max_length=secondary_max_length if has_tokenizer_2 else None,
    )

    # Step 3: Initialize the pipeline
    # The pipeline executes the diffusion model
    if not issubclass(arch.pipeline_model, DiffusionPipeline):
        raise TypeError(
            "Selected architecture does not implement DiffusionPipeline: "
            f"{arch.pipeline_model}"
        )
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[PixelContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )

    print(f"Generating image for prompt: '{args.prompt}'")

    # Step 4: Create a PixelGenerationRequest
    request = PixelGenerationRequest(
        request_id=RequestID(),
        model_name=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        input_image=load_image(args.input_image),
    )

    print(
        f"Parameters: steps={args.num_inference_steps}, guidance={args.guidance_scale}"
    )

    # Step 5: Create a PixelContext object from the request
    # The tokenizer handles prompt tokenization, timestep scheduling,
    # latent initialization, and all other preprocessing
    context = await tokenizer.new_context(request)

    print(
        f"Context created: {context.height}x{context.width}, {context.num_inference_steps} steps"
    )

    # Step 6: Prepare inputs for the pipeline
    # Create a batch with a single context
    inputs = PixelGenerationInputs[PixelContext](
        batch={context.request_id: context}
    )

    # Step 7: Execute the pipeline
    print("Running diffusion model...")
    if args.profile_timings:
        with profile_execute(
            pipeline, patch_concat=True, patch_tensor_ops=True
        ) as prof:
            outputs = pipeline.execute(inputs)
        print(f"Method timings:\n{prof.report(unit='ms')}")
        print(f"Module timings:\n{prof.report_modules(unit='ms')}")
    else:
        outputs = pipeline.execute(inputs)

    # Step 8: Get the output for our request
    output = outputs[context.request_id]

    # Check if generation completed successfully
    if not output.is_done:
        print(f"WARNING: Generation status: {output.final_status}")
        return

    print("Generation complete!")

    # Step 9: Post-process the pixel data
    # The tokenizer's postprocess method converts from model output format
    # (NCHW, [-1, 1]) to display format (NHWC, [0, 1])
    pixel_data = await tokenizer.postprocess(output.pixel_data)

    # Step 10: Save the image
    # Take the first image if multiple were generated
    if pixel_data.shape[0] > 0:
        save_image(pixel_data[0], args.output)
    else:
        print("ERROR: No pixel data generated")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the pixel generation example.

    Args:
        argv: Optional explicit list of argument strings. If None, arguments
            are read from sys.argv[1:].

    Returns:
        Process exit code. 0 indicates success.
    """
    args = parse_args(argv)

    try:
        asyncio.run(generate_image(args))
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
