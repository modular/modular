import argparse
import time

from diffusers import Flux2Pipeline
import torch
from max.examples.diffusion.profiler import profile_execute


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
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=3,
        help="Number of warmups to run before profiling.",
    )
    parser.add_argument(
        "--num-profile-iterations",
        type=int,
        default=3,
        help="Number of iterations to run for profiling.",
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


def main():
    args = parse_args()

    pipe = Flux2Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder = torch.compile(
            pipe.text_encoder, mode="max-autotune", fullgraph=True
        )
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune", fullgraph=True
        )
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = torch.compile(pipe.vae, mode="max-autotune", fullgraph=True)

    if args.profile_timings:
        for i in range(args.num_warmups):
            print(f"Running warmup {i + 1} of {args.num_warmups}")
            pipe(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_length,
                output_type="pil",
            )
        with profile_execute(pipe, is_diffusers=True) as prof:
            for i in range(args.num_profile_iterations):
                print(
                    f"Running inference {i + 1} of {args.num_profile_iterations}"
                )
                pipe(
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_length,
                    output_type="pil",
                )
        prof.report()
    else:
        pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_length,
            output_type="pil",
        )


if __name__ == "__main__":
    main()