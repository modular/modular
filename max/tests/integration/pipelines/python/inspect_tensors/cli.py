#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import click

from inspect_tensors import print_tensors


@click.command()
@click.option(
    "--model-path",
    required=True,
    help="Path to the model or model identifier (e.g., 'google/gemma-3b-it')",
)
@click.option(
    "--framework",
    type=click.Choice(["torch", "max"]),
    default="max",
    help="Framework to use for inference (torch=Torch via Transformers, max=Modular MAX)",
)
@click.option(
    "--device",
    type=str,
    help='Device to use for inference. Examples: "gpu", "cpu", "gpu:0", or "gpu:0,1"',
    default="gpu",
)
def main(
    model_path: str,
    framework: str,
    device: str = "gpu",
) -> None:
    """Run text/image generation using either HuggingFace or MAX framework."""
    print_tensors(
        model_path,
        framework,
        device,
    )


if __name__ == "__main__":
    main()
