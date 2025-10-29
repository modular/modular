# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Utilities for comparing tensors."""

import torch


def assert_tensors_close(
    torch_output: torch.Tensor,
    max_output: torch.Tensor,
    rtol: float,
    atol: float,
    message: str,
) -> None:
    """Compare two tensors with appropriate tolerances."""
    # Convert to same dtype for comparison
    torch_output = torch_output.to(torch.bfloat16)
    max_output = max_output.to(torch.bfloat16)

    # Remove batch dimension if present for comparison
    if torch_output.dim() > max_output.dim():
        torch_output = torch_output.squeeze(0)
    elif max_output.dim() > torch_output.dim():
        max_output = max_output.squeeze(0)

    try:
        torch.testing.assert_close(
            max_output,
            torch_output,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as e:
        # Calculate percentage of incorrect values
        is_close = torch.isclose(torch_output, max_output, rtol=rtol, atol=atol)
        num_incorrect = (~is_close).sum().item()
        total_elements = torch_output.numel()
        percent_incorrect = (num_incorrect / total_elements) * 100

        # Add more context to the error
        shape_info = (
            f"Shapes: torch={torch_output.shape}, max={max_output.shape}"
        )
        dtype_info = (
            f"Dtypes: torch={torch_output.dtype}, max={max_output.dtype}"
        )
        device_info = (
            f"Devices: torch={torch_output.device}, max={max_output.device}"
        )
        accuracy_info = f"Incorrect values: {num_incorrect}/{total_elements} ({percent_incorrect:.2f}%)"

        raise AssertionError(
            f"\n{'=' * 80}\n{message}\n{shape_info}\n{dtype_info}\n{device_info}\n{accuracy_info}\n\nOriginal error:\n{e}\n{'=' * 80}"
        ) from e
