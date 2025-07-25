# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Weight conversion utilities for Qwen2.5VL tests."""

import torch


def convert_hf_to_max_weights(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace format weights to MAX format for Qwen2.5VL.

    Args:
        hf_state_dict: Dictionary of HuggingFace format weights (torch tensors).

    Returns:
        Dictionary of MAX format weights (torch tensors).
    """
    result = {}
    for k, v in hf_state_dict.items():
        new_key = k

        # Convert Conv3D weights from HF format (out_channels, in_channels, depth, height, width)
        # to MAX format (depth, height, width, in_channels, out_channels)
        if k == "proj.weight" and v.dim() == 5:
            # Permute from [out_channels, in_channels, depth, height, width]
            # to [depth, height, width, in_channels, out_channels]
            v = v.permute(2, 3, 4, 1, 0).contiguous()

        # For attention weights, no key transformation needed as MAX uses same naming
        # (qkv.weight, qkv.bias, proj.weight, proj.bias)

        result[new_key] = v

    return result
