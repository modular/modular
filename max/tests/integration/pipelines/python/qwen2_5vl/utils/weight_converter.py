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
        if "patch_embed.proj." in new_key:
            # Convert Conv2d weight to a Linear-equivalent format.
            # Conv3D weight shape: (2, 14, 14, 3, 1152) when permuted
            # Linear weight shape: (1152, 1176)
            out_channels, _in_channels, _kernel_h, _kernel_w, _kernel_d = (
                v.shape
            )
            v = v.reshape(out_channels, -1)
        # For  weights, no key transformation needed as MAX uses same naming
        # (qkv.weight, qkv.bias, proj.weight, proj.bias)

        result[new_key] = v

    return result


def patch_embed_MAX_to_HF(
    patch_embed_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert MAX format weights to HuggingFace format for Qwen2.5VL."""
    out_channels, _in_channels, _kernel_h, _kernel_w, _kernel_d = (
        patch_embed_weights["proj.weight"].shape
    )
    state_dict = {
        "proj.weight": patch_embed_weights["proj.weight"].reshape(
            out_channels, -1
        )
    }
    return state_dict


def patch_merger_MAX_to_HF(
    mlp_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert MAX format weights to HuggingFace format for Qwen2.5VL."""

    state_dict = {
        "ln_q.weight": mlp_weights["norm.weight"],
        "mlp.0.weight": mlp_weights["mlp.0.weight"],
        "mlp.0.bias": mlp_weights["mlp.0.bias"],
        "mlp.2.weight": mlp_weights["mlp.2.weight"],
        "mlp.2.bias": mlp_weights["mlp.2.bias"],
    }

    return state_dict
