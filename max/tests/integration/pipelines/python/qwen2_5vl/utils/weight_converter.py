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
        # For  weights, no key transformation needed as MAX uses same naming
        # (qkv.weight, qkv.bias, proj.weight, proj.bias)

        result[new_key] = v

    return result


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
