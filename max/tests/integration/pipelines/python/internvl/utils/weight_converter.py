# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Weight conversion utilities for InternVL tests."""

import torch


def convert_hf_to_max_weights(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace format weights to MAX format using weight adapters.

    Args:
        hf_state_dict: Dictionary of HuggingFace format weights (torch tensors).

    Returns:
        Dictionary of MAX format weights (torch tensors).
    """
    # Check if these are full model weights (have vision_model prefix) or component weights
    has_vision_model_prefix = any(
        k.startswith("vision_model.") for k in hf_state_dict
    )

    if has_vision_model_prefix:
        # For full model weights, apply weight mapping transformations directly
        result = {}
        for k, v in hf_state_dict.items():
            # Apply the same transformations that the weight adapter would do
            new_key = k

            # Convert HuggingFace vision model keys to MAX format
            if k.startswith("vision_model."):
                # Remove the vision_model prefix
                new_key = k[len("vision_model.") :]

                # Apply specific key transformations
                if "encoder.layers." in new_key:
                    new_key = new_key.replace(
                        "encoder.layers.", "encoder_layers."
                    )

                # Transform attention keys
                if ".attn.qkv." in new_key:
                    new_key = new_key.replace(".attn.qkv.", ".attn.qkv_proj.")
                elif ".attn.proj." in new_key:
                    new_key = new_key.replace(".attn.proj.", ".attn.o_proj.")
                elif "embeddings.patch_embedding." in new_key:
                    # Handle patch embedding transformations
                    if new_key.endswith(".weight") and len(v.shape) == 4:
                        # Convert Conv2D weight to a Linear-equivalent format.
                        # PyTorch Conv2D weight: (out_C, in_C, kH, kW)
                        # Reshape to (out_C, in_C * kH * kW)
                        out_channels, in_channels, kernel_h, kernel_w = v.shape
                        v = v.reshape(out_channels, -1)

            # Handle MLP1 (multimodal projector) key transformations
            # Convert numbered format to descriptive names that MAX expects
            if "mlp1.0." in new_key:
                new_key = new_key.replace("mlp1.0.", "mlp1.layer_norm.")
            elif "mlp1.1." in new_key:
                new_key = new_key.replace("mlp1.1.", "mlp1.fc1.")
            elif "mlp1.3." in new_key:
                new_key = new_key.replace("mlp1.3.", "mlp1.fc2.")

            result[new_key] = v
    else:
        # Handle component weights (encoder layer, attention only) with simplified mapping
        result = {}
        for k, v in hf_state_dict.items():
            # For component-level weights, apply specific transformations
            new_key = k

            # Map HuggingFace format keys to MAX format for component tests
            # For direct attention component tests: qkv.weight -> qkv_proj.weight
            if new_key.startswith("qkv."):
                new_key = new_key.replace("qkv.", "qkv_proj.")
            elif new_key.startswith("proj."):
                new_key = new_key.replace("proj.", "o_proj.")
            # For encoder layer tests: attn.qkv.weight -> attn.qkv_proj.weight
            elif "attn.qkv." in new_key:
                new_key = new_key.replace("attn.qkv.", "attn.qkv_proj.")
            elif "attn.proj." in new_key:
                new_key = new_key.replace("attn.proj.", "attn.o_proj.")

            # Handle patch embedding transformations
            if "patch_embedding." in new_key:
                # Handle patch embedding transformations
                if new_key.endswith(".weight") and len(v.shape) == 4:
                    # Convert Conv2D weight to a Linear-equivalent format.
                    out_channels, in_channels, kernel_h, kernel_w = v.shape
                    v = v.reshape(out_channels, -1)

            result[new_key] = v

    return result
