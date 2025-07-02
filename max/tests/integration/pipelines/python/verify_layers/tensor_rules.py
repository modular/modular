# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tensor processing rules for special cases in layer verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass
class TensorProcessingRule:
    """Rule for processing tensors in special cases."""

    layer_pattern: str | list[str]  # Regex pattern to match layer names
    description: str
    pytorch_transform: Optional[Callable[[Any], Any]] = (
        None  # torch.Tensor type
    )
    max_transform: Optional[Callable[[Any], Any]] = None  # torch.Tensor type


def apply_lm_head_rule(tensor: torch.Tensor) -> torch.Tensor:
    """Take only the last token from lm_head output."""
    if len(tensor.shape) >= 2:
        # Take the last token along the sequence dimension (first dimension)
        return tensor[-1:, ...]  # Keep as [1, vocab_size]
    return tensor


# Special case rules registry
TENSOR_PROCESSING_RULES = [
    TensorProcessingRule(
        layer_pattern=[r".*lm_head.*", r".*model_norm.*"],
        description="LM head: take last token from MAX to match PyTorch shape",
        max_transform=apply_lm_head_rule,
        pytorch_transform=apply_lm_head_rule,
    ),
]


def apply_tensor_processing_rules(
    layer_name: str, max_tensor: torch.Tensor, pytorch_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply special processing rules to tensors based on layer name.

    Returns:
        Tuple of (processed_max_tensor, processed_pytorch_tensor)
    """

    processed_max = max_tensor
    processed_pytorch = pytorch_tensor

    for rule in TENSOR_PROCESSING_RULES:
        # Handle both string and list patterns
        patterns = (
            rule.layer_pattern
            if isinstance(rule.layer_pattern, list)
            else [rule.layer_pattern]
        )

        # Check if any pattern matches
        pattern_matched = False
        for pattern in patterns:
            if re.match(pattern, layer_name, re.IGNORECASE):
                pattern_matched = True
                break

        if pattern_matched:
            print(f"Applying rule for {layer_name}: {rule.description}")

            if rule.max_transform:
                processed_max = rule.max_transform(processed_max)
                print(
                    f"  MAX tensor shape after transform: {processed_max.shape}"
                )

            if rule.pytorch_transform:
                processed_pytorch = rule.pytorch_transform(processed_pytorch)
                print(
                    f"  PyTorch tensor shape after transform: {processed_pytorch.shape}"
                )

            break  # Apply only the first matching rule

    return processed_max, processed_pytorch
