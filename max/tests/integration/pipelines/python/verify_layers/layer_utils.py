# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Utilities for layer name normalization and matching."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def normalize_max_layer_name(max_name: str) -> str:
    """Convert MAX layer name to normalized format for comparison."""
    # Remove file extension if present
    max_name = max_name.removesuffix(".max")

    # Apply mappings
    normalized = max_name

    return normalized


def normalize_pytorch_layer_name(pytorch_name: str) -> str:
    """Convert PyTorch layer name to normalized format for comparison."""
    # Remove file extension if present
    pytorch_name = pytorch_name.removesuffix(".pt")

    # Remove model prefix (e.g., "model.lm_head" -> "lm_head")
    normalized = re.sub(r"^model\.", "", pytorch_name)

    return normalized


def load_execution_order(layer_data_path: Path | None = None) -> dict[str, int]:
    """Load the execution order from the PyTorch modules file.

    Args:
        layer_data_path: Path to the layer data directory containing the module_names_torch.txt file

    Returns:
        Dictionary mapping normalized PyTorch module names to their execution order index
    """
    if layer_data_path is None:
        layer_data_path = Path.cwd()

    execution_order_file = layer_data_path / "module_names_torch.txt"
    execution_order = {}
    try:
        with open(execution_order_file) as f:
            for idx, line in enumerate(f):
                module_name = line.strip()
                if module_name:
                    # Normalize the module name the same way we normalize pytorch layer names
                    normalized_name = normalize_pytorch_layer_name(module_name)
                    execution_order[normalized_name] = idx
    except FileNotFoundError:
        print(
            f"Warning: Execution order file not found at {execution_order_file}"
        )
    except Exception as e:
        print(f"Warning: Error loading execution order file: {e}")

    return execution_order


def find_matching_layers(
    max_layers: dict[str, Any],
    torch_layers: dict[str, Any],
    layer_data_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Find matching layers between MAX and PyTorch based on normalized names.

    Args:
        max_layers: MAX layer metadata
        torch_layers: PyTorch layer metadata
        layer_data_path: Path to the layer data directory

    Returns:
        List of tuples (max_layer_name, torch_layer_name, normalized_name)
        sorted by execution order from the PyTorch modules file.
    """
    matches = []

    # Create normalized name mappings
    max_normalized = {}
    torch_normalized = {}

    for max_name in max_layers.keys():
        normalized = normalize_max_layer_name(max_name)
        max_normalized[normalized] = max_name

    for torch_name in torch_layers.keys():
        normalized = normalize_pytorch_layer_name(torch_name)
        torch_normalized[normalized] = torch_name

    # Find matches
    for normalized_name in max_normalized.keys():
        if normalized_name in torch_normalized:
            matches.append(
                (
                    max_normalized[normalized_name],
                    torch_normalized[normalized_name],
                    normalized_name,
                )
            )

    # Load execution order and sort matches accordingly
    execution_order = load_execution_order(layer_data_path)

    # Sort by execution order, fallback to normalized name if not found in execution order
    def get_sort_key(match_tuple):
        normalized_name = match_tuple[2]
        # Return execution order index if found, otherwise return a large number to put it at the end
        return execution_order.get(normalized_name, float("inf"))

    return sorted(matches, key=get_sort_key)
