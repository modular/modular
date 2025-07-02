# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tensor I/O utilities for loading and saving MAX and PyTorch tensors."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import torch
from max.driver.tensor import load_max_tensor
from torch.utils.dlpack import from_dlpack


def load_max_tensor_data(filepath: Path) -> Optional[torch.Tensor]:
    """Load a MAX tensor from .max file and convert to PyTorch tensor."""
    try:
        # Verify saved file exists
        if not filepath.exists():
            print(f"File {filepath} does not exist")
            return None

        # Load the tensor using the correct function
        loaded_driver_tensor = load_max_tensor(filepath)

        # Convert to torch tensor using dlpack (handles bfloat16 and other dtypes)
        torch_tensor = from_dlpack(loaded_driver_tensor)

        # Debug: print actual shape and dtype
        print(
            f"Loaded MAX tensor from {filepath.name}: shape={torch_tensor.shape}, dtype={torch_tensor.dtype}"
        )

        return torch_tensor
    except Exception as e:
        print(f"Error loading MAX tensor from {filepath}: {e}")
        traceback.print_exc()
        return None


def load_pytorch_tensor(filepath: Path) -> Optional[torch.Tensor]:
    """Load a PyTorch tensor from .pt file."""
    try:
        tensor = torch.load(filepath, map_location="cpu")
        torch_tensor = (
            tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
        )

        # Debug: print actual shape and dtype
        print(
            f"Loaded PyTorch tensor from {filepath.name}: shape={torch_tensor.shape}, dtype={torch_tensor.dtype}"
        )

        return torch_tensor
    except Exception as e:
        print(f"Error loading PyTorch tensor from {filepath}: {e}")
        traceback.print_exc()
        return None
