# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for loading paths as Weights."""

import os
from pathlib import Path

from .format import WeightsFormat, weights_format
from .load_gguf import GGUFWeights
from .load_pytorch import PytorchWeights
from .load_safetensors import SafetensorWeights
from .weights import Weights


def load_weights(paths: list[Path]) -> Weights:
    """Loads weight paths into a Weights object.

    Args:
        paths:
          Local paths of weight files to load.

    Returns:
        A `Weights` object, with all of the associated weights loaded into a single object.

    Raises:
        ValueError: If an empty paths list is passed.

        ValueError: If a path provided does not exist.

    """
    # Check that paths is not empty.
    if not paths:
        raise ValueError("no paths provided, cannot load weights.")

    # Check that all paths exist
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(
                f"file path ({path}) does not exist, cannot load weights."
            )

    _weights_format = weights_format(paths)

    if _weights_format == WeightsFormat.gguf:
        if len(paths) > 1:
            raise ValueError("loading multiple gguf files is not supported.")

        return GGUFWeights(paths[0])
    elif _weights_format == WeightsFormat.safetensors:
        return SafetensorWeights(paths)
    elif _weights_format == WeightsFormat.pytorch:
        return PytorchWeights(paths[0])
    else:
        raise ValueError(
            f"loading weights format '{_weights_format}' not supported."
        )
