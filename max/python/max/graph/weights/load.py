# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Function for loading paths as Weights."""

import os
from pathlib import Path

from ._loader_wrappers import GGUFWeights, PytorchWeights
from .format import WeightsFormat, weights_format
from .load_safetensors import SafetensorWeights
from .weights import Weights


def load_weights(paths: list[Path]) -> Weights:
    """Loads neural network weights from checkpoint files.

    Automatically detects checkpoint formats based on file extensions and returns
    the appropriate `Weights` implementation, creating a seamless interface for
    loading weights from different formats.


    Supported formats:
    - `Safetensors`: .safetensors
    - `PyTorch`: .bin, .pt, .pth
    - `GGUF`: .gguf

    The following example shows how to load weights from a Safetensors file:

    .. code-block:: python

        from pathlib import Path
        from max.graph.weights import load_weights

        # Load multi-file checkpoints
        sharded_paths = [
            Path("model-00001-of-00003.safetensors"),
            Path("model-00002-of-00003.safetensors"),
            Path("model-00003-of-00003.safetensors")
        ]
        weights = load_weights(sharded_paths)
        layer_weight = weights.model.layers[23].mlp.gate_proj.weight.allocate(
            dtype=DType.float32,
            shape=[4096, 14336],
            device=DeviceRef.GPU(0)
        )

    Args:
        paths: List of `pathlib.Path` objects pointing to checkpoint files.
            For multi-file checkpoints (e.g., sharded `Safetensors`), provide
            all file paths in the list. For single-file checkpoints, provide
            a list with one path.
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
