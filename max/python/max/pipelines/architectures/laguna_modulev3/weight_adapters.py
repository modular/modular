# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Weight adapter for the Laguna checkpoint family on ModuleV3.

The checkpoint-level work -- compressed-tensors NVFP4 suffixes, reciprocated
global scales, e4m3 block scales, the Laguna structural renames, dropped FP8
KV-cache scales -- is the graph architecture's adapter, run unchanged. ModuleV3
names a quantized weight as its checkpoint does, so this module only adds the
``language_model.`` prefix its tree nests everything under:

.. code-block:: text

    <name>  -> language_model.<name>

It also repairs two things the ModuleV3 loader is stricter about than the graph
loader: a global scale shipped as a one-element vector, and a block scale whose
storage is still uint8 under a float8 label.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.driver import Buffer
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.laguna.weight_adapters import (
    convert_safetensor_state_dict as convert_graph_state_dict,
)
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

MODULEV3_PREFIX = "language_model."

# ``NVFP4Tensor`` declares its global scales as scalars; the checkpoint may
# ship them as one-element vectors.
_SCALAR_SCALE_SUFFIXES = (".weight_scale_2", ".input_scale")


def _as_scalar(data: WeightData, name: str) -> WeightData:
    """Reshapes a one-element global scale to the scalar ``NVFP4Tensor`` wants."""
    arr = np.from_dlpack(data)
    if arr.ndim == 0:
        return data
    if arr.size != 1:
        raise ValueError(
            f"{name} should be a single per-tensor scale, got shape "
            f"{list(arr.shape)}."
        )
    # ``np.from_dlpack`` views are read-only, which DLPack cannot re-export.
    return WeightData.from_numpy(arr.reshape(()).copy(), name)


def _with_storage_dtype(data: WeightData) -> WeightData:
    """Makes a relabeled tensor's storage carry its declared dtype.

    The graph adapter reinterprets uint8 block scales as ``float8_e4m3fn`` by
    relabeling the ``WeightData``. The graph loader honors the label; the
    ModuleV3 loader reads the storage through DLPack, sees uint8, and rejects
    it. A same-width view fixes the storage without copying.
    """
    storage = Buffer.from_dlpack(data.data)
    if storage.dtype == data.dtype:
        return data
    return WeightData(
        storage.view(data.dtype), data.name, data.dtype, data.shape
    )


def to_modulev3_names(
    state_dict: Mapping[str, WeightData],
) -> dict[str, WeightData]:
    """Moves graph-architecture Laguna weight names onto ModuleV3 paths.

    Args:
        state_dict: Weights as the graph Laguna adapter names them.

    Returns:
        The same tensors under their ModuleV3 attribute paths.
    """
    out: dict[str, WeightData] = {}
    for name, data in state_dict.items():
        v3_name = f"{MODULEV3_PREFIX}{name}"
        if name.endswith(_SCALAR_SCALE_SUFFIXES):
            data = _as_scalar(data, v3_name)
        elif name.endswith(".weight_scale"):
            data = _with_storage_dtype(data)
        out[v3_name] = data
    return out


def to_graph_names(
    state_dict: Mapping[str, WeightData],
) -> dict[str, WeightData]:
    """Inverts :func:`to_modulev3_names`.

    Lets the ModuleV3 pipeline model reuse the graph architecture's per-layer
    precision detection, which is written against the graph names.

    Args:
        state_dict: Weights under their ModuleV3 attribute paths.

    Returns:
        The same tensors under the graph Laguna adapter's names.
    """
    return {
        name.removeprefix(MODULEV3_PREFIX): data
        for name, data in state_dict.items()
    }


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Converts Laguna safetensor weights to ModuleV3 names.

    Args:
        state_dict: The checkpoint tensors, by HuggingFace name.
        huggingface_config: The HuggingFace model configuration.
        pipeline_config: The MAX pipeline configuration.

    Returns:
        The weights under their ModuleV3 attribute paths.
    """
    return to_modulev3_names(
        convert_graph_state_dict(
            state_dict, huggingface_config, pipeline_config
        )
    )
