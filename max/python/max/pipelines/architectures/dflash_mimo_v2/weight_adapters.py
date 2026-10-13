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
"""Drafter checkpoint -> MAX weights for the MiMo-V2 DFlash drafter.

``dflash/`` holds the drafter's tensors in ``dflash_draft_model.safetensors``
and its trained mask embedding in ``mask_embedding.pt``. Both load loudly:
every tensor the drafter declares must be present with its dtype and shape,
no tensor may go unused, and the mask embedding comes from its own file.
The target's embedding row for ``mask_token_id`` is untrained padding, so
substituting it would load and run while drafting from a near-zero vector.

``mask_embedding.pt`` is read without torch, by an unpickler that admits
only the few globals a saved ``{mask_token_id, embedding}`` dict uses.
"""

from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights

from .model_config import DFlashMiMoV2Config

DRAFT_WEIGHTS_FILE = "dflash_draft_model.safetensors"
MASK_EMBEDDING_FILE = "mask_embedding.pt"
MASK_EMBEDDING = "mask_embedding"

_TORCH_STORAGES = {"BFloat16Storage": (DType.bfloat16, np.uint16)}


def drafter_tensor_shapes(
    config: DFlashMiMoV2Config,
) -> dict[str, tuple[int, ...]]:
    """Returns every drafter checkpoint tensor and its shape.

    Args:
        config: The drafter configuration.

    Returns:
        The shape of each tensor, keyed by its checkpoint name.
    """
    hidden = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (hidden, len(config.target_layer_ids) * hidden),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    for i in range(config.num_hidden_layers):
        layer = f"layers.{i}."
        attn = layer + "self_attn."
        shapes |= {
            layer + "input_layernorm.weight": (hidden,),
            layer + "post_attention_layernorm.weight": (hidden,),
            attn + "q_proj.weight": (q_dim, hidden),
            attn + "k_proj.weight": (kv_dim, hidden),
            attn + "v_proj.weight": (kv_dim, hidden),
            attn + "o_proj.weight": (hidden, q_dim),
            attn + "q_norm.weight": (config.head_dim,),
            attn + "k_norm.weight": (config.head_dim,),
            attn + "attention_sink_bias": (config.num_attention_heads,),
            layer + "mlp.gate_proj.weight": (config.intermediate_size, hidden),
            layer + "mlp.up_proj.weight": (config.intermediate_size, hidden),
            layer + "mlp.down_proj.weight": (hidden, config.intermediate_size),
        }
    return shapes


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    config: DFlashMiMoV2Config,
    mask_embedding: WeightData,
) -> dict[str, WeightData]:
    """Checks the drafter's tensors and adds the mask embedding.

    Args:
        state_dict: The tensors of ``dflash_draft_model.safetensors``.
        config: The drafter configuration.
        mask_embedding: From :func:`load_mask_embedding`.

    Returns:
        Every drafter weight, keyed by its name in :class:`DFlashMiMoV2`.

    Raises:
        ValueError: If a tensor is missing, unexpected, or has the wrong dtype
            or shape.
    """
    shapes = drafter_tensor_shapes(config)
    if missing := sorted(shapes.keys() - state_dict.keys()):
        raise ValueError(
            f"DFlash MiMo-V2: {len(missing)} drafter tensor(s) are missing,"
            f" e.g. {missing[:3]}."
        )
    if unused := sorted(state_dict.keys() - shapes.keys()):
        raise ValueError(
            f"DFlash MiMo-V2: {len(unused)} drafter tensor(s) would go unused,"
            f" e.g. {unused[:3]}."
        )
    weights = {}
    for name, shape in shapes.items():
        data = state_dict[name].data()
        if data.dtype != config.dtype or tuple(data.shape) != shape:
            raise ValueError(
                f"DFlash MiMo-V2: {name} is {data.dtype} {list(data.shape)},"
                f" expected {config.dtype} {list(shape)}."
            )
        weights[name] = data
    weights[MASK_EMBEDDING] = mask_embedding
    return weights


def load_mask_embedding(path: Path, config: DFlashMiMoV2Config) -> WeightData:
    """Reads ``mask_embedding.pt``, the embedding of every block mask slot.

    Args:
        path: The ``mask_embedding.pt`` file.
        config: The drafter configuration, whose ``mask_token_id`` the file
            must name.

    Returns:
        The ``[hidden_size]`` embedding.

    Raises:
        ValueError: If the file is not a ``{mask_token_id, embedding}`` dict
            for this drafter.
    """
    with zipfile.ZipFile(path) as archive:
        (pickle_name,) = [
            n for n in archive.namelist() if n.endswith("/data.pkl")
        ]
        root = pickle_name.removesuffix("data.pkl")
        byteorder = root + "byteorder"
        if (
            byteorder in archive.namelist()
            and archive.read(byteorder) != b"little"
        ):
            raise ValueError(f"DFlash MiMo-V2: {path} is not little-endian.")
        try:
            saved = _TensorUnpickler(archive, root).load()
        except pickle.UnpicklingError as e:
            raise ValueError(f"DFlash MiMo-V2: {path}: {e}") from e

    if not isinstance(saved, dict) or saved.keys() != {
        "mask_token_id",
        "embedding",
    }:
        raise ValueError(
            f"DFlash MiMo-V2: {path} is not a {{mask_token_id, embedding}}"
            " dict."
        )
    if saved["mask_token_id"] != config.mask_token_id:
        raise ValueError(
            f"DFlash MiMo-V2: {path} is for mask_token_id"
            f" {saved['mask_token_id']}, the config names"
            f" {config.mask_token_id}."
        )
    embedding = saved["embedding"]
    shape = (config.hidden_size,)
    if embedding.dtype != config.dtype or tuple(embedding.shape) != shape:
        raise ValueError(
            f"DFlash MiMo-V2: {path} holds {embedding.dtype}"
            f" {list(embedding.shape)}, expected {config.dtype} {list(shape)}."
        )
    return WeightData(embedding, MASK_EMBEDDING, embedding.dtype, Shape(shape))


class _TensorUnpickler(pickle.Unpickler):
    """Rebuilds the contiguous CPU tensors of a ``torch.save`` archive."""

    def __init__(self, archive: zipfile.ZipFile, root: str) -> None:
        super().__init__(archive.open(root + "data.pkl"))
        self._archive = archive
        self._root = root

    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return self._rebuild_tensor
        if module == "torch" and name in _TORCH_STORAGES:
            return name
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid: object) -> tuple[str, bytes]:
        if not (
            isinstance(pid, tuple)
            and len(pid) == 5
            and pid[0] == "storage"
            and pid[1] in _TORCH_STORAGES
        ):
            raise pickle.UnpicklingError(f"unsupported storage {pid!r}")
        _, storage, key, _, _ = pid
        return storage, self._archive.read(f"{self._root}data/{key}")

    @staticmethod
    def _rebuild_tensor(
        storage: tuple[str, bytes],
        offset: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        *_: object,
    ) -> Buffer:
        kind, data = storage
        dtype, bits = _TORCH_STORAGES[kind]
        contiguous = tuple(
            int(np.prod(shape[i + 1 :])) for i in range(len(shape))
        )
        if tuple(stride) != contiguous:
            raise pickle.UnpicklingError(f"tensor is not contiguous: {stride}")
        array = np.frombuffer(data, bits)[offset : offset + int(np.prod(shape))]
        return Buffer.from_numpy(array.reshape(shape).copy()).view(dtype)
