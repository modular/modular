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
"""Maps Mistral Large 4's consolidated checkpoint onto MAX module names.

The name table and the per-tensor role decisions live in
:mod:`.weight_contract`, which ``engine_core::mistral_large4`` mirrors; this
module only applies them to a MAX ``Weights`` state dict. Two conversions
touch values, both layout-only: the FP8 block-scale 4-D -> 2-D change is a
reshape of the same elements, the NVFP4 globals are already the convention
MAX's matmul multiplies by, and the vision tower's Q and K rows are permuted
from the checkpoint's interleaved-RoPE layout into the rotate-half layout
the tower computes with (transformers' ``PermuteForRope`` vision rule).
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers import PretrainedConfig

from .weight_contract import (
    BASE_REPO_SCALE_SUFFIX,
    ROLE_FP8_SCALE,
    ROLE_VISION_QK,
    VISION_ATTENTION_HEADS,
    skip_reason,
    transform_weight_name,
    vision_weight_table,
    weight_table,
)

__all__ = [
    "convert_safetensor_state_dict",
    "is_skipped_checkpoint_tensor",
    "transform_weight_name",
]

# The four storage dtypes this checkpoint uses, spelled as its safetensors
# headers spell them, so the role decision reads the same strings on both sides.
_SAFETENSORS_DTYPE = {
    DType.bfloat16: "BF16",
    DType.float32: "F32",
    DType.float8_e4m3fn: "F8_E4M3",
    DType.uint8: "U8",
}


def is_skipped_checkpoint_tensor(name: str) -> bool:
    """Whether no graph loads ``name``: only the DSpark drafter subtree."""
    return skip_reason(name) == "dspark"


def _headers(
    state_dict: dict[str, Weights],
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The ``name -> (dtype, shape)`` view the contract rules resolve against.

    Only the DSpark drafter subtree is left out: the text table skips the
    vision names by prefix, and the vision table reads them, so both
    resolve against the same headers.
    """
    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, value in state_dict.items():
        if skip_reason(name) == "dspark":
            continue
        data = value.data()
        try:
            dtype = _SAFETENSORS_DTYPE[data.dtype]
        except KeyError:
            raise ValueError(
                f"{name}: unexpected storage dtype {data.dtype}; Mistral Large "
                f"4 stores only {sorted(_SAFETENSORS_DTYPE.values())}"
            ) from None
        out[name] = (dtype, tuple(int(dim) for dim in data.shape))
    return out


def _as_block_grid(data: WeightData, name: str) -> WeightData:
    """Reshapes an FP8 block scale's 4-D storage onto MAX's 2-D block grid.

    The elements are already in grid order, so this is a reshape rather than a
    transpose; the copy is the ``np.from_dlpack`` read-only view's price, not a
    reordering.
    """
    array = np.from_dlpack(data)
    grid = array.reshape(array.shape[0], array.shape[2])
    return WeightData.from_numpy(grid.copy(), name)


def _as_rope_half_split(data: WeightData, name: str) -> WeightData:
    """Permutes a vision Q/K weight's rows into the rotate-half rope layout.

    The checkpoint stores each head's rope rows interleaved -- adjacent row
    pairs ``(x0, x1), (x2, x3), ...`` share one frequency -- while the tower
    applies rotate-half rope, so within every head the even rows move ahead
    of the odd ones. This is transformers' ``PermuteForRope`` vision rule,
    which ``from_pretrained`` applies when this checkpoint is loaded as
    ``MistralLarge4ForConditionalGeneration``; parity with that reference
    requires the same permutation here.
    """
    shape = tuple(int(dim) for dim in data.shape)
    rows, width = shape[0], shape[1]
    heads = VISION_ATTENTION_HEADS
    half_head = rows // heads // 2
    # BF16 has no numpy dtype, so the rows are permuted over the weight's
    # uint16 storage view; the permutation is bit-exact.
    raw = np.from_dlpack(Buffer.from_dlpack(data.data).view(DType.uint16))
    permuted = (
        raw.reshape(heads, half_head, 2, width)
        .transpose(0, 2, 1, 3)
        .reshape(rows, width)
    )
    buf = Buffer.from_dlpack(permuted).view(dtype=data.dtype, shape=shape)
    return WeightData(
        data=buf,
        name=name,
        dtype=data.dtype,
        shape=data.shape,
        quantization_encoding=data.quantization_encoding,
    )


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Adapts a Mistral Large 4 checkpoint to MAX's module names.

    Args:
        state_dict: The checkpoint's tensors, keyed by their consolidated names.
        huggingface_config: The generated ``config.json``. Unread: every rule
            here is a property of the tensor set, so the adapter cannot
            disagree with the header contract by reading config differently.
        **unused_kwargs: Ignored, so the signature stays compatible with the
            ``SafetensorStateDictConverter`` protocol.

    Returns:
        The module-name table: FP8 block scales on the 2-D grid, the vision
        subtree under the vision module names, and the drafter subtree
        dropped.

    Raises:
        ValueError: If the checkpoint uses a scale convention this loader does
            not implement, or a scale's shape does not fit its weight's.
    """
    del huggingface_config
    headers = _headers(state_dict)
    # The tables validate every role and shape across the whole tensor set,
    # so the per-tensor pass below can rely on them. The vision table is
    # what refuses a quantized vision tensor, whose conventions this loader
    # does not implement.
    rows = {tensor.source: tensor for tensor in weight_table(headers).tensors}
    vision_rows = {
        tensor.source: tensor for tensor in vision_weight_table(headers).tensors
    }

    out: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        if name in vision_rows:
            row = vision_rows[name]
            data = value.data()
            target = row.name
            if row.role == ROLE_VISION_QK:
                data = _as_rope_half_split(data, target)
            elif target != name:
                data = WeightData(
                    data=data.data,
                    name=target,
                    dtype=data.dtype,
                    shape=data.shape,
                    quantization_encoding=data.quantization_encoding,
                )
            out[target] = data
            continue
        if skip_reason(name) is not None:
            continue
        if name.endswith(BASE_REPO_SCALE_SUFFIX):
            raise ValueError(
                f"{name}: the base FP8 checkpoint's 2-D qscale_weight "
                "convention is a separate loader and is not wired yet"
            )
        data = value.data()
        row = rows[name]
        target = row.name
        if row.role == ROLE_FP8_SCALE:
            data = _as_block_grid(data, target)
        elif target != name:
            data = WeightData(
                data=data.data,
                name=target,
                dtype=data.dtype,
                shape=data.shape,
                quantization_encoding=data.quantization_encoding,
            )
        out[target] = data
    return out
