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
"""Initializes the checkpoint's host Engram tables through a compiled graph."""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU, Buffer, is_virtual_device_mode
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Module
from max.nn.legacy.graph_groups import StaticBuffers
from max.pipelines.lib.graph_specs import GraphSpec
from max.pipelines.lib.registration_runtime import (
    ConstantInputs,
    allocate_static_buffers,
)
from max.tensor import Tensor, default_device

from ..model import _check_engram_host_ram
from ..model_config import EngramTableFormat

# Rows widened per slice, so the widening temporaries (int32 lookup
# indices among them) stay about a GiB while the largest table holds
# hundreds of millions of rows.
_CHUNK_ROWS = 1 << 20


# Each E2M1 code's E4M3 byte. Magnitudes 0..7 are 0, 0.5, 1, 1.5, 2, 3, 4
# and 6, all exact in E4M3; code bit 3 is the sign (E4M3 bit 7).
_E2M1_AS_E4M3 = np.array(
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C], np.uint8
)
_E2M1_AS_E4M3 = np.concatenate([_E2M1_AS_E4M3, _E2M1_AS_E4M3 | 0x80])


def _initialize_table(
    destination: BufferValue, source: TensorValue, *, packed: bool
) -> None:
    """Writes a table in its own dtype, widening packed E2M1 pairs to E4M3.

    An unpacked source already holds ``destination``'s dtype. Packed pairs
    widen, low nibble first, by gathering from the 16 E4M3 codes, over row
    slices, so no full-table temporary exists next to the destination. Only
    data moves on the host: no op computes in an FP8 type.
    """
    if not packed:
        ops.buffer_store(destination, source)
        return
    if destination.dtype != DType.float8_e4m3fn:
        raise ValueError(
            f"packed E2M1 codes widen to float8_e4m3fn, not {destination.dtype}"
        )
    cpu = DeviceRef.CPU()
    codes = ops.constant(
        Buffer.from_numpy(_E2M1_AS_E4M3).view(DType.float8_e4m3fn),
        device=cpu,
    )
    rows = int(source.shape[0])
    sixteen = ops.constant(16, DType.uint8, cpu)
    for begin in range(0, rows, _CHUNK_ROWS):
        end = min(begin + _CHUNK_ROWS, rows)
        chunk = source[begin:end]
        high = ops.floor_div(chunk, sixteen)
        low = chunk - high * sixteen
        nibbles = ops.stack([low, high], axis=-1).cast(DType.int32)
        ops.buffer_store_slice(
            destination,
            ops.gather(
                codes,
                nibbles.reshape([end - begin, 2 * int(source.shape[1])]),
                axis=0,
            ),
            [slice(begin, end), slice(None)],
        )


_initialize_table_op = F.functional(_initialize_table)


class _Container(Module[[], None]):
    engram: Any
    embed: Any

    def forward(self) -> None:
        raise NotImplementedError(
            "Host table parameters run through their initializer"
        )


class _RawTable(_Container):
    weight: Tensor
    scale: Tensor

    def __init__(self, rows: int, head_dim: int, packed: bool) -> None:
        super().__init__()
        # The checkpoint's own dtypes: E4M3 codes (MXFP4 packs E2M1 pairs
        # into bytes) and one E8M0 scale per 32 codes.
        self.weight = Tensor.zeros(
            [rows, head_dim // 2 if packed else head_dim],
            dtype=DType.uint8 if packed else DType.float8_e4m3fn,
        )
        self.scale = Tensor.zeros(
            [rows, head_dim // 32], dtype=DType.float8_e8m0fnu
        )


class EngramStaticInitializer(Module[[Mapping[str, Any]], tuple[()]]):
    """Declares the checkpoint tables and writes the supplied pinned buffers."""

    def __init__(
        self,
        layer_ids: tuple[int, ...],
        rows: tuple[int, ...],
        head_dim: int,
        packed: bool,
    ) -> None:
        super().__init__()
        self.layer_ids = layer_ids
        self.packed = packed
        self.layers = _Container()
        with default_device(CPU()):
            for layer, count in zip(layer_ids, rows, strict=True):
                holder = _Container()
                engram = _Container()
                engram.embed = _RawTable(count, head_dim, packed)
                holder.engram = engram
                setattr(self.layers, str(layer), holder)

    def forward(self, inputs: Mapping[str, Any]) -> tuple[()]:
        """Fills every table exactly once during deployment startup."""
        constants = inputs["constants"]
        for ordinal, layer in enumerate(self.layer_ids):
            table = getattr(self.layers, str(layer)).engram.embed
            _initialize_table_op(
                constants[f"engram_table_{ordinal}_weight"],
                table.weight,
                packed=self.packed,
            )
            _initialize_table_op(
                constants[f"engram_table_{ordinal}_scale"],
                table.scale,
                packed=False,
            )
        return ()


def build_engram_static_inputs(
    config: Any, state_dict: Mapping[str, Any]
) -> ConstantInputs:
    """Builds immutable pinned resources and their raw-weight initializer.

    The allocation is deployment-owned; it is never retained in a global cache.
    Virtual exports record the same initializer without allocating host tables.
    """
    assert config.pipeline_config is not None
    assert config.text_config is not None
    layer_ids = tuple(config.engram_layer_ids)
    rows = tuple(config.engram_num_embeddings)
    if len(layer_ids) != len(rows) or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("Engram table rows must name each layer exactly once")
    if not layer_ids:
        return ConstantInputs(StaticBuffers(), StaticBuffers())
    head_dim = int(config.text_config.engram_head_dim)
    # The tables' own dtypes, which the Engram ops read directly.
    specs = StaticBuffers(
        **{
            f"engram_table_{ordinal}_{part}": TensorType(
                dtype, [count, width], device=DeviceRef.CPU()
            )
            for ordinal, count in enumerate(rows)
            for part, dtype, width in (
                ("weight", DType.float8_e4m3fn, head_dim),
                ("scale", DType.float8_e8m0fnu, head_dim // 32),
            )
        }
    )
    if not is_virtual_device_mode():
        # Reuse the model's host-capacity policy, without its global table cache.
        _check_engram_host_ram(
            "Engram deployment tables",
            sum(rows) * (head_dim + head_dim // 32),
        )
    packed = (
        EngramTableFormat.of(config.pipeline_config.model.huggingface_config)
        is EngramTableFormat.MXFP4
    )
    with F.lazy():
        module = EngramStaticInitializer(layer_ids, rows, head_dim, packed)
    names = {name for name, _ in module.parameters}
    missing = names - state_dict.keys()
    if missing:
        raise ValueError(
            f"Engram initializer is missing weights: {sorted(missing)}"
        )
    initializer_types = tree.map(
        lambda spec: BufferType(spec.dtype, spec.shape, device=spec.device),
        specs,
    )
    initializer = GraphSpec(
        module,
        "engram_static_init",
        ({"constants": initializer_types},),
        weights={name: state_dict[name] for name in names},
    )
    devices = tuple(ref.to_device() for ref in config.devices)
    buffers = allocate_static_buffers(specs, devices)
    if not is_virtual_device_mode():
        logging.getLogger("max.pipelines").info(
            "Engram static resources: buffers=%d requested_pinned_bytes=%d shared_across_ranks=%d",
            len(tree.leaves(buffers)),
            sum(
                buffer.num_elements * buffer.element_size
                for buffer in tree.leaves(buffers)
            ),
            len(devices),
        )
    return ConstantInputs(specs, buffers, initializer, devices)
