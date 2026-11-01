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
"""Checkpoint files shared by writers that do not import torch.

Sidecars and ``config.json`` live here, and so does the safetensors shard
writer. ``quantize_checkpoint`` keeps its own torch writer.
"""

from __future__ import annotations

import json
import shutil
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.tensor import Tensor

_WEIGHT_INDEX = "model.safetensors.index.json"
_SHARD_BYTES = 5 * 1024**3

# One written tensor: dtype, shape, contiguous bytes.
_SavedTensor = tuple[DType, list[int], bytes]

# Names from the safetensors header spec. ``F8_E4M3`` is the name
# ``safetensors`` writes for ``float8_e4m3fn``.
_SAFETENSORS_DTYPE: dict[DType, str] = {
    DType.bool: "BOOL",
    DType.uint8: "U8",
    DType.int8: "I8",
    DType.uint16: "U16",
    DType.int16: "I16",
    DType.uint32: "U32",
    DType.int32: "I32",
    DType.uint64: "U64",
    DType.int64: "I64",
    DType.float16: "F16",
    DType.bfloat16: "BF16",
    DType.float32: "F32",
    DType.float64: "F64",
    DType.float8_e4m3fn: "F8_E4M3",
    DType.float8_e5m2: "F8_E5M2",
}


def copy_auxiliary_files(src: Path, dst: Path) -> None:
    """Copies tokenizer, generation, and other non-weight files."""
    for path in src.iterdir():
        if path.is_dir() or path.suffix == ".safetensors":
            continue
        if path.name in (_WEIGHT_INDEX, "config.json"):
            continue
        shutil.copy2(path, dst / path.name)


def write_quantization_config(
    src: Path, dst: Path, quantization_config: Mapping[str, object]
) -> None:
    """Writes ``dst/config.json`` from ``src``, with ``quantization_config`` set."""
    config_path = src / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config["quantization_config"] = dict(quantization_config)
    (dst / "config.json").write_text(json.dumps(config, indent=2))


def _host_tensor(value: Any) -> _SavedTensor:
    """Returns dtype, shape, and contiguous little-endian bytes."""
    if isinstance(value, Tensor):
        if value.is_distributed:
            value = value.materialize()
        if value.device != CPU():
            value = value.to(CPU())
        buffer: Buffer = value.driver_tensor
        dtype = value.dtype
        shape = [int(dim) for dim in value.shape]
    else:
        buffer = (
            value.to_buffer()
            if hasattr(value, "to_buffer")
            else Buffer.from_dlpack(value)
        )
        if buffer.device != CPU():
            buffer = buffer.to(CPU())
        dtype = value.dtype
        shape = [int(dim) for dim in value.shape]
    count = 1
    for dim in buffer.shape:
        count *= int(dim)
    flat = buffer.view(DType.uint8, [count * buffer.dtype.size_in_bytes])
    raw = np.ascontiguousarray(flat.to_numpy()).tobytes()
    return dtype, shape, raw


def _write_safetensors(
    path: Path, tensors: list[tuple[str, _SavedTensor]]
) -> int:
    """Writes ``tensors`` in the given order. Returns the payload size."""
    header: dict[str, dict[str, Any]] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype, shape, raw) in tensors:
        try:
            dtype_name = _SAFETENSORS_DTYPE[dtype]
        except KeyError:
            raise TypeError(
                f"{name} has dtype {dtype}, which safetensors cannot store"
            ) from None
        header[name] = {
            "dtype": dtype_name,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)
    return offset


def _write_shards(dest: Path, tensors: list[tuple[str, _SavedTensor]]) -> None:
    """Writes ``tensors`` into new safetensors shards and an index."""
    groups: list[list[tuple[str, _SavedTensor]]] = []
    current: list[tuple[str, _SavedTensor]] = []
    size = 0
    for item in tensors:
        nbytes = len(item[1][2])
        if current and size + nbytes > _SHARD_BYTES:
            groups.append(current)
            current = []
            size = 0
        current.append(item)
        size += nbytes
    if current:
        groups.append(current)

    n = len(groups)
    weight_map: dict[str, str] = {}
    total_size = 0
    for index, group in enumerate(groups, start=1):
        filename = (
            "model.safetensors"
            if n == 1
            else f"model-{index:05d}-of-{n:05d}.safetensors"
        )
        total_size += _write_safetensors(dest / filename, group)
        weight_map.update({name: filename for name, _ in group})
    (dest / _WEIGHT_INDEX).write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
    )
