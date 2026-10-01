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
"""Weight loading for Nemotron-H checkpoints.

The module tree uses the checkpoint's names, so tensors pass through as
stored; :func:`dequantize_to_bf16` expands the quantized ones to BF16, except
the NVFP4 routed experts that :func:`stack_nvfp4_experts` keeps 4-bit for the
W4A4 grouped matmul.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Collection, Mapping
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.weights._fp8 import e4m3fn_lut
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    e2m1_decode_table,
)
from max.support.math import ceildiv

from .quantization import NVFP4_GROUP_SIZE, ModuleFormat


def convert_nemotron_h_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs: object
) -> dict[str, WeightData]:
    """Returns the tensors Nemotron-H reads, under their own names.

    Upcasts the router weight to float32; the router matmul runs in float32.
    Stores each depthwise conv weight ``[dim, 1, kernel]`` as the
    ``[dim, kernel]`` the conv kernel reads.
    """
    weights: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        # The MTP head is not built, and the FP8 KV-cache scales are not
        # applied.
        if name.startswith("mtp.") or name.endswith(
            (".k_proj.k_scale", ".v_proj.v_scale")
        ):
            continue
        data = value.data()
        if name.endswith(".mixer.gate.weight") and data.dtype != DType.float32:
            data = data.astype(DType.float32)
        elif name.endswith(".mixer.conv1d.weight"):
            dim, _, kernel = (int(d) for d in data.shape)
            data = WeightData(
                data=data.to_buffer().view(data.dtype, (dim, kernel)),
                name=name,
                dtype=data.dtype,
                shape=Shape([dim, kernel]),
            )
        weights[name] = data
    return weights


# Rows per dequantization task. Splitting the large tensors (``lm_head`` has
# 131072 rows) keeps the temporaries small and the workers evenly loaded.
_DEQUANT_ROWS = 2048


def _bytes(weight: WeightData) -> npt.NDArray[np.uint8]:
    return np.from_dlpack(weight.to_buffer().view(DType.uint8)).view(np.uint8)


def _scalar(weight: WeightData) -> np.float32:
    return np.from_dlpack(weight.to_buffer()).astype(np.float32).reshape(-1)[0]


def _nvfp4_rows(
    packed: npt.NDArray[np.uint8],
    block_scales: npt.NDArray[np.uint8],
    global_scale: np.float32,
    rows: slice,
    out: npt.NDArray[np.uint16],
) -> None:
    table = e2m1_decode_table(FP4Format.NVFP4)
    codes = np.arange(256)
    # A packed byte's two E2M1 values, low nibble first.
    pair_lut = np.stack([table[codes & 0xF], table[codes >> 4]], axis=-1)
    row_bytes = packed[rows]
    values = pair_lut[row_bytes].reshape(
        row_bytes.shape[0], -1, NVFP4_GROUP_SIZE
    )
    values *= e4m3fn_lut()[block_scales[rows]][..., None]
    values *= global_scale
    out[rows] = float32_to_bfloat16_as_uint16(
        values.reshape(row_bytes.shape[0], -1)
    )


def _fp8_rows(
    weight: npt.NDArray[np.uint8],
    scale: np.float32,
    rows: slice,
    out: npt.NDArray[np.uint16],
) -> None:
    out[rows] = float32_to_bfloat16_as_uint16(
        e4m3fn_lut()[weight[rows]] * scale
    )


def dequantize_to_bf16(
    state_dict: Mapping[str, WeightData],
    modules: Mapping[str, ModuleFormat],
) -> dict[str, WeightData]:
    """Returns ``state_dict`` with the given quantized modules in BF16.

    Each weight becomes what a weight-only kernel computes with, in float32
    and rounded once to BF16: ``e2m1 * weight_scale * weight_scale_2`` for
    NVFP4, ``fp8 * weight_scale`` for FP8. The modules' scale tensors are
    dropped, the FP8 ``input_scale`` included, since activations stay BF16.

    Args:
        state_dict: The checkpoint's tensors.
        modules: The modules to dequantize, with their stored formats.

    Returns:
        A new state dict; ``state_dict`` is not modified.
    """
    weights = dict(state_dict)
    outputs: dict[str, npt.NDArray[np.uint16]] = {}
    # numpy releases the GIL inside these array operations, so the row tasks
    # run in parallel on threads.
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        tasks: list[Future[None]] = []
        for module, fmt in modules.items():
            weight = _bytes(weights.pop(f"{module}.weight"))
            scale = weights.pop(f"{module}.weight_scale")
            if fmt is ModuleFormat.NVFP4_WEIGHT_ONLY:
                global_scale = _scalar(weights.pop(f"{module}.weight_scale_2"))
                out = np.empty(
                    (weight.shape[0], weight.shape[1] * 2), dtype=np.uint16
                )
                dequantize_rows = functools.partial(
                    _nvfp4_rows, weight, _bytes(scale), global_scale, out=out
                )
            elif fmt is ModuleFormat.FP8_STATIC_TENSOR:
                weights.pop(f"{module}.input_scale")
                out = np.empty(weight.shape, dtype=np.uint16)
                dequantize_rows = functools.partial(
                    _fp8_rows, weight, _scalar(scale), out=out
                )
            else:
                raise ValueError(f"'{module}' is not quantized ({fmt.name})")
            tasks.extend(
                pool.submit(
                    dequantize_rows, slice(start, start + _DEQUANT_ROWS)
                )
                for start in range(0, weight.shape[0], _DEQUANT_ROWS)
            )
            outputs[module] = out
        for task in tasks:
            task.result()
    for module, bits in outputs.items():
        name = f"{module}.weight"
        weights[name] = WeightData(
            data=Buffer.from_numpy(bits).view(DType.bfloat16, bits.shape),
            name=name,
            dtype=DType.bfloat16,
            shape=Shape(bits.shape),
        )
    return weights


_ROUTED_EXPERT = re.compile(
    r"^(?P<mixer>.*\.mixer)\.experts\.(?P<expert>\d+)\.(?P<proj>up_proj|down_proj)$"
)

# The SM100 block-scaled matmul reads its scales in granules of 128 rows by 4
# scale columns.
_SF_GRANULE_ROWS = 128
_SF_ATOM_ROWS = 32
_SF_ATOM_COLS = 4


def interleave_nvfp4_scales(
    scales: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """Permutes one expert's E4M3 block scales into the kernel layout.

    Element ``(r, c)`` lands at ``[r // 128, c // 4, r % 32, (r % 128) // 32,
    c % 4]``, where ``set_scale_factor`` in
    ``max/kernels/src/linalg/fp4_utils.mojo`` stores it. Rows are zero-padded
    to a whole granule.

    Args:
        scales: The ``[N, K / 16]`` scales as bytes. ``K / 16`` must be a
            multiple of 4.

    Returns:
        The ``[ceil(N / 128), K / 64, 32, 4, 4]`` scales.
    """
    rows, cols = scales.shape
    if cols % _SF_ATOM_COLS:
        raise ValueError(f"cannot interleave NVFP4 scales {scales.shape}")
    granules = ceildiv(rows, _SF_GRANULE_ROWS)
    padded = granules * _SF_GRANULE_ROWS
    if padded != rows:
        scales = np.pad(scales, ((0, padded - rows), (0, 0)))
    atoms = scales.reshape(
        granules,
        _SF_GRANULE_ROWS // _SF_ATOM_ROWS,
        _SF_ATOM_ROWS,
        cols // _SF_ATOM_COLS,
        _SF_ATOM_COLS,
    )
    return np.ascontiguousarray(atoms.transpose(0, 3, 2, 1, 4))


def stack_nvfp4_experts(
    state_dict: Mapping[str, WeightData],
    modules: Mapping[str, ModuleFormat],
    mixers: Collection[str],
) -> tuple[dict[str, WeightData], dict[str, ModuleFormat]]:
    """Stacks the NVFP4 routed experts of ``mixers`` for the W4A4 matmul.

    Each mixer's per-expert ``up_proj`` and ``down_proj`` tensors become
    ``{mixer}.up_weight`` (packed E2M1, ``[experts, N, K / 2]``),
    ``{mixer}.up_block_scale`` (E4M3 in the kernel's interleaved layout,
    ``[experts, ceil(N / 128), K / 64, 32, 4, 4]``), ``{mixer}.up_scale``
    (float32 ``weight_scale_2``, ``[experts]``) and the three ``down_``
    tensors. Nothing is dequantized.

    Args:
        state_dict: The checkpoint's tensors.
        modules: The quantized modules and their formats.
        mixers: The MoE mixers to stack (see
            :meth:`NemotronHConfig.w4a4_mixers`).

    Returns:
        A new state dict, and the modules left to dequantize.
    """
    weights = dict(state_dict)
    remaining = dict(modules)
    experts: dict[tuple[str, str], dict[int, str]] = {}
    for module, fmt in modules.items():
        match = _ROUTED_EXPERT.match(module)
        if (
            match is None
            or match["mixer"] not in mixers
            or fmt is not ModuleFormat.NVFP4_WEIGHT_ONLY
        ):
            continue
        key = (match["mixer"], match["proj"])
        experts.setdefault(key, {})[int(match["expert"])] = module
        del remaining[module]

    def load(
        module: str,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], np.float32]:
        codes = _bytes(weights.pop(f"{module}.weight"))
        scales = _bytes(weights.pop(f"{module}.weight_scale"))
        global_scale = _scalar(weights.pop(f"{module}.weight_scale_2"))
        return codes, interleave_nvfp4_scales(scales), global_scale

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        for (mixer, proj), by_expert in experts.items():
            ids = sorted(by_expert)
            if ids != list(range(len(ids))):
                raise ValueError(f"'{mixer}' is missing routed experts")
            loaded = list(pool.map(load, (by_expert[e] for e in ids)))
            prefix = f"{mixer}.{proj.removesuffix('_proj')}"
            for suffix, array, dtype in (
                ("weight", np.stack([c for c, _, _ in loaded]), DType.uint8),
                (
                    "block_scale",
                    np.stack([s for _, s, _ in loaded]),
                    DType.float8_e4m3fn,
                ),
                (
                    "scale",
                    np.array([g for _, _, g in loaded], dtype=np.float32),
                    DType.float32,
                ),
            ):
                name = f"{prefix}_{suffix}"
                weights[name] = WeightData(
                    data=Buffer.from_numpy(array).view(dtype, array.shape),
                    name=name,
                    dtype=dtype,
                    shape=Shape(array.shape),
                )
    return weights, remaining
