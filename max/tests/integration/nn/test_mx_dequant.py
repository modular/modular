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
"""Argument-validation tests for the MXFP4 / MXFP8 dequant ops.

These only build graphs, so they run on CPU. Execution is covered on a GPU by
``test_mx_dequant_gpu.py``: the graph compiler does not support
``float8_e8m0fnu`` tensors on a CPU device, so the MX ops cannot be executed
there.
"""

from __future__ import annotations

import pytest
import shared_mx_dequant_impl as impl
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy.kernels import mxfp4_dequant, mxfp8_dequant

FORMATS: list[impl.Format] = ["mxfp4", "mxfp8"]


# --------------------------------------------------------------------------- #
# Argument validation: the kernel reads scales on trust, so what can be known
# when the graph is built is checked there.
# --------------------------------------------------------------------------- #
def _build(
    fmt: impl.Format,
    data_shape: tuple[int, ...],
    scales_shape: tuple[int, ...],
    *,
    data_dtype: DType | None = None,
    scales_dtype: DType = DType.float8_e8m0fnu,
    out_type: DType = DType.bfloat16,
) -> None:
    cpu = DeviceRef.CPU()
    op = mxfp4_dequant if fmt == "mxfp4" else mxfp8_dequant
    with Graph(
        "validate",
        input_types=[
            TensorType(
                data_dtype or impl.data_type(fmt), data_shape, device=cpu
            ),
            TensorType(scales_dtype, scales_shape, device=cpu),
        ],
    ) as graph:
        data, scales = (value.tensor for value in graph.inputs)
        graph.output(op(data, scales, out_type=out_type))


def _shape(fmt: impl.Format, rows: int, k: int) -> tuple[int, int]:
    """Stored data shape for rows x k unpacked elements."""
    return (rows, k // 2 if fmt == "mxfp4" else k)


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_non_e8m0_scales(fmt: impl.Format) -> None:
    with pytest.raises(ValueError, match="float8_e8m0fnu"):
        _build(fmt, _shape(fmt, 4, 64), (4, 2), scales_dtype=DType.uint8)


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_wrong_data_dtype(fmt: impl.Format) -> None:
    with pytest.raises(ValueError, match="must be"):
        _build(fmt, _shape(fmt, 4, 64), (4, 2), data_dtype=DType.int8)


def test_mxfp8_rejects_fp8_output() -> None:
    with pytest.raises(ValueError, match="out_type"):
        _build(
            "mxfp8",
            (4, 64),
            (4, 2),
            out_type=DType.float8_e4m3fn,
        )


def test_mxfp4_rejects_fp8_output_with_interleaved_scales() -> None:
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        _build(
            "mxfp4",
            (4, 32),
            (1, 1, 32, 4, 4),
            out_type=DType.float8_e4m3fn,
        )


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_k_not_a_multiple_of_32(fmt: impl.Format) -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        _build(fmt, _shape(fmt, 4, 48), (4, 1))


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_wrong_scale_columns(fmt: impl.Format) -> None:
    with pytest.raises(ValueError, match="do not match"):
        _build(fmt, _shape(fmt, 4, 64), (4, 3))


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_wrong_scale_rows(fmt: impl.Format) -> None:
    with pytest.raises(ValueError, match="rows"):
        _build(fmt, _shape(fmt, 4, 64), (5, 2))


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_interleaved_scales_for_rank3_weights(
    fmt: impl.Format,
) -> None:
    k = 64
    shape = (2, 4, k // 2 if fmt == "mxfp4" else k)
    with pytest.raises(ValueError, match="rank 2"):
        _build(fmt, shape, (1, 1, 32, 4, 4))


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_interleaved_scales_with_wrong_trailing_dims(
    fmt: impl.Format,
) -> None:
    with pytest.raises(ValueError, match=r"\[32, 4, 4\]"):
        _build(fmt, _shape(fmt, 4, 64), (1, 1, 16, 4, 4))


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_interleaved_scales_with_wrong_k_groups(
    fmt: impl.Format,
) -> None:
    # K = 256 spans two 128-column atoms; the NVFP4-style ceildiv(K, 64) = 4
    # would be wrong for 32-element blocks.
    with pytest.raises(ValueError, match="ceildiv"):
        _build(fmt, _shape(fmt, 4, 256), (1, 4, 32, 4, 4))


@pytest.mark.parametrize("fmt", FORMATS)
def test_rejects_interleaved_scales_with_wrong_leading_dims(
    fmt: impl.Format,
) -> None:
    # 130 rows need ceildiv(130, 128) = 2 tiles, not 1.
    with pytest.raises(ValueError, match="rows"):
        _build(fmt, _shape(fmt, 130, 128), (1, 1, 32, 4, 4))
