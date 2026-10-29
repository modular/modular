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
"""Argument-validation tests for ``nvfp4_dequant``.

These only build graphs, so they run on CPU. Execution is covered on a GPU by
``test_nvfp4_dequant_gpu.py``: the graph compiler does not support FP8 tensors
(``float8_e4m3fn`` scales) on a CPU device, so the op cannot be executed there.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy.kernels import nvfp4_dequant


# --------------------------------------------------------------------------- #
# Argument validation: the kernel reads scales on trust, so what can be known
# when the graph is built is checked there.
# --------------------------------------------------------------------------- #
def _build(
    packed_shape: tuple[int, ...],
    scales_shape: tuple[int, ...],
    *,
    packed_dtype: DType = DType.uint8,
    scales_dtype: DType = DType.float8_e4m3fn,
    out_type: DType = DType.bfloat16,
) -> None:
    cpu = DeviceRef.CPU()
    with Graph(
        "validate",
        input_types=[
            TensorType(packed_dtype, packed_shape, device=cpu),
            TensorType(scales_dtype, scales_shape, device=cpu),
        ],
    ) as graph:
        packed, scales = (value.tensor for value in graph.inputs)
        graph.output(nvfp4_dequant(packed, scales, 1.0, out_type=out_type))


def test_rejects_non_e4m3_scales() -> None:
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        _build((4, 32), (4, 4), scales_dtype=DType.uint8)


def test_rejects_non_uint8_weights() -> None:
    with pytest.raises(ValueError, match="uint8"):
        _build((4, 32), (4, 4), packed_dtype=DType.int8)


def test_rejects_unsupported_output_type() -> None:
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        _build((4, 32), (4, 4), out_type=DType.float16)


def test_rejects_k_not_a_multiple_of_16() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        _build((4, 12), (4, 2))


def test_rejects_wrong_scale_columns() -> None:
    with pytest.raises(ValueError, match="do not match"):
        _build((4, 32), (4, 3))


def test_rejects_wrong_scale_rows() -> None:
    with pytest.raises(ValueError, match="row count"):
        _build((4, 32), (5, 4))


def test_rejects_interleaved_scales_for_rank3_weights() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        _build((2, 4, 32), (1, 1, 32, 4, 4))


def test_rejects_interleaved_scales_with_wrong_trailing_dims() -> None:
    with pytest.raises(ValueError, match=r"\[32, 4, 4\]"):
        _build((4, 32), (1, 1, 16, 4, 4))


def test_rejects_interleaved_scales_with_wrong_leading_dims() -> None:
    # 130 rows need ceildiv(130, 128) = 2 tiles, not 1.
    with pytest.raises(ValueError, match="ceildiv"):
        _build((130, 32), (1, 1, 32, 4, 4))
