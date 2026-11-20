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
"""Graph wiring of the iHC pre site op, built without compiling."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import ihc_pre, ihc_pre_supports

HC, HIDDEN = 4, 6144


def _build(
    num_mix: int,
    streams_dtype: DType = DType.bfloat16,
    fn_dtype: DType = DType.float32,
    hidden: int = HIDDEN,
) -> Graph:
    gpu = DeviceRef.GPU()
    types = [
        TensorType(streams_dtype, ["rows", HC, hidden], gpu),
        TensorType(fn_dtype, [num_mix, HC * hidden], gpu),
        TensorType(DType.float32, [num_mix], gpu),
        TensorType(DType.float32, [max(num_mix // HC, 1)], gpu),
        TensorType(DType.bfloat16, [hidden], gpu),
    ]
    with Graph("ihc", input_types=types) as graph:
        streams, fn, base, scale, gamma = (i.tensor for i in graph.inputs)
        normed, gates = ihc_pre(
            streams,
            fn,
            base,
            scale,
            gamma,
            hc_magnitude=2.0,
            hc_eps=1e-6,
            rms_eps=1e-5,
            norm_eps=1e-5,
        )
        assert normed.dtype == DType.bfloat16
        assert [str(d) for d in normed.shape] == ["rows", str(hidden)]
        assert gates.dtype == DType.float32
        assert [str(d) for d in gates.shape] == ["rows", str(num_mix)]
        graph.output(normed, gates)
    return graph


@pytest.mark.parametrize("num_mix", [HC, 2 * HC])
def test_stages_both_ops(num_mix: int) -> None:
    ir = str(_build(num_mix))
    assert "hyper_connection.ihc.mix_gemv" in ir
    assert "hyper_connection.ihc.collapse_rms_norm" in ir
    # 24576 columns split 24 ways, `num_mix` logits and the sum of squares.
    assert f"[24, rows, {num_mix + 1}]" in ir


@pytest.mark.parametrize(
    ("num_mix", "streams_dtype", "fn_dtype", "hidden", "match"),
    [
        (2 * HC, DType.float32, DType.float32, HIDDEN, "bfloat16"),
        (2 * HC, DType.bfloat16, DType.bfloat16, HIDDEN, "float32"),
        (4 * HC, DType.bfloat16, DType.float32, HIDDEN, "gate rows"),
        (2 * HC, DType.bfloat16, DType.float32, 20, "gate rows"),
    ],
)
def test_rejects_unsupported_inputs(
    num_mix: int,
    streams_dtype: DType,
    fn_dtype: DType,
    hidden: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _build(num_mix, streams_dtype, fn_dtype, hidden)


def test_supports_only_cuda_and_kernel_shapes() -> None:
    assert not ihc_pre_supports(
        HC, HIDDEN, 2 * HC, DType.bfloat16, DeviceRef.CPU()
    )
    for hc_mult, hidden, num_mix, dtype in [
        (HC, HIDDEN, 2 * HC, DType.float32),
        (HC, HIDDEN, 3, DType.bfloat16),
        (HC, HIDDEN, 4 * HC, DType.bfloat16),
        (HC, 20, 2 * HC, DType.bfloat16),
        (3, 128, 6, DType.bfloat16),
        (6, 128, 6, DType.bfloat16),
    ]:
        assert not ihc_pre_supports(
            hc_mult, hidden, num_mix, dtype, DeviceRef.GPU()
        )
