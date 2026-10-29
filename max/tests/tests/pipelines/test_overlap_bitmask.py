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
"""Which stream a speculative graph copies the grammar bitmask on."""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, DeviceRef, DimLike, Graph, TensorType
from max.pipelines.speculative.unified_graph_ops import apply_overlap_bitmask

_SHAPE: list[DimLike] = ["batch_size", 8, 6252]


def _build(side_stream: bool) -> str:
    gpu = DeviceRef.GPU(0)
    with Graph(
        "overlap_bitmask",
        input_types=[
            TensorType(DType.int64, ["batch_size", "num_steps"], device=gpu),
            TensorType(DType.int32, _SHAPE, device=DeviceRef.CPU()),
            BufferType(DType.int64, [2], device=DeviceRef.CPU()),
            BufferType(DType.int32, _SHAPE, device=gpu),
        ],
    ) as graph:
        drafts, pinned, payload, scratch = graph.inputs
        bitmask = apply_overlap_bitmask(
            pinned.tensor,
            payload.buffer,
            scratch.buffer,
            num_steps=drafts.tensor.shape[1],
            device=gpu,
            side_stream=side_stream,
        )
        assert bitmask is not None
        graph.output(bitmask)
    return str(graph)


def test_inline_copy_stays_on_the_model_stream() -> None:
    assert "mo.sequence" not in _build(side_stream=False)


def test_side_stream_copy_runs_in_its_own_region() -> None:
    ir = _build(side_stream=True)
    region = ir.index("mo.sequence[2]")
    wait = ir.index('"mo.wait_host_value_with_dep"')
    copy = ir.index('"mo.inplace_memcpy"')
    end = ir.index("mo.yield")
    # The copy follows the host wait, and both sit inside the region.
    assert region < wait < copy < end
