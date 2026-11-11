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
"""Tests the routed-rows norm ops in a graph against the norm they fuse."""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from test_common.graph_utils import is_b100_b200
from torch.utils.dlpack import from_dlpack

HIDDEN = 4096
GROUPS = 16
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not is_b100_b200(), reason="the routed-rows finalize runs on B200 only"
)


def _build_graph(top_k: int) -> Graph:
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    with Graph(
        "routed_rows_norm",
        input_types=(
            TensorType(DType.bfloat16, ["rows", HIDDEN], device=gpu),
            TensorType(DType.bfloat16, [HIDDEN], device=gpu),
            TensorType(DType.uint32, [GROUPS + 1], device=gpu),
            TensorType(DType.uint32, ["rows"], device=gpu),
            TensorType(DType.float32, ["tokens", top_k], device=gpu),
        ),
    ) as graph:
        x, w, offsets, restore, router_weight = (v.tensor for v in graph.inputs)
        eps = ops.constant(EPS, DType.float32, device=cpu)

        def finalize(out_dtype: DType) -> TensorValue:
            return ops.custom(
                "norm.routed_rows_rms_norm_finalize",
                device=gpu,
                values=[x, w, offsets, restore, router_weight, eps],
                out_types=[
                    TensorType(out_dtype, [router_weight.shape[0], HIDDEN], gpu)
                ],
            )[0].tensor

        routed = ops.custom(
            "norm.routed_rows_rms_norm",
            device=gpu,
            values=[x, w, offsets, eps],
            out_types=[x.type],
        )[0].tensor
        ones = ops.constant(1.0, DType.bfloat16, device=gpu)
        # The unfused norm of every row: scale by 1 + w in bf16, then a
        # weightless norm.
        whole = ops.rms_norm(
            x * (w + 1.0),
            ops.broadcast_to(ones, [HIDDEN]),
            EPS,
            multiply_before_cast=True,
        )
        graph.output(
            routed, whole, finalize(DType.float32), finalize(DType.bfloat16)
        )
    return graph


@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize(
    "tokens, routed_share", [(1, 1.0), (8, 0.0), (8, 0.45), (64, 1.0)]
)
def test_routed_rows_norm_ops(
    gpu_session: InferenceSession,
    top_k: int,
    tokens: int,
    routed_share: float,
) -> None:
    rows = tokens * top_k
    routed = int(rows * routed_share)
    gen = torch.Generator().manual_seed(tokens + routed + top_k)
    x = (torch.randn(rows, HIDDEN, generator=gen) * 3.0).bfloat16()
    # Rows past the routed count must not be read.
    x[routed:] = float("nan")
    w = (torch.randn(HIDDEN, generator=gen) * 0.1).bfloat16()
    offsets = torch.zeros(GROUPS + 1, dtype=torch.int64)
    offsets[1:] = torch.sort(torch.randint(0, routed + 1, (GROUPS,)))[0]
    offsets[-1] = routed
    restore = torch.randperm(rows, generator=gen)
    router_weight = torch.rand(tokens, top_k, generator=gen)

    compiled = gpu_session.load(_build_graph(top_k))
    out_routed, whole, fused, fused_bf16 = (
        from_dlpack(t).cpu()
        for t in compiled.execute(
            x.cuda(),
            w.cuda(),
            offsets.to(torch.uint32).cuda(),
            restore.to(torch.uint32).cuda(),
            router_weight.cuda(),
        )
    )

    # Every routed row matches the unfused norm bit for bit.
    assert torch.equal(out_routed[:routed], whole[:routed])

    # The finalize against each token's weighted sum of its routed rows'
    # norms; a slot past the routed count adds nothing.
    slots = whole[restore.clamp(max=max(routed - 1, 0))].float()
    slots = slots.reshape(tokens, top_k, HIDDEN)
    slots[(restore >= routed).reshape(tokens, top_k)] = 0.0
    terms = slots * router_weight.unsqueeze(-1)
    expected = terms.sum(1)
    # Its norm sums each row in another order: a term may move by a bf16 ulp.
    bound = (terms.abs() * 2.0**-7).sum(1)
    assert torch.isfinite(fused).all()
    assert ((fused - expected).abs() <= bound).all()
    # The bf16 output is the float32 one rounded once.
    assert torch.equal(fused_bf16, fused.bfloat16())
