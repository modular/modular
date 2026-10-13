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
"""Validates the fused mHC gate op against the PyTorch reference it replaces.

`mo.hyper_connection.gates` fuses the sigmoids, the softmax, and the
Sinkhorn-Knopp projection that a manifold-constrained hyper-connection site
runs between its stream projection and its stream collapse. The reference is a
transcription of that block, run in float32 -- the thing the kernel has to
match.
"""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import hyper_connection_gates
from torch.utils.dlpack import from_dlpack

_HC_MULT = 4
_SINKHORN_ITERS = 20
_EPS = 1e-6
_MIX_WIDTH = 2 * _HC_MULT + _HC_MULT**2


@pytest.fixture(scope="module")
def compile_gates(gpu_session: InferenceSession) -> Model:
    """Compiles the op once for the module."""
    device_ref = DeviceRef.GPU()
    with Graph(
        "hyper_connection_gates",
        input_types=(
            # A symbolic leading dim keeps one compiled graph valid for every
            # row count, so the row sweep below costs no extra compiles.
            TensorType(
                DType.float32, ["total_seq_len", _MIX_WIDTH], device=device_ref
            ),
            TensorType(DType.float32, [_MIX_WIDTH], device=device_ref),
            TensorType(DType.float32, [3], device=device_ref),
        ),
    ) as graph:
        hc_proj, bias, scale = (v.tensor for v in graph.inputs)
        graph.output(
            *hyper_connection_gates(
                hc_proj,
                bias,
                scale,
                hc_mult=_HC_MULT,
                hc_eps=_EPS,
                hc_sinkhorn_iters=_SINKHORN_ITERS,
            )
        )
    return gpu_session.load(graph)


def test_gates_match_torch(compile_gates: Model) -> None:
    hc = _HC_MULT
    # A warp carries two rows at hc_mult=4, four warps to a block, so a block
    # spans eight rows. The small counts leave the tail warp straddling the
    # row bound, which is the case where a lane's row guard stops being
    # warp-uniform.
    for rows in (1, 3, 4, 7, 128, 4096):
        torch.manual_seed(rows)
        hc_proj = torch.randn(
            rows, _MIX_WIDTH, dtype=torch.float32, device="cuda"
        )
        bias = torch.randn(_MIX_WIDTH, dtype=torch.float32, device="cuda")
        scale = 2 * torch.rand(3, dtype=torch.float32, device="cuda")

        pre, post, comb = (
            from_dlpack(t) for t in compile_gates.execute(hc_proj, bias, scale)
        )

        pre_w, post_w, comb_w = hc_proj.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = bias.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = scale.unbind(0)

        ref_pre = torch.sigmoid(pre_w * pre_scale + pre_b) + _EPS
        ref_post = 2 * torch.sigmoid(post_w * post_scale + post_b)

        comb_logits = comb_w.view(
            *comb_w.shape[:-1], hc, hc
        ) * comb_scale + comb_b.view(hc, hc)
        ref = torch.softmax(comb_logits, dim=-1) + _EPS
        ref = ref / (ref.sum(dim=-2, keepdim=True) + _EPS)
        for _ in range(_SINKHORN_ITERS - 1):
            ref = ref / (ref.sum(dim=-1, keepdim=True) + _EPS)
            ref = ref / (ref.sum(dim=-2, keepdim=True) + _EPS)
        ref_comb = ref.flatten(start_dim=-2)

        assert tuple(pre.shape) == (rows, hc)
        assert tuple(post.shape) == (rows, hc)
        assert tuple(comb.shape) == (rows, hc * hc)

        def label(msg: str, rows: int = rows) -> str:
            return f"rows={rows}: {msg}"

        torch.testing.assert_close(
            pre, ref_pre, rtol=1e-5, atol=1e-6, msg=label
        )
        torch.testing.assert_close(
            post, ref_post, rtol=1e-5, atol=1e-6, msg=label
        )
        # The warp butterfly sums (a0+a2)+(a1+a3) where torch sums pairwise,
        # so the Sinkhorn divisions differ in the last bits rather than being
        # identical.
        torch.testing.assert_close(
            comb, ref_comb, rtol=2e-5, atol=1e-6, msg=label
        )


def test_comb_is_doubly_stochastic(compile_gates: Model) -> None:
    """The point of the Sinkhorn projection, checked without reference to
    torch: 20 iterations on a well-conditioned 4x4 leave both margins at 1."""
    torch.manual_seed(13)
    hc_proj = torch.randn(512, _MIX_WIDTH, dtype=torch.float32, device="cuda")
    bias = torch.randn(_MIX_WIDTH, dtype=torch.float32, device="cuda")
    scale = 2 * torch.rand(3, dtype=torch.float32, device="cuda")

    _, _, comb = (
        from_dlpack(t) for t in compile_gates.execute(hc_proj, bias, scale)
    )

    matrix = comb.view(-1, _HC_MULT, _HC_MULT)
    ones = torch.ones(matrix.shape[0], _HC_MULT, device=matrix.device)
    # The loop ends on a column pass, so columns are the tighter of the two.
    torch.testing.assert_close(matrix.sum(dim=-2), ones, rtol=0, atol=1e-4)
    torch.testing.assert_close(matrix.sum(dim=-1), ones, rtol=0, atol=1e-3)
