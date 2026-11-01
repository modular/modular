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
"""Mixed bf16-A x fp32-B router-gate GEMV op (KERN-3098).

Covers the graph custom op (`mo.router.gate.mixed.gemv`) shape/dtype contract
and numerics, including the tiny-M GEMV and large-M matmul fallback paths.
"""

from __future__ import annotations

import pytest
import torch
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.kernels import (
    _router_gate_mixed_gemv,
    moe_router_group_limited,
)
from torch.utils.dlpack import from_dlpack

# Router-gate shape (KERN-3219): N experts, K hidden.
N = 128
K = 6144


def _build_op_graph(m: int) -> Graph:
    with Graph(
        "router_gate_mixed_gemv",
        input_types=(
            TensorType(DType.bfloat16, [m, K], device=DeviceRef.GPU()),
            TensorType(DType.float32, [N, K], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(_router_gate_mixed_gemv(a, b))
    return graph


# m<=16 = the original mixed-GEMV band, 17-64 = the widened band (formerly
# fallback), 65-128 = the SM100 window (fallback on MI355X), 512 = the large-M
# fallback on both.
@pytest.mark.parametrize("m", [2, 16, 17, 24, 32, 64, 96, 128, 129, 512])
def test_router_gate_mixed_gemv_op(
    gpu_session: InferenceSession, m: int
) -> None:
    """The op returns fp32 [M, N] and is numerically equal to cast-A @ B.T on
    both the tiny-M fused route and the large-M cast + fp32-matmul fallback."""
    torch.manual_seed(0)
    a = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float32, device="cuda")

    compiled = gpu_session.load(_build_op_graph(m))
    got = from_dlpack(compiled.execute(a, b)[0])

    assert tuple(got.shape) == (m, N)
    assert got.dtype == torch.float32

    # Reference: widen bf16 A to fp32 (lossless) then fp32 GEMV — exactly the
    # cast + fp32-GEMV chain the fused op replaces.
    ref = a.to(torch.float32) @ b.t()
    assert torch.equal(got.argmax(dim=1), ref.argmax(dim=1))
    # The large-M fallback on NVIDIA is the SM100 tile GEMM, which truncates
    # the fp32 weight to TF32; the fused GEMV (IEEE fp32) is held to 1e-2.
    atol = 0.25 if torch.version.hip is None and m > 128 else 1e-2
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=atol)


@pytest.mark.skipif(
    torch.version.hip is not None, reason="SM100 tiling is pinned to NVIDIA"
)
def test_router_gate_mixed_gemv_matches_standard_path(
    gpu_session: InferenceSession,
) -> None:
    """Up to M = 64 the fused op is bit-identical to the router's standard
    chain (cast the activation, then the fp32 matmul dispatch), so expert
    selection cannot change. One graph with a runtime M, as in the model."""
    with Graph(
        "router_gate_mixed_gemv_vs_standard",
        input_types=(
            TensorType(DType.bfloat16, ["m", K], device=DeviceRef.GPU()),
            TensorType(DType.float32, [N, K], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        standard = ops.matmul(a.cast(DType.float32), ops.transpose(b, 0, 1))
        graph.output(_router_gate_mixed_gemv(a, b), standard)
    compiled = gpu_session.load(graph)

    torch.manual_seed(1)
    b_t = torch.randn(N, K, dtype=torch.float32, device="cuda")
    for m in [1, 5, 8, 9, 17, 25, 33, 36, 48, 64]:
        a_t = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
        fused, standard_out = compiled.execute(a_t, b_t)
        assert torch.equal(from_dlpack(fused), from_dlpack(standard_out)), m


@pytest.mark.skipif(
    torch.version.hip is not None,
    reason="SM100 extends the fused path to M=128",
)
def test_router_gate_mixed_gemv_corrected_top4(
    gpu_session: InferenceSession,
) -> None:
    """Preserves corrected top-4 routing against an fp64 projection reference."""
    with Graph(
        "router_gate_mixed_gemv_corrected_top4",
        input_types=(
            TensorType(DType.bfloat16, ["m", K], device=DeviceRef.GPU()),
            TensorType(DType.float32, [N, K], device=DeviceRef.GPU()),
            TensorType(DType.float32, [N], device=DeviceRef.GPU()),
            TensorType(DType.float32, ["m", N], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a, b, bias, reference = (v.tensor for v in graph.inputs)
        outputs: list[TensorValue] = []
        for logits in (_router_gate_mixed_gemv(a, b), reference):
            indices, weights = moe_router_group_limited(
                ops.sigmoid(logits),
                bias,
                n_routed_experts=N,
                n_experts_per_tok=4,
                n_groups=1,
                topk_group=1,
                norm_weights=True,
                routed_scaling_factor=1.0,
            )
            outputs.extend((indices, weights))
        graph.output(*outputs)
    compiled = gpu_session.load(graph)

    torch.manual_seed(7)
    for near_cutoff in (False, True):
        b_host = torch.randn(N, K) * K**-0.5
        bias_host = 0.05 * torch.randn(N)
        if near_cutoff:
            b_host.zero_()
            b_host[:, 0] = -2.0
            b_host[:6, 0] = torch.tensor(
                [1.5, 1.0, 0.75, 0.25006, 0.25002, 2.0]
            )
            bias_host.zero_()
            bias_host[:6] = torch.tensor(
                [0.01, 0.02, 0.03, 0.03 - 4e-6, 0.03 + 4e-6, -1.0]
            )
        b_device, bias_device = b_host.cuda(), bias_host.cuda()
        for m in (96, 128):
            a_host = torch.randn(m, K, dtype=torch.bfloat16)
            if near_cutoff:
                a_host.zero_()
                a_host[:, 0] = 1.0
            reference = (a_host.double() @ b_host.double().t()).float()
            if near_cutoff:
                # Both cutoff logits round to the same TF32 value. The bias
                # then reverses their order, while raw top-1 stays expert 5.
                corrected = torch.sigmoid(reference) + bias_host
                best = corrected.topk(5, dim=-1)
                gap = best.values[:, 3] - best.values[:, 4]
                assert torch.all((gap > 0) & (gap < 1e-5))
                assert torch.equal(
                    best.indices[:, :4], torch.arange(4).expand(m, 4)
                )
                assert torch.all(reference.argmax(dim=-1) == 5)
            a_device, reference_device = a_host.cuda(), reference.cuda()
            got_idx, got_weight, ref_idx, ref_weight = (
                from_dlpack(t)
                for t in compiled.execute(
                    a_device, b_device, bias_device, reference_device
                )
            )
            got_order = got_idx.argsort(dim=-1)
            ref_order = ref_idx.argsort(dim=-1)
            expected_idx = (
                (torch.sigmoid(reference) + bias_host).topk(4).indices
            )
            torch.testing.assert_close(
                ref_idx.gather(-1, ref_order).long(),
                expected_idx.sort(dim=-1).values.cuda(),
            )
            torch.testing.assert_close(
                got_idx.gather(-1, got_order), ref_idx.gather(-1, ref_order)
            )
            torch.testing.assert_close(
                got_weight.gather(-1, got_order),
                ref_weight.gather(-1, ref_order),
                rtol=1e-5,
                atol=1e-6,
            )


def test_router_gate_mixed_gemv_validation() -> None:
    """The wrapper rejects wrong dtypes and dynamic N/K at graph build."""
    with Graph(
        "router_gate_mixed_gemv_validation",
        input_types=(
            TensorType(DType.float32, [8, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [N, K], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [8, K], device=DeviceRef.GPU()),
            TensorType(DType.float32, ["N", "K"], device=DeviceRef.GPU()),
        ),
    ) as graph:
        a_f32, b_bf16, a_bf16, b_dyn = (v.tensor for v in graph.inputs)
        # fp32 activation rejected (must be bf16).
        try:
            _router_gate_mixed_gemv(a_f32, b_bf16)
        except ValueError:
            pass
        else:
            raise AssertionError("fp32 hidden states must be rejected")
        # bf16 weight rejected (must be fp32).
        try:
            _router_gate_mixed_gemv(a_bf16, b_bf16)
        except ValueError:
            pass
        else:
            raise AssertionError("bf16 gate weights must be rejected")
        # Dynamic N/K rejected (must be static).
        try:
            _router_gate_mixed_gemv(a_bf16, b_dyn)
        except ValueError:
            pass
        else:
            raise AssertionError("dynamic gate shapes must be rejected")
