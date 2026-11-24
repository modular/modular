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
"""GLM-5.3-Flash's router runs its bf16 gate on one fused op on SM100 and
on the mixed GEMV on gfx950.

The graphs are traced, not compiled. The kernels' numerics are covered by
``max/kernels/test/gpu/nn/test_moe_sigmoid_gemv_router.mojo``,
``max/kernels/test/gpu/linalg/test_router_gate_mixed_gemv_amd.mojo`` and
``test_router_gate_mixed_gemv_sm100.mojo``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from max import functional as F
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.glm5_next.layers.mlp import (
    Glm5NextTopKRouter,
)
from max.tensor import Tensor, default_device, default_dtype

# GLM-5.3-Flash's gate: 288 experts over a 4096-wide hidden state.
_HIDDEN = 4096
_EXPERTS = 288
_TOP_K = 8

_MIXED_GEMV_OP = "moe.router.gate.mixed.gemv"
_FUSED_ROUTER_OP = "moe.sigmoid.gemv.single.group.router"
_TOPK_OP = "moe.single.group.router"
_ROUTER_MODULE = "max.pipelines.architectures.glm5_next.layers.mlp"


@contextmanager
def _arch(architecture: str) -> Iterator[MagicMock]:
    """Answers the router's accelerator queries as ``architecture`` would."""
    with (
        patch(
            f"{_ROUTER_MODULE}.accelerator_architecture_name",
            return_value=architecture,
        ) as query,
        patch(
            f"{_ROUTER_MODULE}._is_sm10x_gpu",
            return_value=architecture.startswith("sm_10"),
        ),
        # The fused op's shape check reads the warp width (64 on AMD) from the
        # GPU the test runs on, so an AMD runner answers as `architecture` too.
        patch(
            "max.nn._graph_ops._is_amd_gpu",
            return_value=architecture.startswith("gfx"),
        ),
    ):
        yield query


def _router(
    hidden: int = _HIDDEN, n_group: int = 1, experts: int = _EXPERTS
) -> Glm5NextTopKRouter:
    return Glm5NextTopKRouter(
        hidden_dim=hidden,
        num_experts=experts,
        num_experts_per_token=_TOP_K,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=n_group,
        topk_group=n_group,
        norm_topk_prob=True,
        correction_bias_dtype=DType.float32,
    )


def _trace(
    activation_dtype: DType,
    weight_dtype: DType = DType.bfloat16,
    hidden: int = _HIDDEN,
    n_group: int = 1,
    experts: int = _EXPERTS,
) -> str:
    # On the GPU device: building a bf16 weight eagerly is a device op.
    with default_device(Accelerator()), default_dtype(weight_dtype):
        router = _router(hidden, n_group, experts)
        assert router.gate_score.weight.dtype == weight_dtype
        staged = router.trace(
            TensorType(activation_dtype, ["m", hidden], device=DeviceRef.GPU())
        )
    return str(staged)


def test_gfx950_router_runs_the_mixed_gemv_on_bf16_operands() -> None:
    with _arch("gfx950"):
        text = _trace(DType.bfloat16)
    assert _MIXED_GEMV_OP in text
    assert _FUSED_ROUTER_OP not in text
    # Neither operand is widened: the gate GEMV reads both as bf16.
    assert "mo.cast" not in text


def test_sm100_router_runs_one_fused_op_on_bf16_operands() -> None:
    with _arch("sm_100a"):
        text = _trace(DType.bfloat16)
    assert text.count(_FUSED_ROUTER_OP) == 1
    # The fused op replaces the gate GEMV, the sigmoid and the top-k.
    assert _MIXED_GEMV_OP not in text
    assert _TOPK_OP not in text
    assert "mo.sigmoid" not in text
    assert "mo.cast" not in text
    assert "n_experts_per_tok = 8" in text
    assert "n_routed_experts = 288" in text
    assert "norm_weights = true" in text


def test_sm100_router_routes_in_float32() -> None:
    """The fused op scores in float32, as the float32 router does, so the
    routing weights come out in float32."""
    with (
        _arch("sm_100a"),
        F.lazy(),
        default_device(Accelerator()),
        default_dtype(DType.bfloat16),
    ):
        hidden = Tensor.zeros([5, _HIDDEN], dtype=DType.bfloat16)
        topk_idx, topk_weight = _router()(hidden)
        ir = str(Graph.current._mlir_op)
    assert ir.count(_FUSED_ROUTER_OP) == 1
    assert topk_idx.dtype == DType.int32
    assert topk_weight.dtype == DType.float32
    assert list(topk_weight.shape) == [5, _TOP_K]


def test_sm100_router_keeps_the_group_limited_router() -> None:
    """The fused op routes one expert group only."""
    with _arch("sm_100a"):
        text = _trace(DType.bfloat16, n_group=4)
    assert _FUSED_ROUTER_OP not in text
    assert _MIXED_GEMV_OP in text


def test_sm100_router_keeps_the_mixed_gemv_for_a_shape_the_op_refuses() -> None:
    """272 experts do not fill whole warps, so the fused op's shape check
    refuses them and the mixed GEMV computes the logits."""
    with _arch("sm_100a"):
        text = _trace(DType.bfloat16, experts=272)
    assert _FUSED_ROUTER_OP not in text
    assert _MIXED_GEMV_OP in text


@pytest.mark.parametrize(
    ("architecture", "activation_dtype", "weight_dtype", "hidden"),
    [
        ("sm_90a", DType.bfloat16, DType.bfloat16, _HIDDEN),
        ("gfx942", DType.bfloat16, DType.bfloat16, _HIDDEN),
        ("gfx950", DType.float32, DType.bfloat16, _HIDDEN),
        ("gfx950", DType.bfloat16, DType.float32, _HIDDEN),
        ("gfx950", DType.bfloat16, DType.bfloat16, 1536),
        ("sm_100a", DType.float32, DType.bfloat16, _HIDDEN),
        ("sm_100a", DType.bfloat16, DType.float32, _HIDDEN),
    ],
    ids=[
        "sm90",
        "gfx942",
        "fp32_activations",
        "fp32_weight",
        "k_not_a_1024_multiple",
        "sm100_fp32_activations",
        "sm100_fp32_weight",
    ],
)
def test_router_keeps_the_fp32_matmul_off_the_mixed_gemv_path(
    architecture: str,
    activation_dtype: DType,
    weight_dtype: DType,
    hidden: int,
) -> None:
    with _arch(architecture):
        text = _trace(activation_dtype, weight_dtype, hidden)
    assert _MIXED_GEMV_OP not in text
    assert _FUSED_ROUTER_OP not in text
    # The float32 matmul widens whichever operand is bf16.
    assert "mo.cast" in text


def test_router_keeps_the_fp32_matmul_on_a_cpu_device_graph() -> None:
    """The op is registered GPU-only, so a host-device graph must stay on
    the float32 matmul even on a gfx950 host."""
    with _arch("gfx950"), default_device(CPU()), default_dtype(DType.bfloat16):
        router = _router()
        staged = router.trace(
            TensorType(DType.bfloat16, ["m", _HIDDEN], device=DeviceRef.CPU())
        )
    text = str(staged)
    assert _MIXED_GEMV_OP not in text
    assert _FUSED_ROUTER_OP not in text
    assert "mo.cast" in text


def test_router_skips_the_arch_query_without_an_accelerator() -> None:
    """The architecture query raises with no accelerator, so the router
    must not reach it."""
    with (
        patch(f"{_ROUTER_MODULE}.accelerator_count", return_value=0),
        patch(
            f"{_ROUTER_MODULE}.accelerator_architecture_name",
            side_effect=RuntimeError("no accelerator"),
        ),
    ):
        text = _trace(DType.bfloat16)
    assert _MIXED_GEMV_OP not in text
    assert _FUSED_ROUTER_OP not in text


def test_router_gate_weight_stays_bf16() -> None:
    """The checkpoint's bf16 gate is held as stored, not widened."""
    with default_device(Accelerator()), default_dtype(DType.bfloat16):
        router = _router()
        assert router.gate_score.weight.dtype == DType.bfloat16
        assert isinstance(router.e_score_correction_bias, Tensor)
        assert router.e_score_correction_bias.dtype == DType.float32
