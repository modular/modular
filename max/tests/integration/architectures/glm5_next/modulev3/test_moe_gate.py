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
"""GLM-5.3-Flash's router gate takes the bf16 mixed GEMV on gfx950 only.

The graphs are traced, not compiled. The GEMV's numerics are covered by
``max/kernels/test/gpu/linalg/test_router_gate_mixed_gemv_amd.mojo``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.glm5_next_modulev3.layers.mlp import (
    Glm5NextTopKRouter,
)
from max.tensor import Tensor, default_device, default_dtype

# GLM-5.3-Flash's gate: 288 experts over a 4096-wide hidden state.
_HIDDEN = 4096
_EXPERTS = 288
_TOP_K = 8

_MIXED_GEMV_OP = "moe.router.gate.mixed.gemv"
_ROUTER_MODULE = "max.pipelines.architectures.glm5_next_modulev3.layers.mlp"


@contextmanager
def _arch(architecture: str) -> Iterator[MagicMock]:
    """Answers the router's accelerator query as ``architecture`` would."""
    with patch(
        f"{_ROUTER_MODULE}.accelerator_architecture_name",
        return_value=architecture,
    ) as query:
        yield query


def _router(hidden: int = _HIDDEN) -> Glm5NextTopKRouter:
    return Glm5NextTopKRouter(
        hidden_dim=hidden,
        num_experts=_EXPERTS,
        num_experts_per_token=_TOP_K,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        correction_bias_dtype=DType.float32,
    )


def _trace(
    activation_dtype: DType,
    weight_dtype: DType = DType.bfloat16,
    hidden: int = _HIDDEN,
) -> str:
    # On the GPU device: building a bf16 weight eagerly is a device op.
    with default_device(Accelerator()), default_dtype(weight_dtype):
        router = _router(hidden)
        assert router.gate_score.weight.dtype == weight_dtype
        staged = router.trace(
            TensorType(activation_dtype, ["m", hidden], device=DeviceRef.GPU())
        )
    return str(staged)


def test_gfx950_router_runs_the_mixed_gemv_on_bf16_operands() -> None:
    with _arch("gfx950"):
        text = _trace(DType.bfloat16)
    assert _MIXED_GEMV_OP in text
    # Neither operand is widened: the gate GEMV reads both as bf16.
    assert "mo.cast" not in text


@pytest.mark.parametrize(
    ("architecture", "activation_dtype", "weight_dtype", "hidden"),
    [
        ("sm_100a", DType.bfloat16, DType.bfloat16, _HIDDEN),
        ("gfx942", DType.bfloat16, DType.bfloat16, _HIDDEN),
        ("gfx950", DType.float32, DType.bfloat16, _HIDDEN),
        ("gfx950", DType.bfloat16, DType.float32, _HIDDEN),
        ("gfx950", DType.bfloat16, DType.bfloat16, 1536),
    ],
    ids=[
        "sm100",
        "gfx942",
        "fp32_activations",
        "fp32_weight",
        "k_not_a_1024_multiple",
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


def test_router_gate_weight_stays_bf16() -> None:
    """The checkpoint's bf16 gate is held as stored, not widened."""
    with default_device(Accelerator()), default_dtype(DType.bfloat16):
        router = _router()
        assert router.gate_score.weight.dtype == DType.bfloat16
        assert isinstance(router.e_score_correction_bias, Tensor)
        assert router.e_score_correction_bias.dtype == DType.float32
