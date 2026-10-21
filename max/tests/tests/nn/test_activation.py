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
"""Tests for max.nn.activation."""

from __future__ import annotations

from functools import partial

import pytest
from max.graph import ops
from max.nn.activation import GatedActivation, activation_function_from_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("silu", ops.silu),
        ("gelu", ops.gelu),
        ("relu", ops.relu),
        ("tanh", ops.tanh),
        ("sigmoid", ops.sigmoid),
    ],
)
def test_returns_expected_function(name: str, expected: object) -> None:
    assert activation_function_from_name(name) is expected


def test_gelu_tanh() -> None:
    result = activation_function_from_name("gelu_tanh")
    assert isinstance(result, partial)
    assert result.func is ops.gelu
    assert result.keywords == {"approximate": "tanh"}


def test_unrecognized_raises() -> None:
    with pytest.raises(
        ValueError, match="Unrecognized activation function name"
    ):
        activation_function_from_name("nonexistent")


def test_gated_activation_defaults_to_silu() -> None:
    assert GatedActivation().kernel_parameters() == {
        "activation": "silu",
        "swiglu_alpha": 0.0,
        "swiglu_limit": 0.0,
    }


def test_gated_activation_encodes_constants_exactly() -> None:
    # The constants ride as f32 kernel parameters; the float32 rounding is
    # the kernel's, not the encoder's.
    params = GatedActivation(
        "swigluoai", alpha=1.702, limit=7.0
    ).kernel_parameters()
    assert params["activation"] == "swigluoai"
    assert params["swiglu_alpha"] == 1.702
    assert params["swiglu_limit"] == 7.0


def test_gated_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="fused_activation must be one of"):
        GatedActivation("gelu")


def test_gated_activation_swigluoai_needs_constants() -> None:
    with pytest.raises(ValueError, match="non-zero alpha and limit"):
        GatedActivation("swigluoai", alpha=1.702)


@pytest.mark.parametrize(
    "activation",
    [
        GatedActivation("silu", limit=7.0),
        GatedActivation("gelu_tanh"),
    ],
)
def test_gated_activation_accepts_fused_variants(
    activation: GatedActivation,
) -> None:
    params = activation.kernel_parameters()
    assert params["activation"] == activation.fused_activation
    assert float(params["swiglu_limit"]) == activation.limit


@pytest.mark.parametrize(
    "name,alpha,limit,message",
    [
        ("silu", 1.702, 0.0, "silu takes no alpha"),
        ("gelu_tanh", 0.0, 7.0, "gelu_tanh takes no limit"),
        ("silu", 0.0, -1.0, "non-negative"),
    ],
)
def test_gated_activation_rejects_unused_or_bad_constants(
    name: str, alpha: float, limit: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GatedActivation(name, alpha=alpha, limit=limit)


@pytest.mark.parametrize(
    "name,alpha,limit",
    [("swigluoai", 1.702, 7.0), ("silu", 0.0, 7.0), ("gelu_tanh", 0.0, 0.0)],
)
def test_gated_activation_graph_fn_needs_default_fused_fields(
    name: str, alpha: float, limit: float
) -> None:
    with pytest.raises(ValueError, match="replaces the fused activation"):
        GatedActivation(
            name,
            alpha=alpha,
            limit=limit,
            gated_activation_fn=lambda gate_up, moe_dim: gate_up,
        )


def test_gated_activation_graph_fn_cannot_be_fused() -> None:
    activation = GatedActivation(
        gated_activation_fn=lambda gate_up, moe_dim: gate_up
    )
    with pytest.raises(ValueError, match="cannot be fused"):
        activation.kernel_parameters()
