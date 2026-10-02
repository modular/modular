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
"""GLM's clamped SwiGLU clamps *before* the activation.

The inputs deliberately straddle the limit. Below it every candidate ordering
agrees, which is why a random-weight smoke test cannot see this and a real
checkpoint can.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.moe.moe import ClampedSwiGLU
from max.pipelines.architectures.glm5_next.layers.clamped_swiglu import (
    clamped_swiglu,
)

LIMIT = 10.0

# Straddles the limit on both sides and on both halves. The +/-20 entries are
# the only ones that separate the orderings.
GATE = np.array(
    [[-20.0, -3.0, 0.0, 3.0, 9.9, 10.0, 12.0, 20.0]], dtype=np.float32
)
UP = np.array(
    [[20.0, 3.0, 0.0, -3.0, -9.9, -10.0, -12.0, -20.0]], dtype=np.float32
)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def glm_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(min(gate, limit)) * clamp(up, -limit, limit)``.

    Transcribed from the reference implementation (transformers PR 48342 at
    ``f57a815``), where both the MoE experts and the vision tower compute::

        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)
    """
    return _silu(torch.clamp(gate, max=LIMIT)) * torch.clamp(
        up, min=-LIMIT, max=LIMIT
    )


def activation_then_clamp(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """What ``linear.py`` and ``make_concatenated_gated_activation_fn`` do."""
    return torch.clamp(_silu(gate), max=LIMIT) * torch.clamp(
        up, min=-LIMIT, max=LIMIT
    )


def gpt_oss_variant(
    gate: torch.Tensor, up: torch.Tensor, alpha: float = 1.702
) -> torch.Tensor:
    """What ``_swigluoai_activation`` does: an ``alpha`` and a ``+1`` bias."""
    gate_c = torch.clamp(gate, max=LIMIT)
    up_c = torch.clamp(up, min=-LIMIT, max=LIMIT)
    return gate_c * torch.sigmoid(alpha * gate_c) * (up_c + 1.0)


@pytest.fixture
def halves() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(GATE), torch.from_numpy(UP)


def test_clamping_before_the_activation_is_not_cosmetic(
    halves: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The two orderings disagree above the limit, and agree below it.

    This is the whole hazard: a test whose inputs stay inside +/-10 passes
    under either ordering.
    """
    gate, up = halves
    ours = glm_reference(gate, up)
    theirs = activation_then_clamp(gate, up)

    straddles = (gate.abs() > LIMIT) | (up.abs() > LIMIT)
    assert straddles.any(), "fixture must exercise values past the limit"

    # At gate = 20: silu(min(20, 10)) = 9.9995, but min(silu(20), 10) = 10.
    assert not torch.allclose(ours, theirs), (
        "activation-then-clamp must differ from clamp-then-activation"
    )
    inside = ~straddles
    assert torch.allclose(ours[inside], theirs[inside], atol=1e-6), (
        "below the limit the orderings must agree, or the fixture is wrong"
    )


def test_gate_saturates_just_below_the_limit(
    halves: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """``silu(min(gate, 10))`` approaches 9.9995, never reaching 10."""
    gate, up = halves
    saturated = glm_reference(gate, torch.ones_like(up))
    expected = _silu(torch.tensor(LIMIT))
    for column in (5, 6, 7):  # gate = 10.0, 12.0, 20.0 all clamp to 10.0
        assert torch.allclose(saturated[0, column], expected, atol=1e-5)
    assert float(expected) < LIMIT, (
        "silu(10) must stay under 10; if it did not, the orderings would be "
        "indistinguishable at saturation and this whole test would be vacuous"
    )


def test_gpt_oss_variant_differs(
    halves: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The other wrong path differs even *below* the limit, via the +1 bias."""
    gate, up = halves
    assert not torch.allclose(
        glm_reference(gate, up), gpt_oss_variant(gate, up)
    )


def test_max_implementation_matches_the_reference(
    halves: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """MAX's ``clamped_swiglu`` graph op against the transcribed reference."""
    gate, up = halves
    device = DeviceRef.CPU()
    spec = TensorType(DType.float32, shape=GATE.shape, device=device)
    with Graph("clamped_swiglu", input_types=[spec, spec]) as graph:
        g, u = graph.inputs
        graph.output(clamped_swiglu(g.tensor, u.tensor, LIMIT))

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    got = model.execute(Buffer.from_dlpack(GATE), Buffer.from_dlpack(UP))[0]
    np.testing.assert_allclose(
        np.from_dlpack(got),
        glm_reference(gate, up).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_glm_clamped_swiglu_matches_the_reference(
    halves: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """``ClampedSwiGLU`` (the MoE's declarative activation marker) against
    the transcribed reference, on the concatenated ``[gate | up]`` layout the
    non-fused MoE paths call it with (`moe_fp8.py`'s ``_expert_matmuls`` and
    ``_local_ep_compute`` fallback branch, `moe.py`'s base-class paths).

    This is the object `Glm5NextMoE` now sets as ``gated_activation_fn``
    instead of the ad hoc closure this module used to build; it must agree
    with the same oracle, including past the clamp bound.
    """
    gate, up = halves
    moe_dim = GATE.shape[1]
    gate_up = torch.cat([gate, up], dim=1)

    activation = ClampedSwiGLU(LIMIT)
    device = DeviceRef.CPU()
    spec = TensorType(
        DType.float32, shape=(gate_up.shape[0], gate_up.shape[1]), device=device
    )
    with Graph("glm_clamped_swiglu", input_types=[spec]) as graph:
        (x,) = graph.inputs
        graph.output(activation(x.tensor, moe_dim))

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    got = model.execute(Buffer.from_dlpack(gate_up.numpy()))[0]
    np.testing.assert_allclose(
        np.from_dlpack(got),
        glm_reference(gate, up).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
