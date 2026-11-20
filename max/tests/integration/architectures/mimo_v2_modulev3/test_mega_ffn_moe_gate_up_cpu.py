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
"""The gate/up pairing MiMo's sigma staging hands the experts, decoded.

Every other gate of the fused MegaFFN holds it to the two-launch chain, and both
arms consume the same stacked weight tensor. So a stack laid out with up where
gate belongs -- MiMo's own ``_stacked_weights`` permutation, or the reshape that
feeds it -- would move both arms together: the two would stay bit-identical
while the model served ``silu(up) * gate``. The kernel's read order is gated
separately, against the chain, which pairs even columns with gate
(``test_megaffn_mimo_gpu.py``).

This test breaks the symmetry on the staging side. Each expert's gate rows carry
E2M1 1.0 at the contraction's first byte and its up rows 4.0, with expert 0
carrying 0.5 and 1.5 instead, so every output-row pair has one right answer and
the codes name which projection sits in which row. The staged stack is then
decoded row by row: pair ``(2i, 2i+1)`` must be (gate row i, up row i) for each
expert, and a swap, an offset, or an expert-order flip fails.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from expert_stacks import realize
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import MiMoV2MoE
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh

HIDDEN, MOE_DIM, EXPERTS = 4096, 2048, 2
"""MiMo-V2.6-Flash's expert geometry on one host mesh: no sharding, so the
whole stack is one device's and the pairing is readable at every row."""

GATE_CODE, UP_CODE = 0x22, 0x66
"""E2M1 in both nibbles of a byte: 0x22 decodes to 1.0, 0x66 to 4.0. Both
nibbles carry the same value, so the pairing does not depend on which K element
of a byte a code belongs to."""

DECOY_GATE, DECOY_UP = 0x11, 0x33
"""Expert 0's 0.5 and 1.5, so an expert-order flip shows as well."""

PATTERNS = ((DECOY_GATE, DECOY_UP), (GATE_CODE, UP_CODE))

Array = npt.NDArray[np.uint8]


def _codes(rows: int, cols: int, byte: int) -> Array:
    """A packed-E2M1 weight with `byte` in column 0 of every row."""
    out = np.zeros([rows, cols], dtype=np.uint8)
    out[:, 0] = np.uint8(byte)
    return out


def _stack(monkeypatch: pytest.MonkeyPatch) -> Array:
    """Stages the experts on a host mesh and returns the gate/up stack."""
    # The E8M0 zeros the experts start with would compile eagerly on a host
    # device, which has no such type; traced, they never reach one.
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = MiMoV2MoE(
            hidden_dim=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=8,
            moe_dim=MOE_DIM,
            norm_topk_prob=True,
            quant_config=QuantConfig.mxfp4(
                mlp_quantized_layers={1},
                embedding_output_dtype=DType.bfloat16,
            ),
            route_on_every_device=True,
        )
    for expert, (gate, up) in zip(layer.experts, PATTERNS, strict=True):
        expert.gate_proj.weight = Tensor.from_dlpack(
            _codes(MOE_DIM, HIDDEN // 2, gate)
        )
        expert.up_proj.weight = Tensor.from_dlpack(
            _codes(MOE_DIM, HIDDEN // 2, up)
        )
        expert.down_proj.weight = Tensor.from_dlpack(
            _codes(HIDDEN, MOE_DIM // 2, GATE_CODE)
        )
        # E8M0 has no host backend, so uint8 codes stand in for the scales:
        # the stacking only moves bytes, and the pairing under test is the
        # weights'.
        for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
            projection.weight_scale = Tensor.from_dlpack(
                np.zeros(
                    [int(dim) for dim in projection.weight_scale.shape],
                    np.uint8,
                )
            )
    # The scale interleave is an E8M0 device op, which the host backend refuses
    # to compile; the pairing under test is the weights', and the scales ride
    # the same permutation.
    monkeypatch.setattr(
        layer.strategy,
        "prepare_weight_scales",
        lambda gate_up, down, device: (gate_up, down),
    )
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    return realize(layer, mesh)[0][0]


def test_each_pair_is_its_own_experts_gate_then_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output row ``2i`` carries gate row ``i``, row ``2i+1`` its up row.

    One staging covers every expert: realizing the stack is an eager CPU
    compile, so the experts share it rather than each paying for one.
    """
    stack = _stack(monkeypatch)
    assert stack.shape == (EXPERTS, 2 * MOE_DIM, HIDDEN // 2), stack.shape
    for expert, (gate, up) in enumerate(PATTERNS):
        rows = stack[expert, :, 0]
        np.testing.assert_array_equal(
            rows[0::2],
            np.full(MOE_DIM, gate, np.uint8),
            err_msg=f"expert {expert}'s gate rows are not in the even columns",
        )
        np.testing.assert_array_equal(
            rows[1::2],
            np.full(MOE_DIM, up, np.uint8),
            err_msg=f"expert {expert}'s up rows are not in the odd columns",
        )
