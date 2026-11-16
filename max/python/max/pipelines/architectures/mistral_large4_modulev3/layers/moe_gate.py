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
"""Mistral Large 4's softmax top-4 router.

The reference is ``MistralLarge4TextTopkRouter.forward``: a float32 gate GEMM,
a float32 softmax over *all* 128 experts, top-4 with ``sorted=False``, then an
unconditional renormalization ``weights / (sum + 1e-20)`` and a
``routed_scaling_factor`` of 1.0.

The group machinery in the reference is a no-op (``n_group = topk_group = 1``),
so none of it is implemented: no group limiting, no DeepSeek ``noaux_tc``
two-stage selection, and no bias correction -- this checkpoint ships no
``e_score_correction_bias`` at all. ``single_group_router`` is the kernel that
implements exactly this, and it renormalizes unconditionally, which is the
settled semantics here.

The gate projection runs in float32 on both operands, as the reference does,
and it runs float32-precise arithmetic up to 256 rows, so a row's routing does
not depend on how many rows share the launch in the band a served request
reaches: a 64-row context step and a 256-row verify step agree bit for bit.
The activation's widening is normally a launch of its own; on SM100 the GEMV
instead widens it in registers (``mo.router.gate.mixed.gemv``), which is
bit-identical to the cast followed by the fp32 GEMV because BF16 widens
losslessly and the fused launch keeps that GEMV's K split. Above the fused
window the op stages the cast and the matmul again, and this model bounds that
matmul's precise route at ``_PRECISE_MAX_ROWS``: its operands stay IEEE fp32
instead of being truncated to TF32's 10-bit mantissa, which is what an fp32
matmul of this shape otherwise does above 64 rows. Above that bound the
baseline's TF32 tile GEMM comes back, which is where the faster kernel matters.
The bound is a statement about the served batch: 256 rows is c64, 64 concurrent
requests at four verify rows each, so a launch wider than that is a prefill
chunk. A deployment above 64 concurrent requests would put a verify step on the
far side of it, where routing still moves with the chunk size the scheduler
picks.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.dtype import DType
from max.nn._graph_ops import _router_gate_mixed_gemv
from max.nn.common_layers.moe import MoEGate
from max.nn.kernels import moe_router_group_limited
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, match_operand_placement

from ._dispatch import fused_fp32_gemv_applies

_GATE_OP = "mo.router.gate.mixed.gemv"
"""The fused router gate's custom op."""

_FUSED_GEMV = F.functional(
    _router_gate_mixed_gemv,
    rule=match_operand_placement(_router_gate_mixed_gemv, "hidden_states"),
)
"""The gate GEMV that widens its BF16 activation in registers.

Its logits take the activation's placement: the router runs replicated, so a
rank's own logits stay on that rank.
"""

_FUSED_GEMV_MAX_ROWS = 64
"""The last row count that takes the fused launch rather than the chain.

The fused launch is IEEE fp32 at every row count; this window is a cost
choice. Up to 64 rows it is the fp32 dispatch's own GEMV with the cast folded
in, which is the precision edge the dispatch itself uses, so a later change to
either side shows up where this edge is tested.
"""

_PRECISE_MAX_ROWS = 256
"""The last row count whose gate projection stays IEEE fp32.

A c64 verify step is 64 requests at 4 rows each, so that band has to be
row-count invariant. Above it a launch is a prefill chunk, which assumes the
served batch stays at or below that c64 -- the batch the deployment sizes its
paged drafter block at (``dspark``) -- and there the fp32 GEMV re-reads the
weight per row tile and loses to the TF32 tile GEMM by an order of magnitude,
so prefill keeps the baseline. Routing above this bound still moves with the
chunk size the scheduler picks; that is the accepted trade.
"""


def _gate_logits(hidden_states: Tensor, weight: Tensor) -> Tensor:
    """Returns the float32 gate logits, IEEE fp32 up to ``_PRECISE_MAX_ROWS``.

    ``precise_max_m`` is what keeps this route row-count invariant above the
    fused window: without it an fp32 matmul of this shape truncates its
    operands to TF32 above 64 rows, so a row's routing changed with the row
    count (a 64-row context step and a 256-row verify step of the same request
    disagreed). Folding the activation's cast into the launch is the fused
    form's own purpose, at every row count it covers.

    Args:
        hidden_states: ``(seq_len, hidden_dim)`` BF16 hidden state.
        weight: The ``(num_experts, hidden_dim)`` gate weight.
    """
    if fused_fp32_gemv_applies(hidden_states, weight):
        return _FUSED_GEMV(
            hidden_states,
            F.cast(weight, DType.float32),
            max_m=_FUSED_GEMV_MAX_ROWS,
            precise_max_m=_PRECISE_MAX_ROWS,
        )
    return (
        F.cast(hidden_states, DType.float32) @ F.cast(weight, DType.float32).T
    )


class MistralLarge4Router(MoEGate):
    """Softmax top-4 router with unconditional weight renormalization."""

    def __init__(
        self, hidden_dim: int, num_experts: int, num_experts_per_token: int
    ) -> None:
        """Stores the routing geometry.

        Args:
            hidden_dim: Width of the hidden state.
            num_experts: Routed experts (128).
            num_experts_per_token: Top-k experts a token takes (4).
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        # `routed_scaling_factor` is 1.0 and there are no groups, so neither is
        # a parameter here; both are no-ops in the checkpoint's config.
        self.routed_scaling_factor = 1.0
        self.n_group = 1
        self.topk_group = 1

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each token's top-4 expert indices and weights.

        Args:
            hidden_states: ``(seq_len, hidden_dim)``, the post-attention-norm
                hidden state.

        Returns:
            A pair ``(topk_idx, topk_weight)``, each ``(seq_len, 4)``.
        """
        weight = self.gate_score.weight
        assert isinstance(weight, Tensor)
        logits = _gate_logits(hidden_states, weight)
        # The reference softmaxes all experts, then renormalizes the selection;
        # with one group the group mask is all-true, so no group selection and
        # no `noaux_tc` two-stage trick.
        scores = F.softmax(logits, axis=-1)
        # No `e_score_correction_bias` tensor exists in this checkpoint, so the
        # kernel's per-expert bias term is an all-zero constant rather than a
        # loaded parameter. It is built at the replicated mapping every rank
        # holds, so each rank makes its own copy: built on one device and
        # transferred it would lower to `mgp.buffer.device_to_device` between
        # two `mgp.device_wait` host calls, which device-graph capture rejects.
        bias = F.constant(
            np.zeros(self.num_experts, dtype=np.float32),
            DType.float32,
            device=DeviceMapping.replicated(scores.mesh),
        )
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            bias,
            self.num_experts,
            self.num_experts_per_token,
            1,
            1,
            True,
            1.0,
        )
        return (
            topk_idx.rebind_mapping(scores.mapping),
            topk_weight.rebind_mapping(scores.mapping),
        )
