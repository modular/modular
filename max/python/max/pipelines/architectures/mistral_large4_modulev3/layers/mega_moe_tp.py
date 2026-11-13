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
"""The fused TP-width MoE graph op (``mega_moe.tp``) and its switch.

With ``MODULAR_MEGA_MOE_TP=1`` set when the graph is built, a
tensor-parallel Mistral Large 4 rank runs its routed experts through one op:
a decode or verify step as one persistent launch (gather, both grouped
matmuls, the SiLU, both quantizes and the finalize fused, with the routing
laid out inside the launch), and any larger step as ``moe_create_indices``
and the unfused chain inside the same op. Both are bit-identical to the
chain the rank runs without the flag. The op reads W13 sigma-permuted, so the
flag also permutes the rank's gate/up bundle for every graph.
"""

from __future__ import annotations

import os

from max import functional as F
from max.dtype import DType
from max.graph import BufferType, BufferValue, TensorType, TensorValue, ops
from max.nn.legacy.activation import GatedActivation

MEGA_MOE_TP_ENV = "MODULAR_MEGA_MOE_TP"
"""``1`` routes a TP rank's experts through ``mega_moe.tp``; read where the
graph is built, so an exported MEF carries whichever arm it was built with."""

MAX_FUSED_TOKENS = 256
"""The most tokens a step may have to take the fused launch: 64 requests (the
TP8 export serves c1 through c64) times ``ML4_DSPARK_VERIFY_ROWS`` (4, a
DSpark verify step's rows a request), so every decode and verify step fits.
Raise it with either factor. Larger steps run the unfused chain in the same
op, bit-identical to the fused launch, so the bound moves time, not output."""

COUNTER_WORDS = 4096 * 32 + 1
"""The op's persistent counter workspace in ``uint32`` words: the component's
one-shot pool counter (4096 slots of 32 words) and one collective word. The
op refuses a smaller buffer."""


def mega_moe_tp_enabled() -> bool:
    """Returns whether ``MODULAR_MEGA_MOE_TP`` selects the fused op."""
    return os.environ.get(MEGA_MOE_TP_ENV) == "1"


def _mega_moe_tp(
    hidden_states: TensorValue,
    topk_ids: TensorValue,
    router_weights: TensorValue,
    gate_up_weight: TensorValue,
    gate_up_scales: TensorValue,
    down_weight: TensorValue,
    down_scales: TensorValue,
    gate_up_expert_scales: TensorValue,
    down_expert_scales: TensorValue,
    gate_up_input_scales: TensorValue,
    down_input_scales: TensorValue,
    max_fused_tokens: int = MAX_FUSED_TOKENS,
    counters: BufferValue | None = None,
) -> TensorValue:
    """Runs one rank's routed experts as ``mega_moe.tp``.

    Args:
        hidden_states: The rows the router read ``(tokens, hidden)`` bf16.
        topk_ids: The router's top-4 experts, token-major ``(tokens x 4,)``
            int32.
        router_weights: The router's top-4 weights ``(tokens, 4)`` float32.
        gate_up_weight: Sigma-permuted NVFP4 W13 ``(E, N1, hidden / 2)``.
        gate_up_scales: W13's block scales, 6-D (interleaved).
        down_weight: NVFP4 W2 ``(E, hidden, N1 / 4)``.
        down_scales: W2's block scales, 6-D (interleaved).
        gate_up_expert_scales: weight_scale_2 x input_scale per expert.
        down_expert_scales: weight_scale_2 x input_scale per expert.
        gate_up_input_scales: 1 / input_scale of the gate/up leg per expert.
        down_input_scales: 1 / input_scale of the down leg per expert.
        max_fused_tokens: The most tokens the fused launch takes.
        counters: The op's ``COUNTER_WORDS`` counter workspace, zeroed once
            and never again. Only the fused launch writes it, so a test that
            passes its own can tell which regime a step ran. ``None`` mints
            one in the graph.

    Returns:
        The routed partial ``(tokens, hidden)`` bf16, as ``mo.moe.finalize``
        writes it.
    """
    if counters is None:
        # The counters persist across executions: one buffer per call site,
        # zeroed once at load and never again (the launches rotate them).
        counters = ops.buffer_create(
            BufferType(
                DType.uint32,
                shape=[COUNTER_WORDS],
                device=hidden_states.device,
            ),
            init_value=0,
        )
    results = ops.inplace_custom(
        "mega_moe.tp",
        device=hidden_states.device,
        values=[
            hidden_states,
            topk_ids,
            router_weights,
            gate_up_weight,
            gate_up_scales,
            down_weight,
            down_scales,
            gate_up_expert_scales,
            down_expert_scales,
            gate_up_input_scales,
            down_input_scales,
            counters,
        ],
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=[hidden_states.shape[0], down_weight.shape[1]],
                device=hidden_states.device,
            )
        ],
        parameters={
            **GatedActivation().kernel_parameters(),
            "max_fused_tokens": max_fused_tokens,
        },
    )
    return results[0].tensor


mega_moe_tp = F.functional(_mega_moe_tp)
