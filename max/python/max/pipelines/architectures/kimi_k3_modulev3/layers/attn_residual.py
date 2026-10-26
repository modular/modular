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
"""Kimi K3's attention-residual mixture (ModuleV3): this model's ONLY residual
stream.

Direct V3 port of ``kimi_k3/layers/attn_residual.py``. There is no
``x = x + f(x)`` anywhere in this decoder. A running ``prefix_sum`` and a
``block_residual`` stack of earlier per-block prefixes feed a learned SOFTMAX
MIXTURE (the shared ``attn_res_mix`` custom kernel, fp32 accumulation
regardless of input dtype) that a sublayer consumes and then discards -- the
value that advances the stream is ``prefix_sum``, never the mixture. See the
V2 module's docstring for the three specific ways a port gets this wrong
(layer-0 has an empty stack so the embedding is NOT added by a plain sum;
scores come from RMS-normalized candidates but the weighted average is over
the RAW ones; ``block_residual`` grows with DEPTH, not sequence length, so it
is a Python list of ``Tensor``, not a cache).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from max.experimental import functional as F
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.graph import TensorType, TensorValue, ops

from .norm import KIMI_RMS_NORM_EPS


def _attn_res_mix(
    candidates: Tensor,
    proj_weight: Tensor,
    norm_weight: Tensor,
    *,
    eps: float,
    out_type: TensorType,
) -> TensorValue:
    # Under TP this runs once per device, so the output takes that device.
    local_out_type = TensorType(
        out_type.dtype, out_type.shape, device=TensorValue(candidates).device
    )
    return cast(
        TensorValue,
        ops.custom(
            "attn_res_mix",
            device=TensorValue(candidates).device,
            values=[
                TensorValue(candidates),
                TensorValue(proj_weight),
                TensorValue(norm_weight),
            ],
            out_types=[local_out_type],
            parameters={"eps": eps},
        )[0],
    )


_attn_res_mix_op = F.functional(_attn_res_mix)


def kimi_k3_attn_res_mix(
    prefix_sum: Tensor,
    block_residual: Sequence[Tensor],
    proj_weight: Tensor,
    norm_weight: Tensor,
    eps: float = KIMI_RMS_NORM_EPS,
) -> Tensor:
    """Applies ``_apply_attn_res``: a softmax mixture over the candidates.

    Args:
        prefix_sum: ``[tokens, hidden]`` running prefix, the LAST candidate.
        block_residual: Earlier prefixes, one ``[tokens, hidden]`` entry per
            completed block; may be empty (mixture is a no-op up to fp32
            rounding -- softmax over a single candidate).
        proj_weight: ``*_res_proj.weight``, ``[1, hidden]``.
        norm_weight: ``*_res_norm.weight``, ``[hidden]``.
    """
    out_dtype = prefix_sum.dtype
    candidates = F.cast(
        F.stack(list(block_residual) + [prefix_sum], axis=1), out_dtype
    )
    # `_attn_res_mix` replaces the device.
    out_device = (
        prefix_sum.mesh.devices[0]
        if prefix_sum.is_distributed
        else prefix_sum.device
    )
    out_type = TensorType(out_dtype, prefix_sum.shape, device=out_device)
    result = _attn_res_mix_op(
        candidates,
        F.cast(proj_weight, out_dtype),
        F.cast(norm_weight, out_dtype),
        eps=eps,
        out_type=out_type,
    )
    # The op has no sharding rule, so restate the replicated placement.
    if prefix_sum.is_distributed:
        result = result.rebind_mapping(
            NamedMapping(prefix_sum.mesh, (None, None))
        )
    return result


def kimi_k3_attn_res_pre_attention(
    hidden_states: Tensor,
    block_residual: Sequence[Tensor],
    layer_idx: int,
    attn_res_block_size: int,
    proj_weight: Tensor,
    norm_weight: Tensor,
    eps: float = KIMI_RMS_NORM_EPS,
) -> tuple[Tensor, Tensor | None, list[Tensor]]:
    """Runs the first half of ``_forward_attn_residual``.

    Returns:
        ``(hidden_states, prefix_sum, block_residual)`` where ``hidden_states``
        feeds ``input_layernorm`` and ``prefix_sum`` is ``None`` exactly on the
        layers that append -- the signal that the stream restarts from this
        layer's attention output.
    """
    x = hidden_states
    entry = x
    prefix_sum: Tensor | None = entry
    stack = list(block_residual)
    if stack:
        x = kimi_k3_attn_res_mix(x, stack, proj_weight, norm_weight, eps)
    if layer_idx % attn_res_block_size == 0:
        stack = stack + [entry]
        prefix_sum = None
    return x, prefix_sum, stack


def kimi_k3_attn_res_post_attention(
    prefix_sum: Tensor | None,
    attn_output: Tensor,
    block_residual: Sequence[Tensor],
    proj_weight: Tensor,
    norm_weight: Tensor,
    eps: float = KIMI_RMS_NORM_EPS,
) -> tuple[Tensor, Tensor]:
    """Runs the second half of ``_forward_attn_residual``.

    Returns:
        ``(hidden_states, prefix_sum)`` where ``hidden_states`` feeds
        ``post_attention_layernorm`` and the layer's output is
        ``prefix_sum + ffn_output``.
    """
    updated = attn_output if prefix_sum is None else prefix_sum + attn_output
    mixed = kimi_k3_attn_res_mix(
        updated, block_residual, proj_weight, norm_weight, eps
    )
    return mixed, updated
