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
"""Kimi K3's attention-residual mixture, fused with the norm after it.

It is the model's only residual stream.
Direct V3 port of ``kimi_k3/layers/attn_residual.py``. There is no
``x = x + f(x)`` anywhere in this decoder. A running ``prefix_sum`` and a
``block_residual`` stack of earlier per-block prefixes feed a learned SOFTMAX
MIXTURE (the ``attn_res_mix_norm`` custom kernel, fp32 accumulation
regardless of input dtype) that a sublayer consumes and then discards -- the
value that advances the stream is ``prefix_sum``, never the mixture. The
mixture's only consumer is the RMSNorm the next sublayer applies, so the
fused kernel reads the candidates in place (no materialized stack) and emits
that normed value directly. See the V2 module's docstring for the three
specific ways a port gets this wrong (layer-0 has an empty stack so the
embedding is NOT added by a plain sum; scores come from RMS-normalized
candidates but the weighted average is over the RAW ones; ``block_residual``
grows with DEPTH, not sequence length, so it is a Python list of ``Tensor``,
not a cache).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from max import functional as F
from max.graph import TensorType, TensorValue, ops
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from .norm import KIMI_RMS_NORM_EPS, kimi_rms_norm


def _attn_res_mix_norm(
    candidates: list[Tensor],
    proj_weight: Tensor,
    res_norm_weight: Tensor,
    out_norm_weight: Tensor,
    *,
    eps: float,
    norm_eps: float,
    out_type: TensorType,
) -> TensorValue:
    # Under TP this runs once per device, so the output takes that device.
    device = TensorValue(candidates[0]).device
    local_out_type = TensorType(out_type.dtype, out_type.shape, device=device)
    return cast(
        TensorValue,
        ops.custom(
            "attn_res_mix_norm",
            device=device,
            values=[TensorValue(c) for c in candidates]
            + [
                TensorValue(proj_weight),
                TensorValue(res_norm_weight),
                TensorValue(out_norm_weight),
            ],
            out_types=[local_out_type],
            parameters={"eps": eps, "norm_eps": norm_eps},
        )[0],
    )


_attn_res_mix_norm_op = F.functional(_attn_res_mix_norm)

# One CTA per token covers `ATTN_RES_MIX_NORM_BLOCK_SIZE` (1024) x
# `ATTN_RES_MIX_NORM_ELEMS_PER_THREAD` (8) hidden elements in
# `Kernels/lib/attn_res/mix_norm.mojo`; the launcher raises beyond it, so
# wider hidden sizes take the unfused path at graph-build time.
_MIX_NORM_MAX_HIDDEN = 1024 * 8


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


def kimi_k3_attn_res_mix_norm(
    prefix_sum: Tensor,
    block_residual: Sequence[Tensor],
    proj_weight: Tensor,
    res_norm_weight: Tensor,
    out_norm_weight: Tensor,
    eps: float = KIMI_RMS_NORM_EPS,
    norm_eps: float = KIMI_RMS_NORM_EPS,
) -> Tensor:
    """Applies ``_apply_attn_res`` and RMS-normalizes the mixture, fused.

    Args:
        prefix_sum: ``[tokens, hidden]`` running prefix, the LAST candidate.
        block_residual: Earlier prefixes, one ``[tokens, hidden]`` entry per
            completed block; may be empty (mixture is a no-op up to fp32
            rounding -- softmax over a single candidate).
        proj_weight: ``*_res_proj.weight``, ``[1, hidden]``.
        res_norm_weight: ``*_res_norm.weight``, ``[hidden]``.
        out_norm_weight: The next sublayer's norm weight -- the
            ``input_layernorm`` / ``post_attention_layernorm`` the mixture's
            output feeds, or the final norm at the model level, ``[hidden]``.
        eps: Epsilon the mixture's score-side RMS norm uses.
        norm_eps: Epsilon the output norm uses.
    """
    out_dtype = prefix_sum.dtype
    # The kernels take the candidates as a variadic list, so no stack is
    # materialized.
    candidates = [
        c.cast(out_dtype) for c in list(block_residual) + [prefix_sum]
    ]
    # The ops replace the device.
    out_device = (
        prefix_sum.mesh.devices[0]
        if prefix_sum.is_distributed
        else prefix_sum.device
    )
    out_type = TensorType(out_dtype, prefix_sum.shape, device=out_device)
    if int(prefix_sum.shape[-1]) > _MIX_NORM_MAX_HIDDEN:
        mixed = _attn_res_mix_op(
            F.stack(candidates, axis=1),
            proj_weight.cast(out_dtype),
            res_norm_weight.cast(out_dtype),
            eps=eps,
            out_type=out_type,
        )
        if prefix_sum.is_distributed:
            mixed = mixed.rebind_mapping(
                NamedMapping(prefix_sum.mesh, (None, None))
            )
        return kimi_rms_norm(mixed, out_norm_weight.cast(out_dtype), norm_eps)
    result = _attn_res_mix_norm_op(
        candidates,
        proj_weight.cast(out_dtype),
        res_norm_weight.cast(out_dtype),
        out_norm_weight.cast(out_dtype),
        eps=eps,
        norm_eps=norm_eps,
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
    out_norm_weight: Tensor,
    eps: float = KIMI_RMS_NORM_EPS,
) -> tuple[Tensor, Tensor | None, list[Tensor]]:
    """Runs the first half of ``_forward_attn_residual``.

    Returns:
        ``(normed, prefix_sum, block_residual)`` where ``normed`` is what
        ``input_layernorm`` produces (the mixture fused into the norm when
        the stack is nonempty, a plain norm at layer 0, whose stack is empty)
        and ``prefix_sum`` is ``None`` exactly on the layers that append --
        the signal that the stream restarts from this layer's attention
        output.
    """
    x = hidden_states
    entry = x
    prefix_sum: Tensor | None = entry
    stack = list(block_residual)
    if stack:
        x = kimi_k3_attn_res_mix_norm(
            x, stack, proj_weight, norm_weight, out_norm_weight, eps, eps
        )
    else:
        x = kimi_rms_norm(x, out_norm_weight, eps)
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
    out_norm_weight: Tensor,
    eps: float = KIMI_RMS_NORM_EPS,
) -> tuple[Tensor, Tensor]:
    """Runs the second half of ``_forward_attn_residual``.

    Returns:
        ``(normed, prefix_sum)`` where ``normed`` is what
        ``post_attention_layernorm`` produces (fused into the mixture) and
        the layer's output is ``prefix_sum + ffn_output``.
    """
    updated = attn_output if prefix_sum is None else prefix_sum + attn_output
    normed = kimi_k3_attn_res_mix_norm(
        updated,
        block_residual,
        proj_weight,
        norm_weight,
        out_norm_weight,
        eps,
        eps,
    )
    return normed, updated
