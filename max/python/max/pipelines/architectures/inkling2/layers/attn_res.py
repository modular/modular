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
"""Inkling 2's block attention residual.

Every sublayer reads a softmax mixture of the earlier blocks' stored sums and
the live partial sum, in place of the plain residual stream. The mixture is
position-local: it holds no state across steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from max import functional as F
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn import Module
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from .norm import unweighted_rms_norm

ATTN_RES_EPS = 1e-6


class AttnResQueries(Module[[], None]):
    """The learned queries, under the checkpoint's names.

    ``attn_queries[i]`` feeds the attention of layer ``i + 1``: ``[3, hidden]``
    on a global layer, whose q, k and v each get their own mixture, else
    ``[hidden]``. ``mlp_queries[i]`` feeds the MLP of layer ``i``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        is_local: Sequence[bool],
    ) -> None:
        self.attn_queries = [
            Tensor.zeros(
                [hidden_size] if is_local[layer] else [3, hidden_size],
                dtype=DType.float32,
            )
            for layer in range(1, num_layers)
        ]
        self.mlp_queries = [
            Tensor.zeros([hidden_size], dtype=DType.float32)
            for _ in range(num_layers)
        ]
        self.output_query = Tensor.zeros([hidden_size], dtype=DType.float32)


def attn_res_mix(
    sources: Sequence[Tensor], queries: Tensor, *, normalize_values: bool
) -> list[Tensor]:
    """Mixes the sources once for each query row, in float32.

    Scores come from the RMS-normalized sources. The mixed values are the
    normalized sources too when ``normalize_values``.

    Args:
        sources: The stored blocks then the live partial sum, each
            ``[tokens, hidden]``.
        queries: ``[hidden]`` or ``[num_queries, hidden]`` float32.
        normalize_values: Whether the mixture averages the normalized sources.

    Returns:
        One float32 ``[tokens, hidden]`` mixture per query row.
    """
    hidden = int(sources[0].shape[-1])
    queries = queries.reshape([-1, hidden])
    stack = F.stack([s.cast(DType.float32) for s in sources], axis=1)
    normed = unweighted_rms_norm(stack, ATTN_RES_EPS)
    values = normed if normalize_values else stack
    mixes = []
    for row in range(int(queries.shape[0])):
        scores = (normed * queries[row]).sum(axis=-1) * hidden**-0.5
        probs = F.softmax(scores, axis=1)
        mixes.append((probs * values).sum(axis=1).squeeze(1))
    return mixes


# One CTA per token covers 1024 threads x 8 elements in
# `Kernels/lib/attn_res/inkling2_mix_norm.mojo`; the launcher raises beyond.
_MIX_NORM_MAX_HIDDEN = 1024 * 8
# The op's variadic block list is compiled for 1..15 entries.
_MIX_NORM_MAX_BLOCKS = 15


def _inkling2_mix_norm(
    blocks: list[Tensor],
    partial: Tensor,
    query: Tensor,
    gamma: Tensor,
    *,
    query_row: int,
    norm_eps: float,
    normalize_values: bool,
    out_type: TensorType,
) -> TensorValue:
    # Under TP this runs once per device, so the output takes that device.
    device = TensorValue(partial).device
    local_out_type = TensorType(out_type.dtype, out_type.shape, device=device)
    return cast(
        TensorValue,
        ops.custom(
            "inkling2_mix_norm",
            device=device,
            values=[TensorValue(b) for b in blocks]
            + [TensorValue(partial), TensorValue(query), TensorValue(gamma)],
            out_types=[local_out_type],
            parameters={
                "eps": ATTN_RES_EPS,
                "norm_eps": norm_eps,
                "normalize_values": normalize_values,
                "query_row": query_row,
                "score_scale": int(out_type.shape[-1]) ** -0.5,
            },
        )[0],
    )


_inkling2_mix_norm_op = F.functional(_inkling2_mix_norm)


def can_fuse_mix_norm(blocks: Sequence[Tensor], partial: Tensor) -> bool:
    """Whether ``attn_res_mix_norm`` can take this site's sources."""
    return (
        1 <= len(blocks) <= _MIX_NORM_MAX_BLOCKS
        and int(partial.shape[-1]) <= _MIX_NORM_MAX_HIDDEN
    )


def attn_res_mix_norm(
    blocks: Sequence[Tensor],
    partial: Tensor,
    query: Tensor,
    gamma: Tensor,
    *,
    query_row: int,
    norm_eps: float,
    normalize_values: bool,
) -> Tensor:
    """``Inkling2RMSNorm(attn_res_mix([*blocks, partial], query)[query_row])``.

    Fused into one launch. The partial keeps its dtype (the MLP site's float32
    residual is never rounded on the way in); the output is bfloat16, as the
    sublayer norm's is.

    Args:
        blocks: The stored block sums, ``[tokens, hidden]`` bfloat16 each.
        partial: The live partial sum, ``[tokens, hidden]``.
        query: ``[hidden]`` or ``[num_queries, hidden]`` float32 queries.
        query_row: The query row to mix for.
        gamma: The sublayer norm's weight.
        norm_eps: The sublayer norm's epsilon.
        normalize_values: Whether the mixture averages the normalized sources.
    """
    hidden = int(partial.shape[-1])
    queries = query.reshape([-1, hidden])
    out_device = (
        partial.mesh.devices[0] if partial.is_distributed else partial.device
    )
    out_type = TensorType(DType.bfloat16, partial.shape, device=out_device)
    result = _inkling2_mix_norm_op(
        [b.cast(DType.bfloat16) for b in blocks],
        partial,
        queries,
        gamma,
        query_row=query_row,
        norm_eps=norm_eps,
        normalize_values=normalize_values,
        out_type=out_type,
    )
    # The op has no sharding rule, so restate the replicated placement.
    if partial.is_distributed:
        result = result.rebind_mapping(NamedMapping(partial.mesh, (None, None)))
    return result
