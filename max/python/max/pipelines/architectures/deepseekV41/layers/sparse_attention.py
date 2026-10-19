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
"""Sparse attention over a sliding window plus selected compressed entries.

Each query attends to its recent sliding-window positions and to a selected
subset of compressed cache entries through one softmax, with a per-head sink
logit that contributes to the denominator only.
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.kernels import latent_sparse_attention_ragged
from max.nn.kv_cache import PagedCacheValues


def sparse_attention(
    q: TensorValue,
    input_row_offsets: TensorValue,
    comp_indices: TensorValue | None,
    attn_sink: TensorValue,
    swa_collection: PagedCacheValues,
    comp_collection: PagedCacheValues | None,
    layer_swa: TensorValue,
    layer_comp: TensorValue | None,
    *,
    scale: float,
    window: int,
    split_k: bool = True,
    portable_p_bf16: bool = False,
) -> TensorValue:
    """Attends a ragged batch of queries to their window and compressed rows.

    Each query sees up to ``window`` sliding-window positions plus up to
    ``comp_indices.shape[-1]`` selected compressed entries. A layer without a
    compressed cache passes ``comp_indices=None``: the kernel always reads the
    compressed operand's device pointer, which a zero-width tensor lacks, so
    the window cache stands in for it with a single skipped ``-1`` index.

    Args:
        q: ``[total_rows, num_heads, head_dim]`` queries, ragged over the
            batch.
        input_row_offsets: ``[batch + 1]`` uint32 row offsets of ``q``.
        comp_indices: ``[total_rows, num_comp]`` int32 entry indices into
            the compressed cache, ``-1`` marking an unused slot; ``None``
            for a layer with only sliding-window attention.
        attn_sink: ``[num_heads]`` float32 per-head sink logits.
        swa_collection: The sliding-window cache (single latent head, paged
            by token position).
        comp_collection: The compressed cache (single latent head, paged by
            entry); must be ``None`` exactly when ``comp_indices`` is
            ``None``.
        layer_swa: uint32 scalar, this layer's index in the window cache.
        layer_comp: uint32 scalar, this layer's index in the compressed
            cache; must be ``None`` exactly when ``comp_indices`` is
            ``None``.
        scale: Softmax scale applied to the scores.
        window: Sliding window length in tokens.
        split_k: Whether the SM100 decode route may split a row's keys
            across CTAs; see ``latent_sparse_attention_ragged``.
        portable_p_bf16: Diagnostic passthrough; see
            ``latent_sparse_attention_ragged``.

    Returns:
        ``[total_rows, num_heads, head_dim]`` in ``q``'s dtype.

    Raises:
        ValueError: If ``comp_indices``, ``comp_collection`` and
            ``layer_comp`` are neither all ``None`` nor all given.
    """
    comp_args = (comp_indices, comp_collection, layer_comp)
    if any(a is None for a in comp_args) and any(
        a is not None for a in comp_args
    ):
        raise ValueError(
            "comp_indices, comp_collection and layer_comp must be all "
            "None (a ratio-0, SWA-only layer) or all given"
        )

    if comp_indices is None:
        # The kernel reads the operand's device pointer, which a zero-width
        # tensor does not provide.
        total_rows = q.shape[0]
        comp_indices = ops.broadcast_to(
            ops.constant(np.full((1, 1), -1, np.int32), DType.int32, q.device),
            [total_rows, 1],
        )
        comp_collection = swa_collection
        layer_comp = layer_swa
    assert comp_collection is not None and layer_comp is not None

    return latent_sparse_attention_ragged(
        q,
        input_row_offsets,
        comp_indices,
        attn_sink,
        swa_collection,
        comp_collection,
        layer_swa,
        layer_comp,
        scale=scale,
        window=window,
        split_k=split_k,
        portable_p_bf16=portable_p_bf16,
    )
