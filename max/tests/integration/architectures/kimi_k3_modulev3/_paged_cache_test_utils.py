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
"""A paged KV cache for kernel tests, built without a cache manager.

Ports ``max/tests/integration/test_common/simple_kv_cache.py``'s
``paged_kv_cache_inputs`` to real, eager Tensor-API buffers -- that
module is visibility-restricted to ``max/tests``, and its own ``Buffer``
outputs would still need wrapping for a ``max.experimental`` layer either
way.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.kv_cache import (
    PACKED_PAGE_STRIDE,
    KVCacheParams,
    MLAAttnKey,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
    padded_lut_cols,
)
from max.support.math import ceildiv
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context


def _zeros(shape: tuple[int, ...], dtype: DType, device: Accelerator) -> Tensor:
    """A real, zero-filled Tensor of any dtype and rank.

    ``Tensor.zeros`` fills via a graph-traced broadcast, which the
    interpreter caps at rank 5 -- too low for the KV blocks buffer. This
    instead loads real zero bytes directly, the same device-transfer-and-
    reinterpret path real checkpoint weights use.
    """
    raw = np.zeros(
        (*shape[:-1], shape[-1] * dtype.size_in_bytes), dtype=np.uint8
    )
    buf = Buffer.from_dlpack(raw).to(device).view(dtype=dtype, shape=shape)
    return Tensor(storage=buf)


def block_ids_for_batch(
    seq_lens: Sequence[int], page_size: int
) -> list[list[int]]:
    """Assigns pages to each request in a batch, sequentially from page 0."""
    blocks = []
    next_block = 0
    for seq_len in seq_lens:
        num_blocks = ceildiv(seq_len, page_size)
        blocks.append(list(range(next_block, next_block + num_blocks)))
        next_block += num_blocks
    return blocks


def build_paged_cache(
    params: KVCacheParams,
    prompt_lens: Sequence[int],
    device: Accelerator,
    *,
    cache_lengths: Sequence[int] | None = None,
) -> PagedCacheValues:
    """Builds a real, single-device paged cache for exactly this batch.

    Args:
        params: Describes the page layout; must name exactly one device.
        prompt_lens: New tokens each request processes this step.
        device: The accelerator ``params`` names.
        cache_lengths: Tokens each request already holds. Defaults to zero
            for every request, i.e. a prefill into an empty cache.
    """
    if cache_lengths is None:
        cache_lengths = [0] * len(prompt_lens)
    seq_lens = [
        cache_len + prompt_len
        for cache_len, prompt_len in zip(
            cache_lengths, prompt_lens, strict=True
        )
    ]
    blocks = block_ids_for_batch(seq_lens, params.page_size)
    total_num_pages = sum(len(req_blocks) for req_blocks in blocks)
    max_prompt_length = max(prompt_lens)
    max_cache_length = max(seq_lens)

    # One page past the pool backs the null block the lookup table's tail
    # padding points at -- see `padded_lut_cols`. The fill value below must
    # be exactly `total_num_pages`, the null block's id: the kernel's SIMD
    # table read multiplies every entry (including the padding) by the page
    # stride with no sentinel check, and only the null page keeps that
    # address in bounds.
    kv_blocks_shape = (total_num_pages + 1, *params.shape_per_block)
    kv_blocks = _zeros(kv_blocks_shape, params.dtype, device)

    num_cols = padded_lut_cols(ceildiv(max_cache_length, params.page_size))
    lookup_table_np = np.full(
        (len(blocks), num_cols), total_num_pages, dtype=np.uint32
    )
    for batch_idx, req_blocks in enumerate(blocks):
        lookup_table_np[batch_idx, : len(req_blocks)] = req_blocks

    attn_key = params.resolve_attn_key(
        len(prompt_lens), max_prompt_length, max_cache_length
    )
    dispatch_metadata = attn_key.pack_into_buffer(device, max_cache_length)

    with ensure_context():
        lookup_table = Tensor(
            storage=Buffer.from_numpy(lookup_table_np).to(device)
        )
        return PagedCacheValues(
            kv_blocks=kv_blocks,
            cache_lengths=Tensor(
                storage=Buffer.from_numpy(
                    np.array(cache_lengths, np.uint32)
                ).to(device)
            ),
            lookup_table=lookup_table,
            max_prompt_length=Tensor(
                storage=Buffer.from_numpy(
                    np.array([max_prompt_length], np.uint32)
                )
            ),
            max_cache_length=Tensor(
                storage=Buffer.from_numpy(
                    np.array([max_cache_length], np.uint32)
                )
            ),
            page_stride=Tensor(
                storage=Buffer.from_numpy(
                    np.array([PACKED_PAGE_STRIDE], np.int64)
                )
            ),
            attention_dispatch_metadata=Tensor(storage=dispatch_metadata),
            mla_num_partitions=(
                Tensor(
                    storage=Buffer.from_numpy(
                        np.array([attn_key.num_partitions], np.int64)
                    )
                )
                if isinstance(attn_key, MLAAttnKey)
                else None
            ),
        )


def build_kda_state(
    *,
    num_kda_layers: int,
    num_requests: int,
    conv_dim: int,
    conv_kernel_size: int,
    num_heads: int,
    head_dim: int,
    conv_dtype: DType,
    recurrent_dtype: DType,
    device: Accelerator,
) -> RecurrentStateInputsPerDevice[Tensor, Tensor]:
    """Builds a real, single-device KDA recurrent state for exactly this batch.

    One page -- a block of ``num_kda_layers`` consecutive rows -- per
    request, so row ``(layer, request)`` is ``request * num_kda_layers +
    layer``, the same allocation :meth:`RecurrentStateRegion.rows_of`
    describes.
    """
    total_rows = num_requests * num_kda_layers
    conv_pool = _zeros(
        (total_rows, conv_dim, conv_kernel_size - 1), conv_dtype, device
    )
    recurrent_pool = _zeros(
        (total_rows, num_heads, head_dim, head_dim), recurrent_dtype, device
    )
    row_ids_np = np.array(
        [
            [
                request * num_kda_layers + layer
                for request in range(num_requests)
            ]
            for layer in range(num_kda_layers)
        ],
        dtype=np.uint32,
    )
    with ensure_context():
        conv_row_ids = Tensor(storage=Buffer.from_numpy(row_ids_np).to(device))
        recurrent_row_ids = Tensor(
            storage=Buffer.from_numpy(row_ids_np).to(device)
        )
        return RecurrentStateInputsPerDevice(
            leaves=(
                RecurrentLeafInputs(pool=conv_pool, live_row_ids=conv_row_ids),
                RecurrentLeafInputs(
                    pool=recurrent_pool, live_row_ids=recurrent_row_ids
                ),
            )
        )
