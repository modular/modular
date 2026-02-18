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
"""Test pipelines attention layer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.kv_cache import PagedKVCacheManager
from max.nn.legacy.kernels import MHAMaskVariant, flash_attention_ragged
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedCacheValues,
)
from modular_graph_test import modular_graph_test
from test_common.context_utils import create_text_context

ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-2
N_HEADS = 1
N_KV_HEADS = 1
HEAD_DIM = 16
HIDDEN_DIM = N_KV_HEADS * HEAD_DIM
MAX_SEQ_LEN = 512
NUM_LAYERS = 10
BATCH_SIZE = 4


@pytest.mark.parametrize(
    "cache_strategy,mask_strategy",
    [
        ("paged", MHAMaskVariant.CAUSAL_MASK),
        ("paged", MHAMaskVariant.CHUNKED_CAUSAL_MASK),
        ("paged", MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK),
    ],
)
def test_kv_cache_ragged_attention(
    session: InferenceSession,
    cache_strategy: KVCacheStrategy,
    mask_strategy: MHAMaskVariant,
) -> None:
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        cache_strategy=cache_strategy,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.float32,
        ["total_seq_len", num_q_heads, kv_params.head_dim],
        DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.CPU()
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_params.get_symbolic_inputs()[0]
    )

    def construct() -> Graph:
        with Graph(
            "call_ragged_attention",
            input_types=[
                input_type,
                input_row_offsets_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                is_cache_empty_type,
            ],
        ) as g:
            (
                input,
                input_row_offsets,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs
            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                is_cache_empty.tensor,
            )
            result = flash_attention_ragged(
                kv_params,
                input.tensor,
                input_row_offsets.tensor,
                kv_collection,
                layer_idx,
                mask_variant=mask_strategy,
                scale=math.sqrt(1.0 / kv_params.head_dim),
                local_window_size=8192,
            )
            g.output(result)
        return g

    g = construct()

    batch = [
        create_text_context(np.empty(prompt_lens[i])) for i in range(batch_size)
    ]

    for context in batch:
        kv_manager.claim(context.request_id, replica_idx=0)
        assert isinstance(kv_manager, PagedKVCacheManager)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.get_runtime_inputs([batch])[0]
    )

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offsets_len": len(prompt_lens) + 1,
        },
        provided_inputs={
            1: input_row_offsets,
            2: blocks,
            3: cache_lengths,
            4: lookup_table_tensor,
            5: is_cache_empty_buf,
        },
    )
    def test_runs_without_nan(
        execute: Callable[[Sequence[Buffer]], Buffer],
        inputs: Sequence[Buffer],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
        assert np.all(np.isfinite(result))
