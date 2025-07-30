# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable

import numpy as np
import pytest
import torch
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import MHAMaskVariant, flash_attention_ragged
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from modular_graph_test import are_all_tensor_values, modular_graph_test
from test_common.context_utils import create_text_context

ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-2
N_HEADS = 1
N_KV_HEADS = 1
HEAD_DIM = 16
HIDDEN_DIM = N_KV_HEADS * HEAD_DIM
MAX_SEQ_LEN = 512
NUM_LAYERS = 10
LAYER_IDX = 0
BATCH_SIZE = 4


@pytest.mark.parametrize(
    "cache_strategy,mask_strategy",
    [
        (KVCacheStrategy.CONTINUOUS, MHAMaskVariant.CAUSAL_MASK),
        (KVCacheStrategy.PAGED, MHAMaskVariant.CAUSAL_MASK),
        (KVCacheStrategy.CONTINUOUS, MHAMaskVariant.CHUNKED_CAUSAL_MASK),
        (KVCacheStrategy.PAGED, MHAMaskVariant.CHUNKED_CAUSAL_MASK),
        (KVCacheStrategy.CONTINUOUS, MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK),
        (KVCacheStrategy.PAGED, MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK),
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
        cache_strategy=cache_strategy,
        page_size=128,
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

    kv_manager: KVCacheManager
    fetch_op: (
        FetchContinuousBatchingKVCacheCollection | FetchPagedKVCacheCollection
    )
    if cache_strategy == KVCacheStrategy.CONTINUOUS:
        kv_manager = ContinuousBatchingKVCacheManager(
            kv_params,
            max_batch_size=2,
            max_seq_len=100,
            num_layers=1,
            devices=[CPU()],
            session=session,
        )
        fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    else:
        kv_manager = PagedKVCacheManager(
            kv_params,
            cache_memory=1024 * 1024 * 1024,
            page_size=128,
            max_batch_size=2,
            max_seq_len=100,
            num_layers=1,
            devices=[CPU()],
            session=session,
        )
        fetch_op = FetchPagedKVCacheCollection(kv_params)

    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_manager.input_symbols()[0]
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
            assert are_all_tensor_values(g.inputs)
            (
                input,
                input_row_offsets,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs
            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = fetch_op(
                blocks, cache_lengths, lookup_table, is_cache_empty
            )
            result = flash_attention_ragged(
                kv_params,
                input,
                input_row_offsets,
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
        kv_manager.external_claim(context.request_id)

    input_row_offsets = Tensor(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(batch)[0]
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
        execute: Callable[[Sequence[Tensor]], Tensor],
        inputs: Sequence[Tensor],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
        assert np.any(result != np.nan)
        assert np.any(result != np.inf)
