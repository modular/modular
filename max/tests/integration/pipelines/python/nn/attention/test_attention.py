# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

import asyncio
import pytest
import numpy as np
from max.dtype import DType
from max.driver import Device, CPU
from max.graph import Graph, TensorType, ops
from modular_graph_test import modular_graph_test
from nn import Linear
from nn.attention import Attention
from nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
    FetchContinuousBatchingKVCacheCollection,
    ContinuousBatchingKVCacheManager,
)

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


def _attention_layer(
    dtype: DType,
    device: Device,
    cache_strategy: KVCacheStrategy,
) -> tuple[Graph, KVCacheParams, ContinuousBatchingKVCacheManager]:
    # Initialize input types
    input_type = TensorType(dtype, ["batch_size", "seq_len", HIDDEN_DIM])
    attn_mask_type = TensorType(
        dtype, ["batch_size", "n_kv_heads", "seq_len", "post_seq_len"]
    )

    wq_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wk_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wv_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wo_type = TensorType(dtype, [N_KV_HEADS * HEAD_DIM, HIDDEN_DIM])

    # Initialize kv cache params and manager
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=16,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=NUM_LAYERS,
        device=device,
    )

    # Fetch
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    blocks_type, cache_lengths, lookup_table, is_cache_empty = (
        kv_manager.input_symbols()
    )

    with Graph(
        "vanilla_opaque_attn",
        input_types=[
            input_type,  # 0
            attn_mask_type,  # 1
            wq_type,  # 2
            wk_type,  # 3
            wv_type,  # 4
            wo_type,  # 5
            blocks_type,  # 6
            cache_lengths,  # 7
            lookup_table,  # 8
            is_cache_empty,  # 9
        ],
    ) as graph:
        (
            x,
            attn_mask,
            wq,
            wk,
            wv,
            wo,
            blocks,
            cache_lengths,
            lookup_table,
            is_cache_empty,
        ) = graph.inputs

        # Concat wq, wk, wv into wqkv
        wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

        # Get KV Collection
        kv_collection = fetch_op(
            blocks, cache_lengths, lookup_table, is_cache_empty
        )

        # Update this if provided
        kv_params.cache_strategy = cache_strategy

        attn_fn = Attention(
            n_heads=N_HEADS,
            kv_params=kv_params,
            layer_idx=ops.constant(LAYER_IDX, DType.uint32),
            wqkv=wqkv,
            wo=Linear(wo),
        )

        attn_out, _ = attn_fn(
            x,
            attn_mask,
            kv_collection,
            cache_lengths,
        )

        graph.output(attn_out)

        return graph, kv_params, kv_manager


def test_attention__wrong_strategy():
    # This is expected to fail when passing a naive kv cache strategy to the opaque attention kernel.
    # Get Graph.
    with pytest.raises(ValueError) as _:
        graph, _, _ = _attention_layer(
            DType.float32, CPU(), KVCacheStrategy.NAIVE
        )


@pytest.mark.parametrize(
    "start_pos,seq_len",
    [
        (0, 10),
    ],
)
def test_attention__valid_logits(session, start_pos, seq_len):
    # This tests that the attention mask is calculating valid logits.
    # It does not test that these logits match a reference implementation.

    # Get Graph
    graph, _, kv_manager = _attention_layer(
        DType.float32, CPU(), KVCacheStrategy.CONTINUOUS
    )

    # Claim seq_ids in cache
    seq_ids = []
    for _ in range(BATCH_SIZE):
        seq_id = asyncio.run(kv_manager.claim(1))
        seq_ids.append(seq_id[0])

    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(seq_ids)
    )

    @modular_graph_test(
        session,
        graph,
        static_dims={
            "seq_len": seq_len,
            "post_seq_len": start_pos + seq_len,
            "batch_size": BATCH_SIZE,
        },
        provided_inputs={
            6: blocks,
            7: cache_lengths,
            8: lookup_table_tensor,
            9: is_cache_empty_buf,
        },
    )
    def test_runs_without_inf(execute, inputs, torch_inputs):
        inputs = list(inputs)
        results = execute(inputs)
        assert np.all(results != np.inf)
