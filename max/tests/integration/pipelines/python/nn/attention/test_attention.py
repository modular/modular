# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

import numpy as np
import pytest
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
    load_kv_manager,
)
from modular_graph_test import are_all_tensor_values, modular_graph_test
from nn import Linear
from nn.attention import Attention
from nn.kernels import MHAMaskVariant, flash_attention_ragged

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
FAKE_TOKEN = 999


def _attention_layer(
    dtype: DType,
    mask_dtype: DType,
    device: Device,
    cache_strategy: KVCacheStrategy,
    session: InferenceSession,
) -> tuple[
    Graph,
    KVCacheParams,
    KVCacheManager,
]:
    # Initialize input types
    input_type = TensorType(dtype, ["batch_size", "seq_len", HIDDEN_DIM])
    attn_mask_type = TensorType(
        mask_dtype, ["batch_size", "n_kv_heads", "seq_len", "post_seq_len"]
    )

    wq_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wk_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wv_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wo_type = TensorType(dtype, [N_KV_HEADS * HEAD_DIM, HIDDEN_DIM])
    valid_lengths_type = TensorType(DType.uint32, ["batch_size"])

    # Initialize kv cache params and manager
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=cache_strategy,
    )

    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=16,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=NUM_LAYERS,
        devices=[device],
        session=session,
        page_size=128,
        available_cache_memory=1024 * 1024 * 1024,
    )

    # Fetch
    if isinstance(kv_manager, ContinuousBatchingKVCacheManager):
        fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    elif isinstance(kv_manager, PagedKVCacheManager):
        fetch_op = FetchPagedKVCacheCollection(kv_params)  # type: ignore
    else:
        raise ValueError("Unsupported kv_manager type")

    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_manager.input_symbols()[0]
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
            valid_lengths_type,  # 6
            blocks_type,  # 7
            cache_lengths_type,  # 8
            lookup_table_type,  # 9
            is_cache_empty_type,  # 10
        ],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        (
            x,
            attn_mask,
            wq,
            wk,
            wv,
            wo,
            valid_lengths,
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

        attn_out = attn_fn(
            x.tensor,
            kv_collection,
            valid_lengths=valid_lengths,
            attention_mask=attn_mask,
        )

        graph.output(attn_out)

        return graph, kv_params, kv_manager


def test_attention__wrong_mask_dtype():
    # This is expected to fail when passing a mask dtype that does not match the activation dtype.
    with pytest.raises(ValueError) as _:
        graph, _, _ = _attention_layer(
            DType.float32,
            DType.uint8,
            CPU(),
            KVCacheStrategy.CONTINUOUS,
            InferenceSession(devices=[CPU()]),
        )


def test_attention__wrong_strategy():
    # This is expected to fail when passing a naive kv cache strategy to the opaque attention kernel.
    # Get Graph.
    with pytest.raises(ValueError) as _:
        graph, _, _ = _attention_layer(
            DType.float32,
            DType.float32,
            CPU(),
            KVCacheStrategy.NAIVE,
            InferenceSession(devices=[CPU()]),
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
        DType.float32,
        DType.float32,
        CPU(),
        KVCacheStrategy.CONTINUOUS,
        InferenceSession(devices=[CPU()]),
    )

    # Claim seq_ids in cache
    seq_ids = []
    for _ in range(BATCH_SIZE):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    # Base the valid lengths on max_seq_len
    valid_lengths = Tensor.from_numpy(
        np.full((BATCH_SIZE), seq_len, dtype=np.uint32)
    )

    cache_lengths_in = {
        s: np.array([FAKE_TOKEN] * seq_len) for i, s in enumerate(seq_ids)
    }
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(cache_lengths_in)[0]
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
            6: valid_lengths,
            7: blocks,
            8: cache_lengths,
            9: lookup_table_tensor,
            10: is_cache_empty_buf,
        },
    )
    def test_runs_without_inf(execute, inputs, torch_inputs):
        inputs = list(inputs)
        results = execute(inputs)
        assert np.all(results != np.inf)


@pytest.mark.parametrize(
    "cache_strategy",
    [
        KVCacheStrategy.CONTINUOUS,
        KVCacheStrategy.PAGED,
    ],
)
def test_kv_cache_ragged_attention(session, cache_strategy):
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
        DType.float32, ["total_seq_len", num_q_heads, kv_params.head_dim]
    )
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])

    manager_kwargs = {
        "max_batch_size": 2,
        "max_seq_len": 100,
        "num_layers": 1,
        "devices": [CPU()],
        "session": session,
    }

    if cache_strategy == KVCacheStrategy.CONTINUOUS:
        kv_manager = ContinuousBatchingKVCacheManager(
            kv_params,
            **manager_kwargs,
        )
        fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    else:
        kv_manager = PagedKVCacheManager(  # type: ignore
            kv_params,
            cache_memory=1024 * 1024 * 1024,
            page_size=128,
            **manager_kwargs,
        )
        fetch_op = FetchPagedKVCacheCollection(kv_params)  # type: ignore

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
            layer_idx = ops.constant(
                0,
                DType.uint32,
            )

            kv_collection = fetch_op(
                blocks, cache_lengths, lookup_table, is_cache_empty
            )
            result = flash_attention_ragged(
                kv_params,
                input,
                input_row_offsets,
                kv_collection,
                layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
            )
            g.output(result)
        return g

    g = construct()

    # Claim seq_ids in cache
    seq_ids = []
    for _ in range(batch_size):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    input_row_offsets = Tensor(
        [batch_size + 1],
        DType.uint32,
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    cache_lengths_in = {
        s: np.array([FAKE_TOKEN] * prompt_lens[i])
        for i, s in enumerate(seq_ids)
    }
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(cache_lengths_in)[0]
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
    def test_runs_without_nan(execute, inputs, torch_inputs):
        inputs = list(inputs)
        result = execute(inputs)
        assert np.any(result != np.nan)
        assert np.any(result != np.inf)
