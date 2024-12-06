# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

import numpy as np
import pytest
from max.driver import CPU, CUDA, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from nn import Linear
from nn.attention import Attention

ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-2
N_HEADS = 32
N_KV_HEADS = N_HEADS
HEAD_DIM = 128
HIDDEN_DIM = N_KV_HEADS * HEAD_DIM
MAX_SEQ_LEN = 512
NUM_LAYERS = 10
LAYER_IDX = 0
BATCH_SIZE = 4


def _attention_layer(
    dtype: DType,
    mask_dtype: DType,
    device: Device,
    cache_strategy: KVCacheStrategy,
    session: InferenceSession,
) -> tuple[Graph, KVCacheParams, ContinuousBatchingKVCacheManager]:
    # Initialize input types
    input_type = TensorType(dtype, ["batch_size", "seq_len", HIDDEN_DIM])
    attn_mask_type = TensorType(
        mask_dtype, ["batch_size", "n_heads", "seq_len", "post_seq_len"]
    )

    wq_type = TensorType(dtype, [HIDDEN_DIM, N_HEADS * HEAD_DIM])
    wk_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wv_type = TensorType(dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM])
    wo_type = TensorType(dtype, [N_HEADS * HEAD_DIM, HIDDEN_DIM])
    valid_lengths_type = TensorType(DType.uint32, ["batch_size"])

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
        devices=[device],
        session=session,
    )

    # Fetch
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    blocks_type, cache_lengths, lookup_table, is_cache_empty = (
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
            cache_lengths,  # 8
            lookup_table,  # 9
            is_cache_empty,  # 10
        ],
    ) as graph:
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
        ) = graph.inputs  # type: ignore

        # Concat wq, wk, wv into wqkv
        wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)  # type: ignore

        # Get KV Collection
        kv_collection = fetch_op(
            blocks,  # type: ignore
            cache_lengths,  # type: ignore
            lookup_table,  # type: ignore
            is_cache_empty,  # type: ignore
        )

        # Update this if provided
        kv_params.cache_strategy = cache_strategy

        attn_fn = Attention(
            n_heads=N_HEADS,
            kv_params=kv_params,
            layer_idx=ops.constant(LAYER_IDX, DType.uint32),
            wqkv=wqkv,
            wo=Linear(wo),  # type: ignore
        )

        attn_out, _ = attn_fn(
            x,  # type: ignore
            kv_collection,
            valid_lengths=valid_lengths,
            attention_mask=attn_mask,
        )

        graph.output(attn_out)

        return graph, kv_params, kv_manager  # type: ignore


@pytest.mark.parametrize(
    "start_pos,seq_len",
    [
        (0, 128),
        (9, 1),
    ],
)
def test_attention_gpu(start_pos, seq_len):
    # This tests that the attention mask is calculating valid logits.
    # It does not test that these logits match a reference implementation.
    host = CPU(0)
    device0 = CUDA(0)
    devices = [device0]
    session = InferenceSession(devices=devices)
    # Get Graph
    graph, _, kv_manager = _attention_layer(
        DType.float32,
        DType.float32,
        device0,
        KVCacheStrategy.CONTINUOUS,
        session,
    )
    compiled = session.load(graph)

    # Claim seq_ids in cache
    seq_ids = []
    for _ in range(BATCH_SIZE):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(seq_ids)[0]
    )

    hidden_states = Tensor.from_numpy(
        np.ones((BATCH_SIZE, seq_len, HIDDEN_DIM), dtype=np.float32),
    ).to(device0)
    attn_mask = Tensor.from_numpy(
        np.ones((BATCH_SIZE, N_HEADS, seq_len, seq_len), dtype=np.float32),
    ).to(device0)
    wq = Tensor.from_numpy(
        np.ones((HIDDEN_DIM, N_HEADS * HEAD_DIM), dtype=np.float32),
    ).to(device0)
    wk = Tensor.from_numpy(
        np.ones((HIDDEN_DIM, N_KV_HEADS * HEAD_DIM), dtype=np.float32),
    ).to(device0)
    wv = Tensor.from_numpy(
        np.ones((HIDDEN_DIM, N_KV_HEADS * HEAD_DIM), dtype=np.float32),
    ).to(device0)
    wo = Tensor.from_numpy(
        np.ones((N_HEADS * HEAD_DIM, HIDDEN_DIM), dtype=np.float32),
    ).to(device0)
    valid_lengths = Tensor.from_numpy(
        np.full((BATCH_SIZE), seq_len, dtype=np.uint32)
    ).to(device0)

    results = compiled.execute(
        hidden_states,
        attn_mask,
        wq,
        wk,
        wv,
        wo,
        valid_lengths,
        blocks,
        cache_lengths,
        lookup_table_tensor,
        is_cache_empty_buf,
        copy_inputs_to_device=False,
    )
    for result in results:
        if isinstance(result, Tensor):
            assert np.all(result.to(host).to_numpy() != np.inf)
