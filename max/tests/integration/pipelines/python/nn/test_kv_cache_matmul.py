# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from modular_graph_test import modular_graph_test
from nn.kernels import fused_qkv_ragged_matmul, matmul_kv_cache_ragged


def test_fused_qkv_ragged_matmul(session: InferenceSession) -> None:
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.float32, ["total_seq_len", num_q_heads * kv_params.head_dim]
    )
    wqkv_type = TensorType(
        DType.float32,
        [
            num_q_heads * kv_params.head_dim,
            (num_q_heads + 2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
        ],
    )
    input_row_offset_type = TensorType(
        DType.uint32,
        [
            "input_row_offset_len",
        ],
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_cache_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_manager.input_symbols()[0]
    )

    with Graph(
        "call_ragged_qkv_matmul",
        input_types=[
            input_type,
            input_row_offset_type,
            wqkv_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            is_cache_empty_type,
        ],
    ) as g:
        (
            input,
            input_row_offset,
            wqkv,
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
        result = fused_qkv_ragged_matmul(
            kv_params,
            input,
            input_row_offset,
            wqkv,
            kv_collection,
            layer_idx,
            num_q_heads,
        )
        g.output(result)

    # Claim seq_ids in cache
    seq_ids = []
    for _ in range(batch_size):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    input_row_offset = Tensor(
        [batch_size + 1],
        DType.uint32,
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offset[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offset[i] = running_sum

    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(seq_ids)[0]
    )

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offset_len": len(prompt_lens) + 1,
        },
        provided_inputs={
            1: input_row_offset,
            3: blocks,
            4: cache_lengths,
            5: lookup_table_tensor,
            6: is_cache_empty_buf,
        },
    )
    def test_runs_without_nan(execute, inputs, torch_inputs):
        inputs = list(inputs)
        result = execute(inputs)
        assert np.any(result != np.nan)
        assert np.any(result != np.inf)


@dataclass(frozen=True)
class MatmulKVRaggedModel:
    """Model containing a single matmul KV ragged op."""

    fetch_layer: FetchContinuousBatchingKVCacheCollection
    """Layer for fetching a kv cache collection."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        hidden_states: TensorValue,
        input_row_offset: TensorValue,
        weight: TensorValue,
        *fetch_args,
    ) -> None:
        matmul_kv_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offset,
            weight,
            kv_collection=self.fetch_layer(*fetch_args),
            layer_idx=self.layer_idx,
        )


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_matmul_kv_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_kv_cache_ragged custom op."""
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    hidden_state_type = TensorType(
        DType.float32, ["total_seq_len", num_q_heads * kv_params.head_dim]
    )

    wkv_type = TensorType(
        DType.float32,
        [
            num_q_heads * kv_params.head_dim,
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
        ],
    )
    input_row_offset_type = TensorType(
        DType.uint32,
        [
            "input_row_offset_len",
        ],
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_cache_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(fetch_layer, kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offset_type,
            wkv_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Claim seq_ids in cache.
    seq_ids = []
    for _ in range(batch_size):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    # Compute input row offsets for ragged tensors.
    input_row_offset = Tensor([batch_size + 1], DType.uint32)
    running_sum = 0
    for i in range(batch_size):
        input_row_offset[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offset[i] = running_sum

    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(seq_ids)[0]
    )

    @modular_graph_test(
        session,
        graph,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offset_len": len(prompt_lens) + 1,
        },
        provided_inputs={
            1: input_row_offset,
            3: blocks,
            4: cache_lengths,
            5: lookup_table_tensor,
            6: is_cache_empty_buf,
        },
    )
    def test_runs_without_nan(execute, inputs, _torch_inputs):
        execute(list(inputs))
