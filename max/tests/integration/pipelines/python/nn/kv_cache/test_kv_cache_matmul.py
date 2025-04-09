# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from context_utils import create_text_context
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue, ops
from max.mlir import StringAttr
from max.nn.kernels import (
    fused_qkv_ragged_matmul,
    matmul_k_cache_ragged,
    matmul_kv_cache_ragged,
)
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from modular_graph_test import are_all_tensor_values, modular_graph_test


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
    input_row_offsets_type = TensorType(
        DType.uint32,
        [
            "input_row_offsets_len",
        ],
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_manager.input_symbols()[0]
    )

    def construct() -> Graph:
        with Graph(
            "call_ragged_qkv_matmul",
            input_types=[
                input_type,
                input_row_offsets_type,
                wqkv_type,
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
                wqkv,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs
            layer_idx = ops.constant(0, DType.uint32)

            kv_collection = fetch_op(
                blocks, cache_lengths, lookup_table, is_cache_empty
            )
            result = fused_qkv_ragged_matmul(
                kv_params,
                input,
                input_row_offsets,
                wqkv,
                kv_collection,
                layer_idx,
                num_q_heads,
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
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[i] = running_sum

    batch = [
        create_text_context(s, np.empty(prompt_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
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
            3: blocks,
            4: cache_lengths,
            5: lookup_table_tensor,
            6: is_cache_empty_buf,
        },
    )
    def test_runs_without_nan(execute, inputs, torch_inputs):
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
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
        input_row_offsets: TensorValue,
        weight: TensorValue,
        *fetch_args: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        matmul_kv_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection=self.fetch_layer(*fetch_args),
            layer_idx=self.layer_idx,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        DType.float32,
        # TODO(bduke): support converting to torch tensor from bfloat16 driver
        # tensor.
        # DType.bfloat16,
    ],
)
def test_matmul_kv_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_kv_cache_ragged custom op."""
    # Set up hyperparameters for the test.
    torch_dtype = {
        DType.float32: torch.float32,
        DType.bfloat16: torch.bfloat16,
    }[dtype]
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype, ["total_seq_len", num_q_heads * kv_params.head_dim]
    )
    wkv_type = TensorType(
        dtype,
        [
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
    )
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(fetch_layer, kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wkv_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Claim seq_ids in cache.
    seq_ids = []
    for _ in range(batch_size):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Tensor(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[i] = running_sum

    batch = [
        create_text_context(s, np.empty(prompt_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
    fetch_args = kv_manager.fetch(batch)[0]
    kv_blocks = fetch_args[0]
    # First check that the KV cache was zeroed out on initialization.
    assert not kv_blocks.to_numpy().any()

    hidden_states = torch.randn(
        size=[total_seq_len, num_q_heads * kv_params.head_dim],
        dtype=torch_dtype,
    )
    wkv = torch.randn(size=wkv_type.shape.static_dims, dtype=torch_dtype)
    model(hidden_states, input_row_offsets, wkv, *fetch_args)

    # Check that the matmul wrote output to the KV cache.
    assert kv_blocks.to_numpy().any()


@dataclass(frozen=True)
class MatmulKRaggedModel:
    """Model containing a single matmul KV ragged op."""

    fetch_layer: FetchPagedKVCacheCollection
    """Layer for fetching a kv cache collection."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        hidden_states: TensorValue,
        input_row_offsets: TensorValue,
        weight: TensorValue,
        *fetch_args: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        matmul_k_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection=self.fetch_layer(*fetch_args),
            layer_idx=self.layer_idx,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        DType.float32,
    ],
)
def test_matmul_k_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_k_cache_ragged custom op."""
    # Set up hyperparameters for the test.
    page_size = 128
    torch_dtype = {
        DType.float32: torch.float32,
        DType.bfloat16: torch.bfloat16,
    }[dtype]
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype, ["total_seq_len", num_q_heads * kv_params.head_dim]
    )
    wk_type = TensorType(
        dtype,
        [
            kv_params.n_kv_heads * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
    )
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])

    kv_manager = PagedKVCacheManager(
        kv_params,
        cache_memory=1024 * 1024 * 1024,
        page_size=page_size,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchPagedKVCacheCollection(kv_params)

    graph = Graph(
        "matmul_k_cache_ragged",
        forward=MatmulKRaggedModel(fetch_layer, kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wk_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Claim seq_ids in cache.
    seq_ids = []
    for _ in range(batch_size):
        seq_id = kv_manager.claim(1)
        seq_ids.append(seq_id[0])

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Tensor(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    batch = [
        create_text_context(s, np.empty(prompt_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
    fetch_args = kv_manager.fetch(batch)[0]
    kv_blocks = fetch_args[0]
    # First check that the KV cache was zeroed out on initialization.
    assert not kv_blocks.to_numpy().any()

    hidden_states = torch.randn(
        size=[total_seq_len, num_q_heads * kv_params.head_dim],
        dtype=torch_dtype,
    )
    wk = torch.randn(size=wk_type.shape.static_dims, dtype=torch_dtype)
    model(hidden_states, input_row_offsets, wk, *fetch_args)

    ref_results = (hidden_states @ wk.T).numpy()

    # Check that the matmul wrote output to the KV cache.
    host_kv_blocks = kv_blocks.to_numpy()
    host_page_table = fetch_args[2].to_numpy()

    for batch_idx in range(batch_size):
        # Calculate starting position for this batch
        batch_start = (
            int(np.cumsum(prompt_lens[:batch_idx])) if batch_idx != 0 else 0
        )
        batch_len = prompt_lens[batch_idx]
        num_pages = int(np.ceil(batch_len / float(page_size)))

        # Validate each page in the batch
        for page_idx in range(num_pages):
            block = host_page_table[batch_idx, page_idx]
            tokens_in_page = min(page_size, batch_len - page_idx * page_size)

            # Validate each head's projections
            for head_idx in range(kv_params.n_kv_heads):
                head_start = head_idx * kv_params.head_dim
                head_end = head_start + kv_params.head_dim

                # Compare cached values with reference results
                cached = host_kv_blocks[
                    block, 0, 0, :tokens_in_page, head_idx, :
                ]
                expected = ref_results[
                    batch_start : (batch_start + tokens_in_page),
                    head_start:head_end,
                ]
                assert np.isclose(
                    cached, expected, rtol=1e-03, atol=1e-03
                ).all()

            batch_start += page_size


@pytest.mark.parametrize(
    "dtype",
    [DType.float32, DType.bfloat16],
)
def test_matmul_kv_cache_ragged_chains(dtype: DType) -> None:
    """Tests that staging matmul_kv_cache_ragged threads chains."""
    # Set up hyperparameters for the test.
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype, ["total_seq_len", num_q_heads * kv_params.head_dim]
    )
    wkv_type = TensorType(
        dtype,
        [
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
    )
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=1,
        max_seq_len=1,
        num_layers=1,
        devices=[CPU()],
        session=InferenceSession(),
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)
    # Stage the fetch op + custom matmul KV cache ragged op graph.
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(fetch_layer, kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wkv_type,
            *kv_manager.input_symbols()[0],
        ],
    )
    matmul_kv_cache_op = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and "kv_matmul" in StringAttr(op.attributes["symbol"]).value
    ][0]
    assert len(matmul_kv_cache_op.results) == 1
    assert "!mo.chain" in str(matmul_kv_cache_op.results[0].type)

    matmul_args = matmul_kv_cache_op.operands
    assert "!mo.chain" in str(matmul_args[0].type)
