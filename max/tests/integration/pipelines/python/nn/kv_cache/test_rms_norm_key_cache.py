# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
from context_utils import create_text_context
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Dim, Graph, TensorType, TensorValue
from max.nn.kernels import rms_norm_key_cache
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    RaggedKVCacheInputs,
)


@dataclass(frozen=True)
class RMSNormKeyCacheModel:
    """Model containing a single matmul KV ragged op."""

    fetch_layer: FetchContinuousBatchingKVCacheCollection
    """Layer for fetching a kv cache collection."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    total_seq_len: int
    """Total sequence length: sum(input_row_offsets)."""

    rms_norm_cols: Optional[int] = None
    """Number of columns in the RMSNorm operation."""

    def __call__(
        self,
        gamma: TensorValue,
        input_row_offsets: TensorValue,
        *fetch_args: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        rms_norm_key_cache(
            self.kv_params,
            self.fetch_layer(*fetch_args),
            gamma=gamma,
            epsilon=1e-5,
            layer_idx=self.layer_idx,
            total_seq_len=Dim(self.total_seq_len),
            input_row_offsets=input_row_offsets,
            rms_norm_cols=self.rms_norm_cols,
        )


@pytest.mark.parametrize(
    "dtype",
    [DType.float32],
)
def test_rms_norm_key_cache(session: InferenceSession, dtype: DType) -> None:
    seq_lens = [10, 4]
    batch_size = 2
    max_seq_len = 16
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(dtype, shape=[kv_params.head_dim])
    input_row_offsets_type = TensorType(DType.uint32, ["batch_size_plus_1"])
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=RMSNormKeyCacheModel(
            fetch_layer, kv_params, layer_idx=0, total_seq_len=sum(seq_lens)
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Claim seq_ids in cache.
    seq_ids = kv_manager.claim(n=batch_size)

    batch = [
        create_text_context(s, np.empty(seq_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
    fetch_args = kv_manager.fetch(batch)[0]
    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = fetch_args[0]
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    fetch_args = RaggedKVCacheInputs(
        Tensor.from_numpy(all_ones.copy()), *fetch_args[1:]
    )

    gamma = np.random.randn(kv_params.head_dim).astype(dtype.to_numpy())
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)
    model(gamma, input_row_offsets, *fetch_args)

    # Check that the RMSNorm wrote output to the KV cache.
    assert (fetch_args[0].to_numpy() != all_ones).any()


@pytest.mark.parametrize(
    "dtype",
    [DType.float32],
)
def test_partial_rms_norm_key_cache(
    session: InferenceSession, dtype: DType
) -> None:
    seq_lens = [
        10,
    ]
    batch_size = 1
    max_seq_len = 16
    gamma_size = 512
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=1,
        head_dim=576,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(dtype, shape=[gamma_size])
    input_row_offsets_type = TensorType(DType.uint32, ["batch_size_plus_1"])
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=RMSNormKeyCacheModel(
            fetch_layer,
            kv_params,
            layer_idx=0,
            total_seq_len=sum(seq_lens),
            rms_norm_cols=gamma_size,
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Claim seq_ids in cache.
    seq_ids = kv_manager.claim(n=batch_size)

    batch = [
        create_text_context(s, np.empty(seq_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
    fetch_args = kv_manager.fetch(batch)[0]
    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = fetch_args[0]
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    fetch_args = RaggedKVCacheInputs(
        Tensor.from_numpy(all_ones.copy()), *fetch_args[1:]
    )

    gamma = np.random.randn(gamma_size).astype(dtype.to_numpy())
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)
    model(gamma, input_row_offsets, *fetch_args)

    # shape: [batch_size,kv_dim,num_layers,max_seq_len,n_kv_heads,head_dim]
    kv_block = fetch_args[0].to_numpy()

    # Check that the first 512 elements of each head is normalized
    for seq_idx in range(seq_lens[0]):
        assert np.isclose(
            kv_block[0, 0, 0, seq_idx, 0, :gamma_size], gamma, rtol=1e-05
        ).all()

    # Check that the last 64 elements of each head is unchanged
    for seq_idx in range(seq_lens[0]):
        assert np.isclose(
            kv_block[0, 0, 0, seq_idx, 0, gamma_size:],
            np.ones((kv_params.head_dim - gamma_size), dtype=np.float32),
        ).all()


@pytest.mark.parametrize(
    "dtype",
    [DType.float32],
)
def test_rms_norm_new_key_cache(
    session: InferenceSession, dtype: DType
) -> None:
    seq_lens = [
        10,
    ]
    batch_size = 1
    max_seq_len = 32
    gamma_size = 128
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(dtype, shape=[gamma_size])
    input_row_offsets_type = TensorType(DType.uint32, ["batch_size_plus_1"])
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=RMSNormKeyCacheModel(
            fetch_layer,
            kv_params,
            layer_idx=0,
            total_seq_len=sum(seq_lens),
            rms_norm_cols=gamma_size,
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Claim seq_ids in cache.
    seq_ids = kv_manager.claim(n=batch_size)

    batch = [
        create_text_context(s, np.empty(seq_lens[i]))
        for i, s in enumerate(seq_ids)
    ]

    # note that unlike previous tests, we step the kv cache by 10 tokens
    # this is to test that we only operate on the new tokens
    fetch_args = kv_manager.fetch(batch)[0]
    for ctx in batch:
        ctx.update(42)
    kv_manager.step(batch)
    fetch_args = kv_manager.fetch(batch)[0]

    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = fetch_args[0]
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    fetch_args = RaggedKVCacheInputs(
        Tensor.from_numpy(all_ones.copy()), *fetch_args[1:]
    )

    gamma = np.random.randn(gamma_size).astype(dtype.to_numpy())
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)
    model(gamma, input_row_offsets, *fetch_args)

    # shape: [batch_size,kv_dim,num_layers,max_seq_len,n_kv_heads,head_dim]
    kv_block = fetch_args[0].to_numpy()

    # check that for the first 10 tokens all heads are unchanged
    for seq_idx in range(10):
        for head_idx in range(kv_params.n_kv_heads):
            assert np.isclose(
                kv_block[0, 0, 0, seq_idx, head_idx, :gamma_size],
                np.ones((gamma_size), dtype=np.float32),
            ).all()

    # check that for the next 10 tokens, all heads are normalized
    for seq_idx in range(10, 20):
        for head_idx in range(kv_params.n_kv_heads):
            assert np.isclose(
                kv_block[0, 0, 0, seq_idx, head_idx, :gamma_size],
                gamma,
                rtol=1e-05,
            ).all()


@pytest.mark.parametrize(
    "kv_dtype, gamma_dtype",
    [
        (DType.float32, DType.float16),
        (DType.bfloat16, DType.float32),
    ],
)
def test_rms_norm_key_cache_dtype_mismatch(
    session: InferenceSession, kv_dtype: DType, gamma_dtype: DType
) -> None:
    """Tests that a TypeError is raised when gamma dtype mismatches kv dtype."""
    seq_lens = [10]
    batch_size = 1
    max_seq_len = 16
    kv_params = KVCacheParams(
        dtype=kv_dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )
    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=1,
        devices=[CPU()],
        session=session,
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(gamma_dtype, shape=[kv_params.head_dim])
    input_row_offsets_type = TensorType(DType.uint32, ["batch_size_plus_1"])
    expected_msg = (
        f"expected gamma dtype {gamma_dtype} to match KV dtype {kv_dtype}"
    )
    with pytest.raises(TypeError, match=expected_msg):
        graph = Graph(
            "matmul_kv_cache_ragged",
            forward=RMSNormKeyCacheModel(
                fetch_layer, kv_params, layer_idx=0, total_seq_len=sum(seq_lens)
            ),
            input_types=[
                gamma_type,
                input_row_offsets_type,
                *kv_manager.input_symbols()[0],
            ],
        )
