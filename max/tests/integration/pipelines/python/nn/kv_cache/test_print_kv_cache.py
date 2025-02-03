# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheManager,
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)


@dataclass(frozen=True)
class PrintKVCacheModel:
    """Model containing a single print KV cache op."""

    fetch_layer: FetchContinuousBatchingKVCacheCollection
    """Layer for fetching a kv cache collection."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        valid_lengths: TensorValue,
        *fetch_args: TensorValue,
    ) -> None:
        """Stages a graph consisting of a print KV cache op.

        This contains both the print KV cache op and a "fetch" op to get a
        KVCacheCollection.
        """
        kv_collection = self.fetch_layer(*fetch_args)
        ops.inplace_custom(
            "mo.print_kv_cache.continuous_batching",
            values=[
                valid_lengths,
                kv_collection,
                ops.constant(self.layer_idx, DType.uint32),
                ops.constant(True, DType.bool),
            ],
            parameters={
                "num_heads": self.kv_params.n_kv_heads_per_device,
                "head_dim": self.kv_params.head_dim,
                "type": self.kv_params.dtype,
            },
        )


@pytest.mark.parametrize(
    "dtype",
    [
        d
        for d in DType
        if d
        not in [
            # Skip types unsupported on CPU.
            DType._unknown,
            DType.f8e4m3,
            DType.f8e4m3fnuz,
            DType.f8e5m2,
            DType.f8e5m2fnuz,
            DType.float16,
            # Skip bf16 since ARM CPU doesn't support it.
            DType.bfloat16,
        ]
    ],
)
def test_print_kv_cache(dtype: DType) -> None:
    """Tests compiling a print KV cache op."""
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_batch_size=1,
        max_seq_len=1,
        num_layers=1,
        devices=[CPU()],
        session=InferenceSession(),
    )
    fetch_layer = FetchContinuousBatchingKVCacheCollection(kv_params)

    batch_size = 2
    graph = Graph(
        "print_kv_cache",
        forward=PrintKVCacheModel(fetch_layer, kv_params, layer_idx=0),
        input_types=[
            TensorType(dtype=DType.uint32, shape=[batch_size]),
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the print KV cache model.
    InferenceSession().load(graph)
