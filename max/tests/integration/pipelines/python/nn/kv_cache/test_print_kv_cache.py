# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass

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
            # TODO(bduke): currently hardcoded to h8/d128.
            # Fix once we store parameters in the opaque MLIR type.
            "mo.print_kv_cache.continuous_batching.nhead_8.hdim_128.fp32",
            values=[
                valid_lengths,
                kv_collection,
                ops.constant(self.layer_idx, DType.uint32),
                ops.constant(True, DType.bool),
            ],
        )


def test_print_kv_cache() -> None:
    """Tests compiling a print KV cache op."""
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
    )

    kv_manager = ContinuousBatchingKVCacheManager(
        kv_params,
        max_cache_batch_size=1,
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
