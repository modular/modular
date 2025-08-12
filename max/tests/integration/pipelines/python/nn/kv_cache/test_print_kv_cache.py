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
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)


@dataclass(frozen=True)
class PrintKVCacheModel:
    """Model containing a single print KV cache op."""

    fetch_layer: FetchPagedKVCacheCollection
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
        page_size = self.kv_params.page_size
        if page_size is None:
            raise ValueError(
                "KVCacheParams.page_size cannot be none, when printing."
            )
        ops.inplace_custom(
            "mo.print_kv_cache.paged",
            device=valid_lengths.device,
            values=[
                valid_lengths,
                kv_collection,
                ops.constant(
                    self.layer_idx, DType.uint32, device=DeviceRef.CPU()
                ),
                ops.constant(True, DType.bool, device=DeviceRef.CPU()),
            ],
            parameters={
                "num_heads": self.kv_params.n_kv_heads_per_device,
                "head_dim": self.kv_params.head_dim,
                "dtype": self.kv_params.dtype,
                "page_size": page_size,
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
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
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
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        max_batch_size=1,
        max_seq_len=1,
        num_layers=1,
        devices=[CPU()],
        session=InferenceSession(),
        cache_memory=1024 * 1024 * 1024,
        page_size=128,
    )
    fetch_layer = FetchPagedKVCacheCollection(kv_params)

    batch_size = 2
    graph = Graph(
        "print_kv_cache",
        forward=PrintKVCacheModel(fetch_layer, kv_params, layer_idx=0),
        input_types=[
            TensorType(
                dtype=DType.uint32, shape=[batch_size], device=DeviceRef.CPU()
            ),
            *kv_manager.input_symbols()[0],
        ],
    )

    # Compile and init the print KV cache model.
    InferenceSession().load(graph)
