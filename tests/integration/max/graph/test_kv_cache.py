# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, TensorValue, ops

from nn.kv_cache import ContiguousKVCacheCollectionType
from nn.kv_cache_params import KVCacheParams


class KVCacheModel:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        key_cache: TensorValue,
        value_cache: TensorValue,
        cache_lengths: TensorValue,
        is_cache_empty: TensorValue,
        seq_ids: TensorValue,
        num_layers: TensorValue,
        batch_size: TensorValue,
    ) -> ContiguousKVCacheCollectionType:
        """Constructs an initial ContiguousKVCacheCollection for use downstream.
        """
        op_name = f"contiguous_kv_cache_collection_h{self.kv_params.n_kv_heads}_d{self.kv_params.head_dim}_{self.kv_params.layout}"
        return ops.custom(
            op_name,
            values=[
                key_cache,
                value_cache,
                cache_lengths,
                is_cache_empty,
                seq_ids,
                num_layers,
                batch_size,
            ],
            out_types=[ContiguousKVCacheCollectionType()],
        )[0]


def test_kv_cache(session: InferenceSession) -> None:
    kv_params = KVCacheParams(
        DType.float32, n_kv_heads=8, head_dim=128, device=CPU()
    )
    cache_type = TensorType(
        kv_params.dtype,
        (
            "start_pos",
            "n_layers",
            "batch_size",
            "n_kv_heads",
            "head_dim",
        ),
    )
    batch_size_param = 32
    cache_lengths_type = TensorType(DType.int32, (batch_size_param,))
    seq_ids_type = TensorType(DType.int32, ("seq_len",))
    int_scalar_type = TensorType(DType.int32, (1,))
    is_cache_empty_type = TensorType(DType.bool, (1,))
    graph = Graph(
        "kv_cache_collection",
        KVCacheModel(kv_params),
        input_types=[
            cache_type,
            cache_type,
            cache_lengths_type,
            is_cache_empty_type,
            seq_ids_type,
            int_scalar_type,
            int_scalar_type,
        ],
    )
    model = session.load(graph)

    max_seq_len = 5
    num_layers_param = 32
    k_cache = Tensor(
        shape=(
            num_layers_param,
            batch_size_param,
            kv_params.n_kv_heads,
            max_seq_len,
            kv_params.head_dim,
        ),
        dtype=kv_params.dtype,
    )
    v_cache = Tensor(shape=k_cache.shape, dtype=kv_params.dtype)
    cache_lengths = Tensor(shape=(batch_size_param,), dtype=DType.int32)
    for i in range(5):
        cache_lengths[i] = 1
    for i in range(6, cache_lengths.shape[0]):
        cache_lengths[i] = 0

    is_cache_empty = Tensor.zeros(shape=(1,), dtype=DType.bool)
    is_cache_empty[0] = False
    seq_ids = Tensor(shape=(1,), dtype=DType.int32)
    seq_ids[0] = 0
    num_layers = Tensor(shape=(1,), dtype=DType.int32)
    num_layers[0] = 1
    batch_size = Tensor(shape=(1,), dtype=DType.int32)
    batch_size[0] = 1

    kv_cache_collection = model.execute(
        k_cache,
        v_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )
    assert kv_cache_collection and kv_cache_collection[0]


if __name__ == "__main__":
    test_kv_cache(InferenceSession())
