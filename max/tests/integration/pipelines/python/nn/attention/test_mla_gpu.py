# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import torch
from context_utils import create_text_context
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import flare_mla_decompress_k_cache, flare_mla_prefill_plan
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from torch.utils.dlpack import from_dlpack


def test_mla_prefill_plan() -> None:
    """Tests the mla_prefill_plan custom op."""
    # Set up hyperparameters for the test.
    host = CPU(0)
    device0 = Accelerator(0)
    devices = [device0]
    session = InferenceSession(devices=devices)

    page_size = 128
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)

    # Set MLIR types for the graph.
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])

    kv_manager = PagedKVCacheManager(
        kv_params,
        cache_memory=1024 * 1024 * 1024,
        page_size=page_size,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[Accelerator(0)],
        session=session,
    )
    fetch_op = FetchPagedKVCacheCollection(kv_params)

    def construct() -> Graph:
        with Graph(
            "call_mla_prefill_plan",
            input_types=[
                input_row_offsets_type,
                *kv_manager.input_symbols()[0],
            ],
        ) as g:
            input_row_offsets = g.inputs[0].tensor
            layer_idx = ops.constant(0, DType.uint32)

            kv_collection = fetch_op(*[v.tensor for v in g.inputs[1:]])

            results = flare_mla_prefill_plan(
                kv_params, input_row_offsets, kv_collection, layer_idx, 32
            )

            g.output(results[0].tensor, results[1].tensor, results[2].tensor)
        return g

    graph = construct()

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

    results = model.execute(
        input_row_offsets.to(device0), *fetch_args, copy_inputs_to_device=False
    )

    buffer_row_offsets_ref = np.zeros((16, batch_size + 1), dtype=np.int32)
    buffer_row_offsets_ref[0, 1] = 10
    buffer_row_offsets_ref[0, 2] = 32
    buffer_row_offsets_ref[1, 2] = 8

    cache_offsets_ref = np.zeros((16, batch_size), dtype=np.int32)
    cache_offsets_ref[1] = np.array([10, 22], dtype=np.int32)
    cache_offsets_ref[2:16, 0] = 10
    cache_offsets_ref[2:16, 1] = 30

    buffer_lengths_ref = -1 * np.ones((16,), dtype=np.int32)
    buffer_lengths_ref[0] = 32
    buffer_lengths_ref[1] = 8

    assert np.all(
        from_dlpack(results[0]).cpu().numpy() == buffer_row_offsets_ref
    )
    assert np.all(from_dlpack(results[1]).cpu().numpy() == cache_offsets_ref)
    assert np.all(from_dlpack(results[2]).cpu().numpy() == buffer_lengths_ref)


def test_mla_decompress_k_cache() -> None:
    """Tests the mla_decompress_k_cache custom op."""
    # Set up hyperparameters for the test.
    host = CPU(0)
    device0 = Accelerator(0)
    devices = [device0]
    session = InferenceSession(devices=devices)

    page_size = 128
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=576,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)

    # Set MLIR types for the graph.
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])
    weight_type = TensorType(
        DType.float32,
        [4096, 512],
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        cache_memory=1024 * 2 * 576,
        page_size=page_size,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=1,
        devices=[Accelerator(0)],
        session=session,
    )
    fetch_op = FetchPagedKVCacheCollection(kv_params)

    def construct() -> Graph:
        with Graph(
            "call_mla_decompress_k_cache",
            input_types=[
                input_row_offsets_type,
                weight_type,
                *kv_manager.input_symbols()[0],
            ],
        ) as g:
            input_row_offsets = g.inputs[0].tensor
            weight = g.inputs[1].tensor
            layer_idx = ops.constant(0, DType.uint32)

            kv_collection = fetch_op(*[v.tensor for v in g.inputs[2:]])

            # Allocate a buffer to hold KV cache for 60 decompressed tokens
            buffer_tok_size = 60

            (buffer_row_offsets, cache_offsets, buffer_lengths) = (
                flare_mla_prefill_plan(
                    kv_params,
                    input_row_offsets,
                    kv_collection,
                    layer_idx,
                    buffer_tok_size,
                )
            )

            buffer_lengths_host = buffer_lengths.to(DeviceRef.CPU())

            result = flare_mla_decompress_k_cache(
                kv_params,
                buffer_row_offsets[0, :],  # Process first chunk only
                cache_offsets[0, :],
                buffer_lengths_host[0],
                weight,
                kv_collection,
                layer_idx,
                buffer_tok_size,
            )

            g.output(result)
        return g

    graph = construct()

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
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(batch)[0]
    )

    new_blocks = torch.randn(size=blocks.shape, dtype=torch.float32)

    weight = (
        torch.randn(size=weight_type.shape.static_dims, dtype=torch.float32)
        / 128.0
    )

    results = model.execute(
        input_row_offsets.to(device0),
        Tensor.from_numpy(weight.numpy()).to(device0),
        Tensor.from_numpy(new_blocks.numpy()).to(device0),
        cache_lengths,
        lookup_table_tensor,
        is_cache_empty_buf,
        copy_inputs_to_device=False,
    )

    # Concatenate tokens from blocks to form ragged reference cache
    # blocks shape: [block_num, kv_dim, layers, page_size, num_heads, head_dim]
    # Extract first 10 tokens from block 0 and first 30 tokens from block 1 to match prompt_lens
    ref_ragged_cache = torch.concatenate(
        (new_blocks[0, 0, 0, :10, 0, :512], new_blocks[1, 0, 0, :30, 0, :512]),
        dim=0,
    )

    ref_output = ref_ragged_cache @ weight.T

    graph_output = from_dlpack(results[0]).cpu()[: ref_output.shape[0], :]

    torch.testing.assert_close(
        ref_output,
        graph_output,
        rtol=1e-3,
        atol=1e-3,
    )
