# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

from typing import List

import numpy as np
import pytest
from max.driver import CPU, CUDA, Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Device as GraphDevice
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
HIDDEN_DIM = N_HEADS * HEAD_DIM
MAX_SEQ_LEN = 512
NUM_LAYERS = 10
LAYER_IDX = 0
BATCH_SIZE = 4

N_DEVICES = 2


def _attention_block(params, inputs):
    (cache_strategy, fetch_op, kv_params) = params

    (
        x,
        attn_mask,
        wq,
        wk,
        wv,
        wo,
        valid_length,
        blocks,
        cache_lengths,
        lookup_table,
        is_cache_empty,
    ) = inputs

    # Concat wq, wk, wv into wqkv
    wqkv = ops.concat((wq, wk, wv), axis=1).transpose(0, 1)

    # Get KV Collection
    kv_collection = fetch_op(
        blocks, cache_lengths, lookup_table, is_cache_empty
    )

    # Update this if provided
    kv_params.cache_strategy = cache_strategy
    attn_fn = Attention(
        n_heads=kv_params.n_kv_heads_per_device,  # Should be n_heads_per_device
        kv_params=kv_params,
        layer_idx=ops.constant(LAYER_IDX, DType.uint32),
        wqkv=wqkv,
        wo=Linear(wo),
    )

    return attn_fn(
        x,
        kv_collection,
        valid_lengths=valid_length,
        attention_mask=attn_mask,
    )


def distribute_value(v, devices: List[Device]):
    return [v.to(GraphDevice(device.label, device.id)) for device in devices]


def shard_attn_mask_value(v, devices: List[Device]):
    n_devices = len(devices)
    size = v.shape[1] // n_devices
    return [
        v[:, i * size : (i + 1) * size, :, :].to(
            GraphDevice(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def shard_col_value(v, devices: List[Device]):
    n_devices = len(devices)
    col_size = v.shape[1].dim // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(
            GraphDevice(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def shard_row_value(v, devices: List[Device]):
    n_devices = len(devices)
    row_size = v.shape[0].dim // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(
            GraphDevice(device.label, device.id)
        )
        for i, device in enumerate(devices)
    ]


def _attention_layer(
    dtype: DType,
    mask_dtype: DType,
    cache_strategy: KVCacheStrategy,
    session: InferenceSession,
    devices: List[Device],
) -> tuple[Graph, KVCacheParams, ContinuousBatchingKVCacheManager]:
    # Initialize input types
    input_type = TensorType(
        dtype, ["batch_size", "seq_len", HIDDEN_DIM], device=GraphDevice.CPU()
    )
    attn_mask_type = TensorType(
        mask_dtype,
        ["batch_size", "n_heads", "seq_len", "post_seq_len"],
        device=GraphDevice.CPU(),
    )
    wq_type = TensorType(
        dtype, [HIDDEN_DIM, N_HEADS * HEAD_DIM], device=GraphDevice.CPU()
    )
    wk_type = TensorType(
        dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM], device=GraphDevice.CPU()
    )
    wv_type = TensorType(
        dtype, [HIDDEN_DIM, N_KV_HEADS * HEAD_DIM], device=GraphDevice.CPU()
    )
    wo_type = TensorType(
        dtype, [N_HEADS * HEAD_DIM, HIDDEN_DIM], device=GraphDevice.CPU()
    )
    valid_lengths_type = TensorType(
        DType.uint32, ["batch_size"], device=GraphDevice.CPU()
    )

    # Initialize kv cache params and manager
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
        n_devices=len(devices),
    )

    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=16,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=NUM_LAYERS,
        devices=devices,
        session=session,
    )

    # Fetch
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    kv_inputs_all = kv_manager.input_symbols()
    kv_input_types = [
        inp for device_inputs in kv_inputs_all for inp in device_inputs
    ]  # flatten list of tuples to list of elements

    with Graph(
        "vanilla_opaque_attn",
        # TODO: Clean this up so we don't need to manually iterate types per device
        input_types=[
            input_type,  # 0
            attn_mask_type,  # 1
            wq_type,  # 2
            wk_type,  # 3
            wv_type,  # 4
            wo_type,  # 5
            valid_lengths_type,  # 6
            *kv_input_types,
        ],
    ) as graph:
        (x, attn_mask, wq, wk, wv, wo, valid_lengths, *kv_inputs) = graph.inputs

        blocks_all = [
            kv_inputs[i * 4 + 0] for i in range(len(devices))
        ]  # id = 0, 4 (first elem of each tuple is blocks)
        cache_lengths_all = [
            kv_inputs[i * 4 + 1] for i in range(len(devices))
        ]  # id = 1, 5 (second elem of each tuple is cache_lengths)
        lookup_table_all = [
            kv_inputs[i * 4 + 2] for i in range(len(devices))
        ]  # id = 2, 6 (third elem of each tuple is lookup_table)
        is_cache_empty_all = [
            kv_inputs[i * 4 + 3] for i in range(len(devices))
        ]  # id = 3, 7 (fourth elem of each tuple is is_cache_empty)

        x_devs = distribute_value(x, devices)
        valid_lengths_devs = distribute_value(valid_lengths, devices)
        attn_mask_devs = shard_attn_mask_value(attn_mask, devices)
        wo_devs = shard_row_value(wo, devices)
        wq_devs = shard_col_value(wq, devices)
        wk_devs = shard_col_value(wk, devices)
        wv_devs = shard_col_value(wv, devices)
        attn_out = [
            _attention_block(
                (cache_strategy, fetch_op, kv_params),
                (
                    x_devs[dev_id],
                    attn_mask_devs[dev_id],
                    wq_devs[dev_id],
                    wk_devs[dev_id],
                    wv_devs[dev_id],
                    wo_devs[dev_id].transpose(0, 1),
                    valid_lengths_devs[dev_id],
                    blocks_all[dev_id],
                    cache_lengths_all[dev_id],
                    lookup_table_all[dev_id],
                    is_cache_empty_all[dev_id],
                ),
            )
            for dev_id in range(len(devices))
        ]
        graph.output(*ops.allreduce.sum(attn_out))

        return graph, kv_params, kv_manager  # type: ignore


def execute_attn_for_devices(inputs, session, devices: List[Device]):
    session = InferenceSession(devices=devices)
    graph, _, kv_manager = _attention_layer(
        DType.float32,
        DType.float32,
        KVCacheStrategy.CONTINUOUS,
        session,
        devices,
    )
    # Claim seq_ids in cache
    seq_ids = kv_manager.claim(BATCH_SIZE)

    kv_cache_inputs = kv_manager.fetch(seq_ids)
    flattened_kv_cache_inputs = [
        inp for device_inputs in kv_cache_inputs for inp in device_inputs
    ]
    model = session.load(graph)

    all_inputs = [
        *inputs,
        *flattened_kv_cache_inputs,
    ]
    return model.execute(*all_inputs, copy_inputs_to_device=False)


@pytest.mark.parametrize(
    "start_pos,seq_len",
    [
        (0, 128),
        (9, 1),
    ],
)
def test_attention(start_pos, seq_len):
    # This tests that the attention mask is calculating valid logits.
    # It does not test that these logits match a reference implementation.

    # Initialize the device-contexts
    host = CPU(0)
    device0 = CUDA(0)
    device1 = CUDA(1)

    # Initialize Model inputs
    hidden_states = Tensor.from_numpy(
        np.ones((BATCH_SIZE, seq_len, HIDDEN_DIM), dtype=np.float32),
    )
    attn_mask = Tensor.from_numpy(
        np.ones((BATCH_SIZE, N_HEADS, seq_len, seq_len), dtype=np.float32),
    )
    wq = Tensor.from_numpy(
        np.ones((HIDDEN_DIM, N_HEADS * HEAD_DIM), dtype=np.float32),
    )
    wk = Tensor.from_numpy(
        np.ones((HIDDEN_DIM, N_KV_HEADS * HEAD_DIM), dtype=np.float32),
    )
    wv = Tensor.from_numpy(
        np.ones((HIDDEN_DIM, N_KV_HEADS * HEAD_DIM), dtype=np.float32),
    )
    wo = Tensor.from_numpy(
        np.ones((N_HEADS * HEAD_DIM, HIDDEN_DIM), dtype=np.float32),
    )
    valid_lengths = Tensor.from_numpy(
        np.full((BATCH_SIZE), seq_len, dtype=np.uint32)
    )
    model_inputs = (hidden_states, attn_mask, wq, wk, wv, wo, valid_lengths)

    # Run distributed and ensure all ranks are the same.
    devices = [device0, device1]
    devices_with_host = [host, *devices]
    res_rank0, res_rank1 = execute_attn_for_devices(
        model_inputs, InferenceSession(devices=devices_with_host), devices
    )
    np.testing.assert_allclose(
        res_rank0.to(host).to_numpy(), res_rank1.to(host).to_numpy()
    )

    # Run on single device and ensure same result
    devices = [device0]
    devices_with_host = [host, *devices]
    (expected_res,) = execute_attn_for_devices(
        model_inputs, InferenceSession(devices=devices_with_host), devices
    )
    np.testing.assert_allclose(
        res_rank0.to(host).to_numpy(), expected_res.to(host).to_numpy()
    )
