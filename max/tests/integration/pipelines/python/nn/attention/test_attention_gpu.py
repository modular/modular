# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test pipelines attention layer."""

import math
from functools import partial
from typing import cast

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Tensor, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.nn import LinearV1, RMSNormV1
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flare_mla_prefill_ragged,
    flash_attention_gpu,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
    load_kv_manager,
)
from max.pipelines.architectures.llama_vision.cross_attention_decoder import (
    CrossSdpaAttention,
)
from max.support.math import ceildiv
from modular_graph_test import are_all_tensor_values
from test_common.context_utils import create_text_context
from test_common.distance_metrics import is_euclidean_distance_close
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaTextCrossAttention,
)

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


class CrossAttentionModel:
    """Model containing fetch and cross attention layers."""

    fetch: FetchPagedKVCacheCollection
    """Layer for fetching a kv cache collection."""

    cross_attention: CrossSdpaAttention
    """Layer for computing multimodal cross attention."""

    dtype: DType
    """DType of the model weights."""

    def __init__(
        self,
        config: MllamaTextConfig,
        kv_params: KVCacheParams,
        torch_cross_attn: MllamaTextCrossAttention,
        dtype: DType,
    ) -> None:
        """Inits fetch and cross attention layers using the torch model."""
        self.dtype = dtype

        self.fetch = FetchPagedKVCacheCollection(kv_params)

        # Use torch model weights to initialize MAX graph cross attention
        # shapes.
        self.cross_attention = CrossSdpaAttention(
            config.num_attention_heads,
            kv_params,
            layer_idx=0,
            q_proj=LinearV1(
                Weight(
                    name="wq",
                    dtype=self.dtype,
                    shape=torch_cross_attn.q_proj.weight.shape,
                    device=DeviceRef.GPU(),
                )
            ),
            wk=Weight(
                name="wk",
                dtype=self.dtype,
                shape=torch_cross_attn.k_proj.weight.shape,
                device=DeviceRef.GPU(),
            ),
            wv=Weight(
                name="wv",
                dtype=self.dtype,
                shape=torch_cross_attn.v_proj.weight.shape,
                device=DeviceRef.GPU(),
            ),
            o_proj=LinearV1(
                Weight(
                    name="wo",
                    dtype=self.dtype,
                    shape=torch_cross_attn.o_proj.weight.shape,
                    device=DeviceRef.GPU(),
                )
            ),
            q_norm=RMSNormV1(
                Weight(
                    name="q_norm",
                    dtype=self.dtype,
                    shape=torch_cross_attn.q_norm.weight.shape,
                    device=DeviceRef.GPU(),
                ),
                weight_offset=0.0,
            ),
            k_norm=RMSNormV1(
                Weight(
                    name="k_norm",
                    dtype=self.dtype,
                    shape=torch_cross_attn.k_norm.weight.shape,
                    device=DeviceRef.GPU(),
                ),
                weight_offset=0.0,
            ),
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        hidden_input_row_offsets: TensorValue,
        hidden_max_seq_len: TensorValue,
        cross_attention_states: TensorValue,
        cross_input_row_offsets: TensorValue,
        *fetch_args: TensorValue,
    ) -> TensorValue:
        """Builds the cross attention model graph."""
        kv_collection = self.fetch(*fetch_args)
        return self.cross_attention(
            hidden_states,
            hidden_input_row_offsets,
            hidden_max_seq_len,
            cross_attention_states,
            cross_input_row_offsets,
            kv_collection,
        )


@pytest.mark.skipif(accelerator_api() == "hip", reason="KERN-1466")
@pytest.mark.parametrize(
    "hidden_seq_lens",
    [
        [10, 4],
        [1, 2],
    ],
)
def test_cross_attention_gpu(hidden_seq_lens: list[int]) -> None:
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])
    session.set_debug_print_options("COMPACT")

    # Globally disable saving activations for backprop.
    torch.set_grad_enabled(False)

    num_tiles = 4
    # image_dim**2 // patch_dim**2 + 1 (cls token)
    num_vision_tokens = 1025
    cross_seq_len = num_tiles * num_vision_tokens

    config = MllamaTextConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        rope_theta=10000.0,
        max_position_embeddings=8192,
        _attn_implementation="sdpa",
    )
    # Set up PyTorch attention layer.
    torch_dtype = torch.float32
    torch_cross_attn = MllamaTextCrossAttention(config, layer_idx=0)
    torch_cross_attn.to(torch_dtype)

    # Set up MAX graph attention layer.
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    batch_size = len(hidden_seq_lens)

    dtype = DType.float32
    hidden_states_type = TensorType(
        dtype, ["total_seq_len", config.hidden_size], device=DeviceRef.GPU()
    )
    cross_attention_states_type = TensorType(
        dtype,
        shape=[batch_size * cross_seq_len, config.hidden_size],
        device=DeviceRef.GPU(),
    )

    input_row_offsets_type = TensorType(
        DType.uint32, shape=[batch_size + 1], device=DeviceRef.GPU()
    )
    hidden_max_seq_len_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )

    page_size = ceildiv(cross_seq_len, 128) * 128

    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=page_size,
    )
    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=batch_size,
        max_seq_len=config.max_position_embeddings,
        num_layers=config.num_hidden_layers,
        session=session,
        devices=[cuda],
        available_cache_memory=8 * 1024 * 1024 * 1024,
        page_size=page_size,
    )

    # Phase 1: op staging.

    # Construct and compile the MAX graph cross attention.
    graph = Graph(
        "test_cross_attn",
        forward=CrossAttentionModel(config, kv_params, torch_cross_attn, dtype),
        input_types=[
            # NOTE: 2 input row offsets: for hidden and cross attention states.
            hidden_states_type,
            input_row_offsets_type,
            hidden_max_seq_len_type,
            cross_attention_states_type,
            input_row_offsets_type,
            *kv_manager.input_symbols()[0],
        ],
    )

    # Phase 2: model compilation and weight initialization.

    # Map torch weight values to their MAX graph counterparts.
    weights_registry = {
        "wq": torch_cross_attn.q_proj.weight.detach(),
        "wk": torch_cross_attn.k_proj.weight.detach(),
        "wv": torch_cross_attn.v_proj.weight.detach(),
        "wo": torch_cross_attn.o_proj.weight.detach(),
        "q_norm": torch_cross_attn.q_norm.weight.detach(),
        "k_norm": torch_cross_attn.k_norm.weight.detach(),
    }
    cross_attn_model = session.load(graph, weights_registry=weights_registry)

    # Phase 3: execution.

    seq_ids = kv_manager.claim(n=batch_size)
    # Use cross states sequence length when fetching from the KV manager since
    # KV are cross states.
    batch = [
        create_text_context(
            s, np.empty(cross_seq_len), max_length=int(cross_seq_len * 1.1)
        )
        for s in seq_ids
    ]
    kv_cache_inputs = kv_manager.fetch(batch)[0]

    # Initialize model inputs.
    total_seq_len = sum(hidden_seq_lens)
    hidden_states = torch.randn(
        [total_seq_len, config.hidden_size], dtype=torch_dtype
    )
    cross_attention_states = torch.randn(
        cross_attention_states_type.shape.static_dims, dtype=torch_dtype
    )
    hidden_input_row_offsets = torch.tensor(
        [0, *np.cumsum(hidden_seq_lens)], dtype=torch.uint32
    )
    cross_input_row_offsets = torch.tensor(
        [i * cross_seq_len for i in range(batch_size + 1)],
        dtype=torch.uint32,
    )

    predicted = cross_attn_model.execute(
        Tensor.from_numpy(hidden_states.numpy()).to(cuda),
        Tensor.from_numpy(hidden_input_row_offsets.numpy()).to(cuda),
        np.array([max(hidden_seq_lens)], dtype=np.uint32),
        Tensor.from_numpy(cross_attention_states.numpy()).to(cuda),
        Tensor.from_numpy(cross_input_row_offsets.numpy()).to(cuda),
        *kv_cache_inputs,
    )[0]
    assert isinstance(predicted, Tensor)

    # Marshal extra inputs for torch.
    # Create padded inputs since the torch model doesn't support ragged
    # tensors.
    hidden_states_padded = torch.zeros(
        size=[batch_size, max(hidden_seq_lens), config.hidden_size],
        dtype=torch_dtype,
    )
    # Convert to int since torch can't subtract uint32.
    hidden_input_row_offsets = hidden_input_row_offsets.to(dtype=torch.int32)
    for batch_idx, (start, stop) in enumerate(
        zip(hidden_input_row_offsets[:-1], hidden_input_row_offsets[1:])
    ):
        hidden_states_padded[batch_idx, : stop - start] = hidden_states[
            start:stop
        ]

    attention_mask = torch.ones(
        [1, 1, max(hidden_seq_lens), cross_seq_len], dtype=torch.bool
    )
    position_ids = torch.arange(
        0, max(hidden_seq_lens), dtype=torch.long
    ).unsqueeze(0)
    expected = (
        torch_cross_attn(
            hidden_states=hidden_states_padded,
            position_embeddings=position_ids,
            cross_attention_states=cross_attention_states.reshape(
                [batch_size, num_tiles * num_vision_tokens, config.hidden_size]
            ),
            attention_mask=attention_mask,
        )[0]
        .detach()
        .numpy()
    )
    expected_ragged = np.empty(
        shape=[total_seq_len, config.hidden_size], dtype=dtype.to_numpy()
    )
    for batch_idx, (start, stop) in enumerate(
        zip(hidden_input_row_offsets[:-1], hidden_input_row_offsets[1:])
    ):
        expected_ragged[start:stop] = expected[batch_idx, : stop - start]

    # Compare the outputs.
    assert is_euclidean_distance_close(
        predicted.to_numpy(),
        expected_ragged,
        # Use bfloat16 epsilon since MHA accumulates in bfloat16 even for
        # float32 dtype.
        rtol=torch.finfo(torch.bfloat16).eps,
    )


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_kv_cache_paged_mla_prefill() -> None:
    cuda = Accelerator()
    session = InferenceSession(devices=[cuda])
    num_q_heads = 32
    q_head_dim = 192
    k_head_dim = 128
    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        cache_strategy=KVCacheStrategy.PAGED,
        page_size=128,
    )
    num_layers = 1
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, q_head_dim],
        DeviceRef.GPU(),
    )
    k_buffer_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, k_head_dim],
        DeviceRef.GPU(),
    )
    v_buffer_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, k_head_dim],
        DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )
    kv_manager = PagedKVCacheManager(
        kv_params,
        cache_memory=1024 * 1024 * 32,
        page_size=128,
        max_batch_size=2,
        max_seq_len=100,
        num_layers=num_layers,
        devices=[cuda],
        session=session,
    )
    fetch_op = FetchPagedKVCacheCollection(kv_params)

    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_manager.input_symbols()[0]
    )

    def construct() -> Graph:
        with Graph(
            "call_mla_prefill",
            input_types=[
                input_type,
                input_row_offsets_type,
                k_buffer_type,
                v_buffer_type,
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
                k_buffer,
                v_buffer,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = fetch_op(
                blocks, cache_lengths, lookup_table, is_cache_empty
            )
            result = flare_mla_prefill_ragged(
                kv_params,
                input,
                k_buffer,
                v_buffer,
                input_row_offsets,
                input_row_offsets,  # actually buffer_row_offsets
                cache_lengths,
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                1,  # scale
            )
            g.output(result[0].cast(DType.float32))
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
    input_row_offsets[batch_size] = running_sum
    input_row_offsets = input_row_offsets.to(cuda)

    batch = [
        create_text_context(s, np.empty(prompt_lens[i]))
        for i, s in enumerate(seq_ids)
    ]
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.fetch(batch)[0]
    )
    model = session.load(g)

    input_tensor = Tensor.zeros(
        (total_seq_len, num_q_heads, q_head_dim), dtype=DType.bfloat16
    )
    k_buffer_tensor = Tensor.zeros(
        (total_seq_len, num_q_heads, k_head_dim), dtype=DType.bfloat16
    )
    v_buffer_tensor = Tensor.zeros(
        (total_seq_len, num_q_heads, k_head_dim), dtype=DType.bfloat16
    )

    result = model.execute(
        input_tensor.to(cuda),
        input_row_offsets.to(cuda),
        k_buffer_tensor.to(cuda),
        v_buffer_tensor.to(cuda),
        blocks.to(cuda),
        cache_lengths.to(cuda),
        lookup_table_tensor.to(cuda),
        is_cache_empty_buf,
    )[0]
    assert isinstance(result, Tensor)

    host = CPU(0)
    assert np.all(result.to(host).to_numpy() != np.inf)
    assert np.all(result.to(host).to_numpy() != np.nan)


def causal_max_flash_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> np.ndarray:
    assert q.dtype == torch.bfloat16
    dtype = DType.bfloat16
    batch, q_seq_len, nheads, head_dim = q.shape

    # Graph types.
    q_type = TensorType(
        dtype,
        shape=["batch", "q_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", "kv_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    # Stage ops.

    # Construct and compile the MAX graph flash attention.
    graph = Graph(
        "flash_attn",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
        ),
        input_types=[
            q_type,
            kv_type,
            kv_type,
        ],
    )

    # Compile model.
    model = session.load(graph)

    # Execute.
    return torch.from_dlpack(
        cast(
            Tensor,
            model.execute(q.detach(), k.detach(), v.detach())[0],
        )
    )


@pytest.mark.parametrize(
    "q_seqlen,k_seqlen",
    [
        (128, 128),
        # TODO(KERN-1634): support num_keys != seq_len.
        # (2, 3),
    ],
)
def test_causal_flash_attention_gpu(q_seqlen: int, k_seqlen: int) -> None:
    dtype = DType.bfloat16
    head_dim = 128
    batch_size = 1
    nheads = 6
    nheads_k = 6
    torch_device = "cuda"
    torch_dtype = torch.bfloat16

    q_shape = (batch_size, q_seqlen, nheads, head_dim)
    kv_shape = (batch_size, k_seqlen, nheads_k, head_dim)

    # Set seed.
    torch.random.manual_seed(42)

    q = torch.randn(q_shape, device=torch_device, dtype=torch_dtype)
    k = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)
    v = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)

    out_max = causal_max_flash_attn(q, k, v).squeeze()
    out_flash_attn = (
        scaled_dot_product_attention(
            q.to(torch_device).permute(0, 2, 1, 3),
            k.to(torch_device).permute(0, 2, 1, 3),
            v.to(torch_device).permute(0, 2, 1, 3),
            is_causal=True,
            scale=math.sqrt(1.0 / head_dim),
        )
        .permute(0, 2, 1, 3)
        .squeeze()
    )

    torch.testing.assert_close(out_max, out_flash_attn, rtol=1e-2, atol=2e-2)


def null_mask_max_flash_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> np.ndarray:
    assert q.dtype == torch.bfloat16
    dtype = DType.bfloat16
    batch, q_seq_len, nheads, head_dim = q.shape

    # Graph types.
    q_type = TensorType(
        dtype,
        shape=["batch", "q_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", "kv_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    # Stage ops.

    # Construct and compile the MAX graph flash attention.
    graph = Graph(
        "flash_attn",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=MHAMaskVariant.NULL_MASK,
        ),
        input_types=[
            q_type,
            kv_type,
            kv_type,
        ],
    )

    # Compile model.
    model = session.load(graph)

    # Execute.
    return torch.from_dlpack(
        cast(
            Tensor,
            model.execute(q.detach(), k.detach(), v.detach())[0],
        )
    )


@pytest.mark.parametrize(
    "q_seqlen,k_seqlen",
    [
        (20, 20),
        (128, 128),
        # TODO(KERN-1634): support num_keys != seq_len.
        # (2, 3),
    ],
)
def test_null_mask_flash_attention_gpu(q_seqlen: int, k_seqlen: int) -> None:
    dtype = DType.bfloat16
    head_dim = 128
    batch_size = 1
    nheads = 6
    nheads_k = 6
    torch_device = "cuda"
    torch_dtype = torch.bfloat16

    q_shape = (batch_size, q_seqlen, nheads, head_dim)
    kv_shape = (batch_size, k_seqlen, nheads_k, head_dim)

    # Set seed.
    torch.random.manual_seed(42)

    q = torch.randn(q_shape, device=torch_device, dtype=torch_dtype)
    k = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)
    v = torch.randn(kv_shape, device=torch_device, dtype=torch_dtype)

    out_max = null_mask_max_flash_attn(q, k, v).squeeze()

    out_flash_attn = (
        scaled_dot_product_attention(
            q.to(torch_device).permute(0, 2, 1, 3),
            k.to(torch_device).permute(0, 2, 1, 3),
            v.to(torch_device).permute(0, 2, 1, 3),
            attn_mask=None,
            is_causal=False,
            scale=math.sqrt(1.0 / head_dim),
        )
        .permute(0, 2, 1, 3)
        .squeeze()
    )

    torch.testing.assert_close(out_max, out_flash_attn, rtol=1e-2, atol=2e-2)


def padded_max_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_length: torch.Tensor,
) -> torch.Tensor:
    assert q.dtype == torch.bfloat16
    dtype = DType.bfloat16
    batch, q_seq_len, nheads, head_dim = q.shape

    # Graph types.
    q_type = TensorType(
        dtype,
        shape=["batch", "q_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        dtype,
        shape=["batch", "kv_seq_len", nheads, head_dim],
        device=DeviceRef.GPU(),
    )
    valid_length_type = TensorType(
        DType.uint32,
        shape=["batch"],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    # Construct and compile the MAX graph flash attention.
    def construct() -> Graph:
        with Graph(
            "padded_flash_attn",
            input_types=[
                q_type,
                kv_type,
                kv_type,
                valid_length_type,
            ],
        ) as g:
            assert are_all_tensor_values(g.inputs)
            q, k, v, valid_length = g.inputs

            result = flash_attention_gpu(
                q,
                k,
                v,
                scale=math.sqrt(1.0 / head_dim),
                mask_variant=MHAMaskVariant.NULL_MASK,
                valid_length=valid_length,
            )
            g.output(result)
        return g

    graph = construct()

    # Compile model.
    model = session.load(graph)

    # Execute.
    return torch.from_dlpack(
        cast(
            Tensor,
            model.execute(
                q.detach(), k.detach(), v.detach(), valid_length.detach()
            )[0],
        )
    )


@pytest.mark.parametrize(
    "actual_seq_len, padded_seq_len",
    [
        (128, 160),
        (20, 128),
    ],
)
def test_padded_flash_attention_gpu(
    actual_seq_len: int, padded_seq_len: int
) -> None:
    dtype = DType.bfloat16
    head_dim = 128
    batch_size = 1
    nheads = 1
    torch_device = "cuda"
    torch_dtype = torch.bfloat16

    actual_shape = (batch_size, actual_seq_len, nheads, head_dim)
    padded_shape = (batch_size, padded_seq_len, nheads, head_dim)

    torch.random.manual_seed(42)

    q_actual = torch.randn(actual_shape, dtype=torch_dtype, device=torch_device)
    k_actual = torch.randn(actual_shape, dtype=torch_dtype, device=torch_device)
    v_actual = torch.randn(actual_shape, dtype=torch_dtype, device=torch_device)

    q_padded = torch.ones(padded_shape, dtype=torch_dtype, device=torch_device)
    k_padded = torch.ones(padded_shape, dtype=torch_dtype, device=torch_device)
    v_padded = torch.ones(padded_shape, dtype=torch_dtype, device=torch_device)

    # Copy actual data to the beginning of padded tensors
    q_padded[:, :actual_seq_len, :, :] = q_actual
    k_padded[:, :actual_seq_len, :, :] = k_actual
    v_padded[:, :actual_seq_len, :, :] = v_actual

    # Run null mask flash attention on actual length data
    out_null_mask = null_mask_max_flash_attn(q_actual, k_actual, v_actual)

    # Run padded flash attention on padded data with valid_length
    valid_length = torch.tensor(
        [actual_seq_len], dtype=torch.uint32, device=torch_device
    )
    out_padded = padded_max_flash_attn(
        q_padded, k_padded, v_padded, valid_length
    )

    # Compare only the first actual_seq_len positions
    out_padded_trimmed = out_padded[:, :actual_seq_len]

    # Assert that the results are close
    torch.testing.assert_close(
        out_null_mask, out_padded_trimmed, rtol=1e-2, atol=2e-2
    )
