# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch
from context_utils import create_text_context
from max._core.engine import PrintStyle
from max.driver import Accelerator, Tensor, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
from max.nn.attention.attention_with_rope import LatentAttentionWithRope
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
)
from max.nn.rotary_embedding import OptimizedRotaryEmbedding
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import DeepseekV2Config
from torch_reference.modeling_deepseek import DeepseekV2Attention


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    layer = DeepseekV2Attention(config=config, layer_idx=0).to(torch.bfloat16)
    layer.load_state_dict(attention_weights)

    torch_output = layer(
        input_tensor,
        attention_mask=attention_mask,
    )
    return torch_output[0]


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    use_prefill: bool = True,
) -> torch.Tensor:
    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    rope = OptimizedRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.rope_theta,
        max_seq_len=config.max_position_embeddings,
    )

    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        cache_strategy=KVCacheStrategy.PAGED,
        n_devices=1,
        page_size=128,
    )

    latent_attention = LatentAttentionWithRope(
        rope=rope,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_size=config.hidden_size,
        kv_params=kv_params,
        layer_idx=0,
        dtype=DType.bfloat16,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
    )
    latent_attention.load_state_dict(attention_weights)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        cache_memory=1024 * 1024 * 1024,
        page_size=128,
        max_batch_size=1,
        max_seq_len=config.max_position_embeddings,
        num_layers=config.num_hidden_layers,
        devices=[Accelerator(0)],
        session=session,
    )

    # Fetch
    fetch_op = FetchPagedKVCacheCollection(kv_params)

    # Set input types for the graph.
    hidden_state_type = TensorType(
        DType.bfloat16, ["total_seq_len", config.hidden_size]
    )
    input_row_offsets_type = TensorType(DType.uint32, ["input_row_offsets_len"])

    def construct() -> Graph:
        with Graph(
            "LatentAttentionWithRope",
            input_types=(
                hidden_state_type,
                input_row_offsets_type,
                *kv_manager.input_symbols()[0],
            ),
        ) as graph:
            hidden_states = graph.inputs[0].tensor
            input_row_offsets = graph.inputs[1].tensor
            kv_collection = fetch_op(*[v.tensor for v in graph.inputs[2:]])

            result = latent_attention(
                hidden_states,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )
            graph.output(result)
        return graph

    g = construct()

    compiled = session.load(g, weights_registry=latent_attention.state_dict())

    batch_size = 1
    total_tokens = input_tensor.shape[1]
    prompt_lens = [total_tokens] if use_prefill else [1]

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

    if not use_prefill:
        # for MLA, we actually run different graphs for max_seq_len = 1 and
        # max_seq_len > 1, In this case, we loop through the tokens to run
        # the decode graph.
        all_outputs = []
        for tok_idx in range(total_tokens):
            # prepare current token's inputs
            fetch_args = kv_manager.fetch(batch)[0]
            input_tensor_device = (
                Tensor.from_numpy(
                    input_tensor[:, tok_idx, :].view(torch.float16).numpy()
                )
                .view(DType.bfloat16)
                .to(device0)
            )

            max_output = compiled.execute(
                input_tensor_device, input_row_offsets.to(device0), *fetch_args
            )

            for ctx in batch:
                # update the context a dummy token
                ctx.update(42)

            kv_manager.step(batch)
            torch_output = from_dlpack(max_output[0]).to(torch.bfloat16)
            all_outputs.append(torch_output[:, None, :].to("cpu"))

        return torch.concat(all_outputs, dim=1)

    else:
        fetch_args = kv_manager.fetch(batch)[0]
        input_tensor_device = (
            Tensor.from_numpy(input_tensor[0, :, :].view(torch.float16).numpy())
            .view(DType.bfloat16)
            .to(device0)
        )

        max_output = compiled.execute(
            input_tensor_device, input_row_offsets.to(device0), *fetch_args
        )

        torch_output = from_dlpack(max_output[0]).to(torch.bfloat16).to("cpu")

        return torch_output[None, :, :]


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_latent_attention_prefill(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    max_output = generate_max_outputs(
        config, input_tensor, attention_weights, use_prefill=True
    )

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="MLA kernel only supports Nvidia GPUs"
)
def test_latent_attention_decode(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
) -> None:
    torch_output = generate_torch_outputs(
        config, input_tensor, attention_mask, attention_weights
    )
    max_output = generate_max_outputs(
        config, input_tensor, attention_weights, use_prefill=False
    )

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=1e-2,
        atol=1e-2,
    )
