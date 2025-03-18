# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, Shape, TensorType
from max.nn.attention.attention_with_rope import LatentAttentionWithRope
from max.nn.rotary_embedding import OptimizedRotaryEmbedding
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import DeepseekV2Config
from torch_reference.modeling_deepseek import DeepseekV2Attention


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = DeepseekV2Attention(config=config, layer_idx=0).to(torch.bfloat16)
    torch_output = layer(
        input_tensor,
        attention_mask=attention_mask,
        seq_len=input_tensor.shape[2],
    )
    return torch_output


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    rope = OptimizedRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        n_heads=config.num_attention_heads,
        theta=config.num_rope_theta,
        max_seq_len=config.max_position_embeddings,
    )

    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.v_head_dim,
        cache_strategy=KVCacheStrategy.CONTINUOUS,
        n_devices=1,
    )

    kv_manager = load_kv_manager(
        params=kv_params,
        max_batch_size=1,
        max_seq_len=config.max_position_embeddings,
        num_layers=config.num_hidden_layers,
        devices=[Accelerator(0)],
        session=session,
    )

    # Fetch
    fetch_op = FetchContinuousBatchingKVCacheCollection(kv_params)
    kv_inputs_all = kv_manager.input_symbols()
    kv_input_types = [
        inp for device_inputs in kv_inputs_all for inp in device_inputs
    ]  # flatten list of tuples to list of elements

    with Graph(
        "LatentAttentionWithRope",
        input_types=(
            TensorType(
                DType.bfloat16,
                (Shape(input_tensor.shape)),
            ),
            *kv_input_types,
        ),
    ) as graph:
        kv_collection = fetch_op(*[v.tensor for v in graph.inputs[1:]])
        latent_attention = LatentAttentionWithRope(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=kv_params,
            layer_idx=0,
            dtype=DType.bfloat16,
        )
        graph.output(latent_attention(graph.inputs[0].tensor, kv_collection))

    compiled = session.load(graph)
    max_output = compiled.execute(input_tensor)
    return from_dlpack(max_output).to(torch.bfloat16)


@pytest.mark.skip(reason="E2EOPT-44: MLA kernel is not implemented")
def test_latent_attention(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    torch_output = generate_torch_outputs(config, input_tensor, attention_mask)
    max_output = generate_max_outputs(config, input_tensor)

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=2e-2,
        atol=2e-2,
    )
