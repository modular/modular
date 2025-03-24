# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import torch
from max._core.engine import PrintStyle
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, Shape, TensorType
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)
from torch_reference.modeling_deepseek import DeepseekV2YarnRotaryEmbedding


def generate_torch_outputs(
    config: DeepseekV2Config,
    input_tensor_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {}
    scaling_factor = 1.0  # default value
    if config.rope_scaling is not None:
        scaling_factor = config.rope_scaling["factor"]
        kwargs = {
            key: config.rope_scaling[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in config.rope_scaling
        }
    layer = DeepseekV2YarnRotaryEmbedding(
        dim=config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        scaling_factor=scaling_factor,
        base=int(config.rope_theta),
        **kwargs,
    ).to(torch.bfloat16)
    torch_output = layer(input_tensor_rope, seq_len=input_tensor_rope.shape[2])
    torch_cos = torch_output[0].to(torch.bfloat16)
    torch_sin = torch_output[1].to(torch.bfloat16)
    return torch.stack([torch_cos, torch_sin], axis=0)


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaling_params = DeepseekYarnRopeScalingParams(
        scaling_factor=config.rope_scaling["factor"],
        original_max_position_embeddings=config.rope_scaling[
            "original_max_position_embeddings"
        ],
        beta_fast=config.rope_scaling["beta_fast"],
        beta_slow=config.rope_scaling["beta_slow"],
        mscale=config.rope_scaling["mscale"],
        mscale_all_dim=config.rope_scaling["mscale_all_dim"],
    )
    session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "YarnRope",
        DeepseekYarnRotaryEmbedding(
            config.qk_rope_head_dim,
            theta=config.rope_theta,
            n_heads=config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
        ),
        input_types=(
            TensorType(
                DType.bfloat16,
                (Shape(input_tensor_rope.shape)),
            ),
        ),
    )

    compiled = session.load(graph)
    max_output = compiled.execute(input_tensor_rope)
    return from_dlpack(max_output[0]).to(torch.bfloat16)


def test_yarn_rope(
    config: DeepseekV2Config,
    input_tensor_rope: torch.Tensor,
) -> None:
    torch_output = generate_torch_outputs(config, input_tensor_rope)
    max_output = generate_max_outputs(config, input_tensor_rope)

    # TODO (MODELS-396): These tolerances are likely too permissive. This should be revisited and adjusted if needed when the model is E2E validated.
    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=2e-2,
        atol=2e-2,
    )
