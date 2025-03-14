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
from max.pipelines.architectures.deepseekV2.layers.yarn_rope import (
    YarnRotaryEmbedding,
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
    return torch_cos, torch_sin


def generate_max_outputs(
    config: DeepseekV2Config,
    input_tensor_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "YarnRope",
        YarnRotaryEmbedding(config.qk_rope_head_dim),
        input_types=(
            TensorType(
                DType.bfloat16,
                (Shape(input_tensor_rope.shape)),
            ),
        ),
    )

    compiled = session.load(graph)
    max_output = compiled.execute(input_tensor_rope)
    max_cos = from_dlpack(max_output[0]).to(torch.bfloat16)
    max_sin = from_dlpack(max_output[1]).to(torch.bfloat16)
    return max_cos, max_sin


def test_yarn_rope(
    config: DeepseekV2Config,
    input_tensor_rope: torch.Tensor,
) -> None:
    torch_cos, torch_sin = generate_torch_outputs(config, input_tensor_rope)
    max_cos, max_sin = generate_max_outputs(config, input_tensor_rope)

    # TODO (MODELS-396): These tolerances are likely too permissive. This should be revisited and adjusted if needed when the model is E2E validated.
    torch.testing.assert_close(
        torch_cos,
        max_cos,
        rtol=2e-2,
        atol=2e-2,
    )

    torch.testing.assert_close(
        torch_sin,
        max_sin,
        rtol=2e-2,
        atol=2e-2,
    )
