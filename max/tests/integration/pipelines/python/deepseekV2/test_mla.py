# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
import torch
from max._core.engine import PrintStyle
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, Shape, TensorType
from max.pipelines.architectures.deepseekV2.layers.mla import DeepseekAttention
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_deepseek import (
    DeepseekV2Config,
)
from torch_reference.modeling_deepseek import DeepseekV2Attention


@pytest.fixture
def input_tensor(
    config: DeepseekV2Config, seq_len: int = 40, batch_size: int = 1
) -> torch.Tensor:
    torch.manual_seed(42)  # Set fixed seed for reproducibility
    return torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def attention_mask(
    config: DeepseekV2Config, seq_len: int = 40, batch_size: int = 1
) -> torch.Tensor:
    # TODO: This likely needs to be generated differently to produce a valid attention mask (MODELS-369).
    torch.manual_seed(42)  # Set fixed seed for reproducibility
    return torch.randn(
        1,
        batch_size,
        seq_len,
        seq_len,
        dtype=torch.bfloat16,
    )


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
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    session = InferenceSession()
    session.set_debug_print_options(style=PrintStyle.COMPACT)
    graph = Graph(
        "DeepseekAttention",
        DeepseekAttention(layer_idx=0),
        input_types=(
            TensorType(
                DType.bfloat16,
                (Shape(input_tensor.shape)),
            ),
            TensorType(
                DType.bfloat16,
                (Shape(attention_mask.shape)),
            ),
        ),
    )

    compiled = session.load(graph)
    max_output = compiled.execute(input_tensor)
    return from_dlpack(max_output).to(torch.bfloat16)


@pytest.mark.skip(reason="E2EOPT-44: MLA kernel is not implemented")
def test_mla(
    config: DeepseekV2Config,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    torch_output = generate_torch_outputs(config, input_tensor, attention_mask)
    max_output = generate_max_outputs(config, input_tensor, attention_mask)

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=2e-2,
        atol=2e-2,
    )
