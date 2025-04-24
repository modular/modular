# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import torch
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.pipelines.architectures.gemma3.layers.rms_norm import (
    Gemma3RMSNorm as MaxRMSNorm,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3RMSNorm as TorchRMSNorm,
)


def generate_torch_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> torch.Tensor:
    layer = TorchRMSNorm(
        dim=text_config.hidden_size,
        eps=1e-6,
    ).to(dtype=torch.bfloat16, device="cuda")
    layer.weight.data = rms_weight.to(dtype=torch.float32, device="cuda")
    return layer(input_tensor.to("cuda"))


def generate_max_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> torch.Tensor:
    layer = MaxRMSNorm(
        dim=text_config.hidden_size,
        eps=1e-6,
    )
    state_dict = {"weight": rms_weight.cpu()}
    layer.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator()])
    graph = Graph(
        "Gemma3RMSNorm",
        layer,
        input_types=(
            TensorType(
                dtype=DType.bfloat16,
                shape=Shape(input_tensor.shape),
                device=DeviceRef.GPU(),
            ),
        ),
    )

    compiled = session.load(graph, weights_registry=state_dict)
    return from_dlpack(compiled.execute(input_tensor.to("cuda"))[0]).to(
        torch.bfloat16
    )


def test_gemma3_rms_norm(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> None:
    """Test `Gemma3RMSNorm` against HuggingFace implementation."""
    torch_output = generate_torch_outputs(text_config, input_tensor, rms_weight)
    max_output = generate_max_outputs(text_config, input_tensor, rms_weight)
    # Note: This test uses bfloat16, which has limited precision (only ~2–3
    # decimal digits). Small differences (e.g. 0.03125 or 0.0625) are expected
    # and can arise from rounding during intermediate steps like `rsqrt` or
    # `x^2`. Because of this, we use relaxed tolerances.
    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
