# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


import torch
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.pipelines.architectures.gemma3.layers.rms_norm import (
    Gemma3RMSNorm as MaxRMSNorm,
)
from max.pipelines.architectures.gemma3.layers.rms_norm import (
    gemma3_rms_norm_fused_residual_add,
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
        dtype=DType.float32,
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


def generate_torch_fused_residual_add_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    first_rms_weight: torch.Tensor,
    second_rms_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_norm = TorchRMSNorm(
        dim=text_config.hidden_size,
        eps=1e-6,
    ).to(dtype=torch.bfloat16, device="cuda")
    second_norm = TorchRMSNorm(
        dim=text_config.hidden_size,
        eps=1e-6,
    ).to(dtype=torch.bfloat16, device="cuda")
    first_norm.weight.data = first_rms_weight.to(
        dtype=torch.float32, device="cuda"
    )
    second_norm.weight.data = second_rms_weight.to(
        dtype=torch.float32, device="cuda"
    )

    residual_output = residual_tensor.to("cuda") + first_norm(
        input_tensor.to("cuda")
    )
    fused_output = second_norm(residual_output)
    return fused_output, residual_output


def generate_max_fused_residual_add_outputs(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    first_rms_weight: torch.Tensor,
    second_rms_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_norm = MaxRMSNorm(
        dim=text_config.hidden_size,
        dtype=DType.float32,
        eps=1e-6,
    )
    second_norm = MaxRMSNorm(
        dim=text_config.hidden_size,
        dtype=DType.float32,
        eps=1e-6,
    )
    first_norm.weight.name = "first_rms_weight"
    second_norm.weight.name = "second_rms_weight"

    session = InferenceSession(devices=[Accelerator()])
    input_type = TensorType(
        dtype=DType.bfloat16,
        shape=Shape(input_tensor.shape),
        device=DeviceRef.GPU(),
    )
    residual_type = TensorType(
        dtype=DType.bfloat16,
        shape=Shape(residual_tensor.shape),
        device=DeviceRef.GPU(),
    )

    with Graph(
        "Gemma3RMSNormFusedResidualAdd",
        input_types=(input_type, residual_type),
    ) as graph:
        x, residual = graph.inputs
        graph_fused_output, graph_residual_output = (
            gemma3_rms_norm_fused_residual_add(
                x.tensor,
                residual.tensor,
                first_norm,
                second_norm,
            )
        )
        graph.output(graph_fused_output, graph_residual_output)

    compiled = session.load(
        graph,
        weights_registry={
            "first_rms_weight": first_rms_weight.cpu(),
            "second_rms_weight": second_rms_weight.cpu(),
        },
    )
    compiled_outputs = compiled.execute(
        input_tensor.to("cuda"),
        residual_tensor.to("cuda"),
    )
    return (
        from_dlpack(compiled_outputs[0]).to(torch.bfloat16),
        from_dlpack(compiled_outputs[1]).to(torch.bfloat16),
    )


def test_gemma3_rms_norm(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> None:
    """Test `Gemma3RMSNorm` against HuggingFace implementation."""
    torch_output = generate_torch_outputs(text_config, input_tensor, rms_weight)
    max_output = generate_max_outputs(text_config, input_tensor, rms_weight)
    # Note: This test uses bfloat16, which has limited precision (only ~2-3
    # decimal digits). Small differences (e.g. 0.03125 or 0.0625) are expected
    # and can arise from rounding during intermediate steps like `rsqrt` or
    # `x^2`. Because of this, we use relaxed tolerances.
    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )


def test_gemma3_rms_norm_fused_residual_add(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> None:
    torch.manual_seed(7)
    residual_tensor = torch.randn_like(input_tensor)
    second_rms_weight = torch.roll(rms_weight, shifts=17).contiguous()

    torch_output, torch_residual = generate_torch_fused_residual_add_outputs(
        text_config,
        input_tensor,
        residual_tensor,
        rms_weight,
        second_rms_weight,
    )
    max_output, max_residual = generate_max_fused_residual_add_outputs(
        text_config,
        input_tensor,
        residual_tensor,
        rms_weight,
        second_rms_weight,
    )

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
    torch.testing.assert_close(
        torch_residual,
        max_residual,
        rtol=2 * torch.finfo(torch.bfloat16).eps,
        atol=8 * torch.finfo(torch.bfloat16).eps,
    )
