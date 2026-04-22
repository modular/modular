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

import pytest
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.nn.norm.rms_norm import RMSNorm
from torch.utils.dlpack import from_dlpack
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# TODO: Update integration test infra for model layers, then move this
# test to the location chosen for layer tests.


def generate_torch_outputs(
    config: LlamaConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> torch.Tensor:
    layer = LlamaRMSNorm(
        hidden_size=config.hidden_size,
        eps=config.rms_norm_eps,
    ).to(dtype=torch.bfloat16, device="cuda")

    layer.weight.data = rms_weight.to(dtype=torch.bfloat16, device="cuda")
    return layer(input_tensor.to("cuda")).to(torch.bfloat16)


class LlamaStyleRMSNorm(RMSNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multiply_before_cast = False


def generate_max_outputs(
    config: LlamaConfig, input_tensor: torch.Tensor, rms_weight: torch.Tensor
) -> torch.Tensor:
    layer = LlamaStyleRMSNorm(
        dim=config.hidden_size,
        dtype=DType.float32,
        eps=config.rms_norm_eps,
    )

    state_dict = {"weight": rms_weight.cpu()}
    layer.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator()])
    graph = Graph(
        "LlamaRMSNorm",
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
    out = compiled.execute(input_tensor.to("cuda"))
    return from_dlpack(out[0]).to(torch.bfloat16)


@pytest.fixture
def llama_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=2048,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def input_tensor(llama_config: LlamaConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 7, llama_config.hidden_size, dtype=torch.bfloat16)


@pytest.fixture
def rms_weight(llama_config: LlamaConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(llama_config.hidden_size, dtype=torch.float32)


def test_rms_norm(
    llama_config: LlamaConfig,
    input_tensor: torch.Tensor,
    rms_weight: torch.Tensor,
) -> None:
    """Test that RMSNorm matches LlamaRMSNorm."""
    torch_output = generate_torch_outputs(
        llama_config, input_tensor, rms_weight
    )
    max_output = generate_max_outputs(llama_config, input_tensor, rms_weight)

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=1e-3,
        atol=1e-3,
    )
