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
"""bf16-KV Gemma4 attention tests.

The fp8-KV variants live in `test_attention_fp8.py` so the two bazel
targets compile and run in parallel — keeping the fp8 path's expensive
double-compile off the critical path of the bf16 cases.
"""

import pytest
import torch
from _attention_helpers import (  # type: ignore[import-not-found]
    MAX_DTYPE,
    TORCH_DTYPE,
    CompiledAttention,
    build_max_attention,
    execute_max_attention,
    generate_torch_outputs,
)
from max.driver import Device
from max.engine import InferenceSession
from max.graph import DeviceRef
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

# Fixtures (`text_config`, `input_tensor`, `attention_weights_*`,
# `session`, `device`) are defined in conftest.py and auto-discovered
# by pytest.


@pytest.fixture(scope="module")
def compiled_local_bf16(
    session: InferenceSession,
    text_config: Gemma3TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
) -> CompiledAttention:
    return build_max_attention(
        session,
        text_config,
        attention_weights_local,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=0,
    )


@pytest.fixture(scope="module")
def compiled_global_bf16(
    session: InferenceSession,
    text_config: Gemma3TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
) -> CompiledAttention:
    return build_max_attention(
        session,
        text_config,
        attention_weights_global,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=5,
    )


def test_attention_local(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_local: dict[str, torch.Tensor],
    compiled_local_bf16: CompiledAttention,
    device: Device,
) -> None:
    max_output = execute_max_attention(
        compiled_local_bf16, input_tensor, device
    )

    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights_local, layer_idx=0
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )


def test_attention_global(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_global: dict[str, torch.Tensor],
    compiled_global_bf16: CompiledAttention,
    device: Device,
) -> None:
    max_output = execute_max_attention(
        compiled_global_bf16, input_tensor, device
    )
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        attention_weights_global,
        layer_idx=5,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )
