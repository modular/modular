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
"""fp8-KV Gemma4 attention regression test — full (global) layer.

Split into its own bazel target (separate from the sliding-layer fp8
test and the bf16 tests) so each unique pair of compiles runs in
parallel on CI. See `test_attention_fp8_local.py` for the regression
rationale and `_attention_helpers.py` for shared fixtures and helpers.
"""

import pytest
import torch
from _attention_helpers import (  # type: ignore[import-not-found]
    MAX_DTYPE,
    CompiledAttention,
    assert_fp8_matches_bf16,
    build_max_attention,
)
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

# Fixtures (`text_config`, `input_tensor`, `attention_weights_global`,
# `session`, `device`) are defined in conftest.py and auto-discovered
# by pytest.


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


@pytest.fixture(scope="module")
def compiled_global_fp8(
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
        cache_dtype=DType.float8_e4m3fn,
        quantization_granularity=64,
    )


def test_attention_fp8_kv_matches_bf16_global(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    compiled_global_bf16: CompiledAttention,
    compiled_global_fp8: CompiledAttention,
    device: Device,
) -> None:
    """Regression: global (head_dim=512) layer fp8-vs-bf16 cosine must
    stay above 0.99.  Catches RoPE pairing / storage-layout mismatches.
    """
    assert_fp8_matches_bf16(
        compiled_global_bf16,
        compiled_global_fp8,
        input_tensor,
        device,
        layer_idx=5,
        head_dim_for_log=text_config.global_head_dim,
    )
