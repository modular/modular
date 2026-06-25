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
"""Gemma4 attention tests (bf16 and fp8-KV variants).

Uses Bazel test sharding (`shard_count`) to parallelize tests across CI
workers. With 4 tests and 4 shards, round-robin distribution assigns each
test to its own shard, so compilation happens in parallel.

Module-scoped fixtures ensure each unique graph compiles once per shard.
See `_attention_helpers.py` for build/execute helpers and `conftest.py`
for shared fixtures.
"""

import pytest
import torch
from _attention_helpers import (  # type: ignore[import-not-found]
    MAX_DTYPE,
    TORCH_DTYPE,
    CompiledAttention,
    assert_fp8_matches_bf16,
    build_max_attention,
    execute_max_attention,
    generate_torch_outputs,
)
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from test_common.graph_utils import is_b100_b200
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

# Fixtures (`text_config`, `input_tensor`, `attention_weights_*`,
# `session`, `device`) are defined in conftest.py and auto-discovered
# by pytest.


# ---------------------------------------------------------------------------
# BF16 compiled fixtures (shared by bf16 tests and as baselines for fp8)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
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
    text_config: Gemma4TextConfig,
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


# ---------------------------------------------------------------------------
# Native pure-fp8 compiled fixtures (no per-block scales, scale=1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_native_fp8(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
) -> CompiledAttention:
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")
    return build_max_attention(
        session,
        text_config,
        attention_weights_local,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=0,
        cache_dtype=DType.float8_e4m3fn,
    )


@pytest.fixture(scope="module")
def compiled_global_native_fp8(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
) -> CompiledAttention:
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")
    return build_max_attention(
        session,
        text_config,
        attention_weights_global,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=5,
        cache_dtype=DType.float8_e4m3fn,
    )


# ---------------------------------------------------------------------------
# BF16 attention tests
# ---------------------------------------------------------------------------


def test_attention_local(
    text_config: Gemma4TextConfig,
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
    text_config: Gemma4TextConfig,
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


# ---------------------------------------------------------------------------
# Native pure-fp8 regression tests
# ---------------------------------------------------------------------------
# These exercise the native pure-fp8 path: fp8 Q/K/V read directly from the
# paged cache, Q@K^T and P@V both raw fp8 MMAs at tensorwise scale=1, bf16
# output. Routes through `mo.rope_split_store.ragged.paged` (outputs roped Q
# as fp8) and `mo.mha.ragged.paged` (fp8 Q+KV in, bf16 out).
# Cosine vs the bf16 baseline must clear the same 0.99 smoke bar. Same
# sensitivity caveat: random-weight smoke gate, not a sufficient bug-detector
# — end-to-end serving accuracy (e.g. gsm8k under a real checkpoint) is the
# authoritative correctness gate.


def test_attention_native_fp8_matches_bf16_local(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    compiled_local_bf16: CompiledAttention,
    compiled_local_native_fp8: CompiledAttention,
    device: Device,
) -> None:
    """Native pure-fp8 (no scales) sliding (head_dim=256) layer vs bf16."""
    assert_fp8_matches_bf16(
        compiled_local_bf16,
        compiled_local_native_fp8,
        input_tensor,
        device,
        layer_idx=0,
        head_dim_for_log=text_config.head_dim,
    )


def test_attention_native_fp8_matches_bf16_global(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    compiled_global_bf16: CompiledAttention,
    compiled_global_native_fp8: CompiledAttention,
    device: Device,
) -> None:
    """Native pure-fp8 (no scales) global (head_dim=512) layer vs bf16."""
    assert_fp8_matches_bf16(
        compiled_global_bf16,
        compiled_global_native_fp8,
        input_tensor,
        device,
        layer_idx=5,
        head_dim_for_log=text_config.global_head_dim,
    )
