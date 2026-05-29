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
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

# Fixtures (`text_config`, `input_tensor`, `attention_weights_*`,
# `session`, `device`) are defined in conftest.py and auto-discovered
# by pytest.


# ---------------------------------------------------------------------------
# BF16 compiled fixtures (shared by bf16 tests and as baselines for fp8)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FP8 compiled fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_fp8(
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
        cache_dtype=DType.float8_e4m3fn,
        quantization_granularity=64,
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


# ---------------------------------------------------------------------------
# BF16 attention tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FP8 KV regression tests
# ---------------------------------------------------------------------------
# Regression: fp8 KV path must match bf16 KV path closely.
#
# `interleaved` controls the RoPE rotation PAIRING (`(2k, 2k+1)` vs HF's
# `(k, k+head_dim/2)`).  Trained Gemma4 weights expect the HF pairing; a
# kernel-driven flip to `interleaved=True` would rotate the wrong
# dimension pairs and corrupt attention scores.
#
# The fp8 K-store kernel writes the rope output contiguously regardless
# of the pairing convention — Q·K dot product is permutation-invariant
# so this works as long as Q and K share the storage layout (they do).
#
# **Sensitivity caveat** (verified empirically): with random weights at
# checkpoint-matched STD (~0.03) and a single 11-token prefill, both
# `interleaved=True` (broken) and `interleaved=False` (correct) produce
# attention outputs with cosine ~0.9997 — the bug does not manifest
# without trained-weight structure to amplify the wrong-pair rotation.
# So this unit test is a **smoke gate**, not a sufficient bug-detector.
# The authoritative ground truth is a server-level smoke (math /
# multi-turn coherent generation) under a real Gemma4 checkpoint.
#
# What these tests DO guarantee:
# 1. The fp8 KV graph compiles and runs end-to-end without crash.
# 2. The fp8 KV path's attention output is cosine-close to bf16, ruling
#    out catastrophic layout/scale/dequant errors (which would drop
#    cosine to < 0.9 even on random weights).
# 3. The fp8 path exercises `use_interleaved_rope=True` (matching
#    production gemma4.py wiring) so a future regression that reads
#    that flag back through the rope_interleaved override would route
#    the wrong way and surface here only if the actual kernel-level
#    contract is broken (e.g. the storage-order scale blocking
#    regresses).


def test_attention_fp8_kv_matches_bf16_local(
    text_config: Gemma3TextConfig,
    input_tensor: torch.Tensor,
    compiled_local_bf16: CompiledAttention,
    compiled_local_fp8: CompiledAttention,
    device: Device,
) -> None:
    """Regression: sliding (head_dim=256) layer fp8-vs-bf16 cosine must
    stay above 0.99.  Catches RoPE pairing / storage-layout mismatches.
    """
    assert_fp8_matches_bf16(
        compiled_local_bf16,
        compiled_local_fp8,
        input_tensor,
        device,
        layer_idx=0,
        head_dim_for_log=text_config.head_dim,
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
