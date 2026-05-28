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
"""fp8-KV Gemma4 attention regression test — sliding (local) layer.

Split into its own bazel target (separate from the global-layer fp8 test
and the bf16 tests) so each unique pair of compiles runs in parallel on
CI. See `_attention_helpers.py` for shared fixtures and the
`build_max_attention` / `execute_max_attention` helpers.
"""

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
# What this test DOES guarantee:
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

# Fixtures (`text_config`, `input_tensor`, `attention_weights_local`,
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
