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

"""Harness tests for Gemma3Attention (google/gemma-3-1b-it config)."""

from __future__ import annotations

import pytest
from max.interfaces import TextGenerationContext
from testbed.harnesses.gemma3_attention import (
    Gemma3AttentionHarness,
    Gemma3AttentionStaticParams,
)
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner, create_session

# google/gemma-3-1b-it config
_STATIC_PARAMS = Gemma3AttentionStaticParams(
    hidden_size=2304,
    n_heads=8,
    n_kv_heads=4,
    head_dim=256,
    max_seq_len=32768,
    rope_theta=1000000.0,
    qk_norm_eps=1e-6,
    sliding_window_pattern=6,
    local_window_size=1024,
    # layer_idx=5 => (5+1) % 6 == 0 => global/causal attention (no sliding).
    layer_idx=5,
)

_SMOKE_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=8192, ctx_len=8192),
    AttentionDynamicParams(batch_size=1, seq_len=1, ctx_len=8192),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    Gemma3AttentionStaticParams,
    AttentionDynamicParams,
    list[TextGenerationContext],
]:
    session, device = create_session()
    return LayerTestRunner(
        Gemma3AttentionHarness(_STATIC_PARAMS, session, device)
    )


def test_benchmark_smoke(
    runner: LayerTestRunner[
        Gemma3AttentionStaticParams,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0


def test_correctness(
    runner: LayerTestRunner[
        Gemma3AttentionStaticParams,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ],
) -> None:
    # Correctness only works for prefill (ctx_len=0), batch_size=1.
    shapes = [
        AttentionDynamicParams(batch_size=1, seq_len=11),
        AttentionDynamicParams(batch_size=1, seq_len=128),
    ]
    results = runner.correctness(
        shapes, atol=0.0625, rtol=0.016, cos_threshold=0.001
    )
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"
