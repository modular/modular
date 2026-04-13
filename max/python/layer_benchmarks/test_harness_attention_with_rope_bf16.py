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

"""Harness tests for AttentionWithRope (bf16)."""

from __future__ import annotations

import pytest
from max.interfaces import TextGenerationContext
from testbed.harnesses.attention_with_rope import (
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams,
)
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner, create_session

_STATIC_PARAMS = AttentionWithRopeStaticParams(
    hidden_size=4096,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    max_seq_len=65536,
    rope_theta=500000.0,
    dtype="bf16",
)

_SMOKE_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=8192, ctx_len=8192),
    AttentionDynamicParams(batch_size=1, seq_len=1, ctx_len=8192),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    AttentionWithRopeStaticParams,
    AttentionDynamicParams,
    list[TextGenerationContext],
]:
    session, device = create_session()
    return LayerTestRunner(
        AttentionWithRopeHarness(_STATIC_PARAMS, session, device)
    )


def test_benchmark_smoke(
    runner: LayerTestRunner[
        AttentionWithRopeStaticParams,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0


def test_correctness(
    runner: LayerTestRunner[
        AttentionWithRopeStaticParams,
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
