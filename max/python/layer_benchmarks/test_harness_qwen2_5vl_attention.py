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

"""Harness tests for Qwen2.5VL decoder attention (Qwen2.5-VL-3B-like config)."""

from __future__ import annotations

import pytest
from max.interfaces import TextGenerationContext
from testbed.harnesses.qwen2_5vl_attention import (
    Qwen25VLAttentionHarness,
    Qwen25VLAttentionStaticParams,
)
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner, create_session

# Qwen2.5-VL 3B config values.
_STATIC_PARAMS = Qwen25VLAttentionStaticParams(
    hidden_size=2048,
    n_heads=16,
    n_kv_heads=2,
    head_dim=128,
    max_seq_len=1024,
    rope_theta=1000000.0,
    mrope_section=[16, 24, 24],
)

_SMOKE_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=11),
    AttentionDynamicParams(batch_size=1, seq_len=128),
    AttentionDynamicParams(batch_size=1, seq_len=1, ctx_len=128),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    Qwen25VLAttentionStaticParams,
    AttentionDynamicParams,
    list[TextGenerationContext],
]:
    session, device = create_session()
    return LayerTestRunner(
        Qwen25VLAttentionHarness(_STATIC_PARAMS, session, device)
    )


def test_benchmark_smoke(
    runner: LayerTestRunner[
        Qwen25VLAttentionStaticParams,
        AttentionDynamicParams,
        list[TextGenerationContext],
    ],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0
