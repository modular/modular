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

"""Tests for TPOT computation in calculate_metrics."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

from max.benchmark.benchmark_serving import calculate_metrics
from max.benchmark.benchmark_shared.request import RequestFuncOutput


def _make_mock_tokenizer(token_counts: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer that returns specified token counts.

    Args:
        token_counts: Mapping from generated text to the number of tokens.
    """
    tokenizer = MagicMock()

    def side_effect(text: str, add_special_tokens: bool = True) -> MagicMock:
        result = MagicMock()
        result.input_ids = list(range(token_counts.get(text, 0)))
        return result

    tokenizer.side_effect = side_effect
    return tokenizer


def test_per_chunk_tpot_collected_from_outputs() -> None:
    """Per-chunk TPOT values are correctly collected from outputs."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello world",
        itl=[0.1, 0.2, 0.3],
        tpot=[0.05, 0.1, 0.15],
    )

    tokenizer = _make_mock_tokenizer({"hello world": 5})

    metrics, _ = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics={},
        skip_first_n_requests=0,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    # TPOT percentiles should be based on the per-chunk values [0.05, 0.1, 0.15]
    # scaled by 1000 (to ms)
    assert math.isclose(metrics.tpot_ms.median, 100.0, rel_tol=1e-3)


def test_tpot_weighted_mean() -> None:
    """TPOT mean = sum(ITL) / decode_tokens * 1000 ms."""
    # Request 1: 10 output tokens, ITL sum = 0.9s
    output1 = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="ten tokens out",
        itl=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        tpot=[0.1] * 9,
    )

    # Request 2: 4 output tokens, ITL sum = 0.6s
    output2 = RequestFuncOutput(
        success=True,
        latency=0.8,
        ttft=0.2,
        prompt_len=5,
        generated_text="four tok",
        itl=[0.2, 0.2, 0.2],
        tpot=[0.2] * 3,
    )

    # Mock tokenizer: output1 -> 10 tokens, output2 -> 4 tokens
    tokenizer = _make_mock_tokenizer({"ten tokens out": 10, "four tok": 4})

    metrics, _ = calculate_metrics(
        outputs=[output1, output2],
        dur_s=2.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics={},
        skip_first_n_requests=0,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    # total_output = 10 + 4 = 14
    # completed = 2
    # decode_tokens = 14 - 2 = 12
    # sum(itl) = 0.9 + 0.6 = 1.5
    # weighted mean TPOT = 1.5 / 12 * 1000 = 125.0 ms
    expected_mean = 1.5 / 12 * 1000.0
    assert math.isclose(metrics.tpot_ms.mean, expected_mean, rel_tol=1e-6)


def test_tpot_zero_decode_tokens() -> None:
    """When all requests produce <= 1 token, TPOT mean is NaN."""
    # Output with 1 token (only TTFT, no decode)
    output = RequestFuncOutput(
        success=True,
        latency=0.1,
        ttft=0.1,
        prompt_len=10,
        generated_text="a",
        itl=[],
        tpot=[],
    )

    # 1 output token, 1 completed -> decode_tokens = 0
    tokenizer = _make_mock_tokenizer({"a": 1})

    metrics, _ = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics={},
        skip_first_n_requests=0,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    # With empty tpots, StandardPercentileMetrics gets [nan], so mean is nan
    assert math.isnan(metrics.tpot_ms.mean)


def test_empty_outputs_no_crash() -> None:
    """Empty outputs list doesn't crash."""
    tokenizer = _make_mock_tokenizer({})

    metrics, actual_output_lens = calculate_metrics(
        outputs=[],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics={},
        skip_first_n_requests=0,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    assert metrics.completed == 0
    assert actual_output_lens == []
    # TPOT mean should be NaN since there are no outputs
    assert math.isnan(metrics.tpot_ms.mean)


def test_itl_metrics_unchanged() -> None:
    """ITL metrics remain unchanged by the TPOT refactor."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello world",
        itl=[0.1, 0.2, 0.3],
        tpot=[0.05, 0.1, 0.15],
    )

    tokenizer = _make_mock_tokenizer({"hello world": 5})

    metrics, _ = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics={},
        skip_first_n_requests=0,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    # ITL should be computed from the raw itl values [0.1, 0.2, 0.3] * 1000
    assert math.isclose(metrics.itl_ms.mean, 200.0, rel_tol=1e-3)
    assert math.isclose(metrics.itl_ms.median, 200.0, rel_tol=1e-3)


def test_failed_requests_excluded() -> None:
    """Failed requests don't contribute to TPOT."""
    success_output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1, 0.2],
        tpot=[0.1, 0.2],
    )
    failed_output = RequestFuncOutput(
        success=False,
        error="test error",
        itl=[999.0],
        tpot=[999.0],
    )

    tokenizer = _make_mock_tokenizer({"hello": 3, "": 0})

    metrics, _ = calculate_metrics(
        outputs=[success_output, failed_output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics={},
        skip_first_n_requests=0,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    # Only successful request's TPOT should be used
    assert metrics.completed == 1
    assert metrics.failures == 1
    # TPOT values should only include [0.1, 0.2], not [999.0]
    assert metrics.tpot_ms.median < 500.0
