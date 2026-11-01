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
"""The serve half of the chat-encoder seam: outcome counts to a counter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from unittest.mock import MagicMock, call, patch

import pytest
from max.serve.pipelines._chat_encoder_stats import ChatEncoderOutcomesRecorder


class _CountingTokenizer:
    def __init__(self, *takes: Mapping[str, int]) -> None:
        self._takes = list(takes)

    def take_chat_encoder_outcomes(self) -> Mapping[str, int]:
        return self._takes.pop(0)


class _RaisingTokenizer:
    calls = 0

    def take_chat_encoder_outcomes(self) -> Mapping[str, int]:
        self.calls += 1
        raise RuntimeError("gone")


def _record(tokenizer: object, times: int = 1) -> MagicMock:
    metrics = MagicMock()
    with patch("max.serve.pipelines._chat_encoder_stats.METRICS", metrics):
        recorder = ChatEncoderOutcomesRecorder(tokenizer)
        for _ in range(times):
            recorder.record()
    return metrics


def test_each_outcome_publishes_its_count() -> None:
    metrics = _record(
        _CountingTokenizer({"custom": 3, "fallback": 1}, {"custom": 1}), times=2
    )
    assert metrics.tokenizer_chat_encoder_requests.call_args_list == [
        call(3, "custom"),
        call(1, "fallback"),
        call(1, "custom"),
    ]


def test_an_outcome_with_no_requests_publishes_nothing() -> None:
    metrics = _record(_CountingTokenizer({"custom": 0}))
    metrics.tokenizer_chat_encoder_requests.assert_not_called()


def test_a_tokenizer_without_the_capability_is_silent() -> None:
    metrics = _record(object())
    metrics.tokenizer_chat_encoder_requests.assert_not_called()


def test_a_failing_tokenizer_logs_once_and_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tokenizer = _RaisingTokenizer()
    with caplog.at_level(logging.WARNING, logger="max.serve"):
        metrics = _record(tokenizer, times=3)
    assert tokenizer.calls == 1
    assert caplog.text.count("publishing no chat-encoder metrics") == 1
    metrics.tokenizer_chat_encoder_requests.assert_not_called()
