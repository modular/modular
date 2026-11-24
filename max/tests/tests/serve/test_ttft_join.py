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

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import RequestID
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.telemetry.ttft_join import (
    announce_enabled,
    is_enabled,
    leftover_ms,
    log_kv_hold,
    log_ttft_join,
)


def test_leftover_subtracts_named_hops() -> None:
    leftover = leftover_ms(
        1000.0,
        100.0,
        {
            "admit_ms": 200.0,
            "disp_ms": 50.0,
            "span_ms": 300.0,
            "reply_ms": 10.0,
        },
    )
    assert leftover == 340.0


def test_leftover_subtracts_step2_hops() -> None:
    leftover = leftover_ms(
        1000.0,
        100.0,
        {
            "admit_ms": 50.0,
            "api_pre_tokenize_ms": 20.0,
            "api_submit_ms": 30.0,
            "mw_queue_wait_ms": 200.0,
        },
    )
    assert leftover == 600.0


def test_leftover_without_hops_is_server_minus_ipt() -> None:
    assert leftover_ms(500.0, 150.0, None) == 350.0


def test_leftover_ignores_handoff_marker() -> None:
    leftover = leftover_ms(
        1000.0,
        100.0,
        {"admit_ms": 200.0, "handoff": 1.0},
    )
    assert leftover == 700.0


def test_join_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_SERVE_TTFT_JOIN", raising=False)
    assert is_enabled() is False


def test_log_ttft_join_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "0")
    log_ttft_join(
        request_id="r1",
        server_ttft_ms=100.0,
        ipt_ms=10.0,
        hops={"admit_ms": 5.0},
    )
    captured = capsys.readouterr()
    assert "ttft_join" not in captured.err
    assert "ttft_join" not in captured.out


def test_log_ttft_join_prints_json_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    log_ttft_join(
        request_id="r1",
        server_ttft_ms=1000.0,
        ipt_ms=100.0,
        hops={"admit_ms": 200.0, "span_ms": 300.0},
    )
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["event"] == "ttft_join"
    assert payload["rid"] == "r1"
    assert payload["server_ttft_ms"] == 1000.0
    assert payload["ipt_ms"] == 100.0
    assert payload["admit_ms"] == 200.0
    assert payload["span_ms"] == 300.0
    assert payload["disp_ms"] is None
    assert payload["api_pre_tokenize_ms"] is None
    assert payload["api_submit_ms"] is None
    assert payload["mw_queue_wait_ms"] is None
    assert payload["skipped_n"] == 0
    assert payload["yield_gap_ms"] == 0.0
    assert payload["handoff"] is False
    assert payload["leftover_ms"] == 400.0


def test_log_ttft_join_prints_step4_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    log_ttft_join(
        request_id="r1",
        server_ttft_ms=1000.0,
        ipt_ms=100.0,
        hops={"admit_ms": 200.0, "handoff": 1.0},
        skipped_n=2,
        yield_gap_ms=2742.4,
    )
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["skipped_n"] == 2
    assert payload["yield_gap_ms"] == 2742.4
    assert payload["handoff"] is True
    assert payload["leftover_ms"] == 700.0


def test_announce_enabled_prints_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    announce_enabled()
    assert capsys.readouterr().err.strip() == "ttft_join enabled"


@pytest.mark.asyncio
async def test_next_token_chunk_prints_join_on_first_yield(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The serve first-chunk path, not a direct log_ttft_join call."""
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    request_id = RequestID(value="join-req")
    hops = {
        "admit_ms": 985.0,
        "disp_ms": 59.0,
        "span_ms": 974.0,
        "reply_ms": 5.0,
    }

    async def mock_stream(_stream_request_id: str, context: Any) -> Any:
        async def _gen() -> Any:
            yield (
                [
                    TextGenerationOutput(
                        request_id=request_id,
                        tokens=[101],
                        final_status=GenerationStatus.END_OF_SEQUENCE,
                        ttft_join_hops_ms=hops,
                    )
                ],
                0,
            )

        return _gen()

    mock_tokens = Mock()
    mock_tokens.prompt_length = 4
    mock_request = Mock(request_id=request_id, tools=None, timestamp_ns=0)
    mock_request.sampling_params.stop = []

    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(
        return_value=Mock(request_id=request_id, tokens=mock_tokens)
    )
    pipeline.tokenizer.decode = AsyncMock(return_value="x")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(return_value=None)

    with patch("max.serve.pipelines.llm.METRICS", MagicMock()):
        bound = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        chunks = [chunk async for chunk in await bound(mock_request)]

    assert len(chunks) == 1
    err = capsys.readouterr().err
    payload = json.loads(err.strip().splitlines()[-1])
    assert payload["event"] == "ttft_join"
    assert payload["rid"] == str(request_id)
    assert payload["admit_ms"] == 985.0
    assert payload["span_ms"] == 974.0
    assert payload["disp_ms"] == 59.0
    assert payload["reply_ms"] == 5.0
    assert payload["api_pre_tokenize_ms"] == 0.0
    assert payload["api_submit_ms"] >= 0.0
    assert payload["skipped_n"] == 0
    assert payload["yield_gap_ms"] >= 0.0
    assert payload["handoff"] is False


class _StripThenContent:
    """First stream() call is delimiter-only; later calls are content."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        return

    def will_reason_after_prompt(self, prompt: Any) -> bool:
        return True

    def stream(
        self, tokens: list[int], is_currently_reasoning: bool = True
    ) -> Any:
        self.calls += 1
        span = Mock()
        if self.calls == 1:
            span.extract_content = Mock(return_value=None)
            span.extract_reasoning = Mock(return_value=None)
        else:
            span.extract_content = Mock(return_value=tokens)
            span.extract_reasoning = Mock(return_value=None)
        return Mock(
            span=span,
            is_still_reasoning=self.calls == 1,
            reasoning_text_formatter=None,
        )


@pytest.mark.asyncio
async def test_next_token_chunk_keeps_hops_across_delimiter_skip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hops stamped on a skipped delimiter still join the first yield."""
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    request_id = RequestID(value="join-skip")
    hops = {
        "admit_ms": 6.0,
        "disp_ms": 100.0,
        "span_ms": 1170.0,
        "reply_ms": 0.1,
    }

    async def mock_stream(_stream_request_id: str, context: Any) -> Any:
        async def _gen() -> Any:
            yield (
                [
                    TextGenerationOutput(
                        request_id=request_id,
                        tokens=[1],
                        final_status=GenerationStatus.ACTIVE,
                        ttft_join_hops_ms=hops,
                    )
                ],
                0,
            )
            await asyncio.sleep(0.02)
            yield (
                [
                    TextGenerationOutput(
                        request_id=request_id,
                        tokens=[101],
                        final_status=GenerationStatus.END_OF_SEQUENCE,
                    )
                ],
                1,
            )

        return _gen()

    mock_tokens = Mock()
    mock_tokens.prompt_length = 4
    mock_request = Mock(request_id=request_id, tools=None, timestamp_ns=0)
    mock_request.sampling_params.stop = []

    pipeline = Mock()
    pipeline.tokenizer.new_context = AsyncMock(
        return_value=Mock(request_id=request_id, tokens=mock_tokens)
    )
    pipeline.tokenizer.decode = AsyncMock(return_value="x")
    pipeline.model_worker.stream = mock_stream
    pipeline.debug_logging = False
    pipeline._min_chunk_tokens = 1
    pipeline._reasoning_parser = AsyncMock(return_value=_StripThenContent())

    with patch("max.serve.pipelines.llm.METRICS", MagicMock()):
        bound = TokenGeneratorPipeline.next_token_chunk.__get__(
            pipeline, type(pipeline)
        )
        chunks = [chunk async for chunk in await bound(mock_request)]

    assert len(chunks) == 1
    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert payload["event"] == "ttft_join"
    assert payload["admit_ms"] == 6.0
    assert payload["disp_ms"] == 100.0
    assert payload["span_ms"] == 1170.0
    assert payload["reply_ms"] == 0.1
    assert payload["skipped_n"] == 1
    assert payload["yield_gap_ms"] >= 15.0
    assert payload["handoff"] is False


def test_echo_server_prints_probe_and_join_on_one_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_pipeline_config: PipelineConfig,
) -> None:
    """Start the real FastAPI lifespan and send one echo completion."""
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda *args, **kwargs: TextContext,
    )
    app = fastapi_app(
        Settings(api_types=[APIType.OPENAI], use_heartbeat=False),
        ServingTokenGeneratorSettings(
            model_factory=EchoTokenGenerator,
            pipeline_config=mock_pipeline_config,
            tokenizer=EchoPipelineTokenizer(),
        ),
    )
    with TestClient(app) as client:
        probe = capsys.readouterr().err
        assert "ttft_join enabled" in probe

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "stream": False,
            },
        )
        assert response.status_code == 200
        err = capsys.readouterr().err
        join_lines = [
            line
            for line in err.splitlines()
            if line.startswith('{"event":"ttft_join"')
        ]
        assert join_lines, f"no ttft_join JSON in stderr:\n{err}"
        payload = json.loads(join_lines[0])
        assert payload["event"] == "ttft_join"
        assert payload["server_ttft_ms"] >= 0.0


def test_log_ttft_join_prints_yield_gap_split_without_changing_leftover(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    log_ttft_join(
        request_id="r1",
        server_ttft_ms=1000.0,
        ipt_ms=100.0,
        hops={
            "admit_ms": 200.0,
            "onload_wait_ms": 50.0,
            "transfer_wait_ms": 120.0,
            "tg_queue_ms": 300.0,
            "tg_first_out_ms": 40.0,
        },
    )
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["onload_wait_ms"] == 50.0
    assert payload["transfer_wait_ms"] == 120.0
    assert payload["tg_queue_ms"] == 300.0
    assert payload["tg_first_out_ms"] == 40.0
    # The split describes the wait inside yield_gap, so leftover keeps
    # its definition: server TTFT minus tokenize and the named hops.
    assert payload["leftover_ms"] == 700.0


def test_log_kv_hold_prints_json_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "0")
    log_kv_hold(replica=0, blocks_awaiting_prefill=10)
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("MAX_SERVE_TTFT_JOIN", "1")
    log_kv_hold(replica=1, blocks_awaiting_prefill=10, blocks_free=5)
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload == {
        "event": "kv_hold",
        "replica": 1,
        "blocks_awaiting_prefill": 10,
        "blocks_free": 5,
    }
