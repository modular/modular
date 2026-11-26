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
"""Benchmark serving dev unit tests"""

from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
import uvloop
from max.benchmark.benchmark_serving import (
    BenchmarkSession,
    EndCondition,
    _make_gpu_recorder,
    _run_benchmark_sweep,
    _run_dry_run_sweep,
    _sample_for_seed,
    parse_args,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    ChatSession,
    RequestSamples,
    ResponseFormat,
    SampledRequest,
    SessionMessage,
)
from max.benchmark.benchmark_shared.iteration import (
    resolve_skip_counts,
    retain_result,
)
from max.benchmark.benchmark_shared.metrics import BenchmarkResult
from max.benchmark.benchmark_shared.workload_stats import WorkloadStats
from tests.benchmark.benchmark_fakes import make_sampled_request


@patch("max.profiler.gpu.BackgroundRecorder")
@patch("max.benchmark.benchmark_serving.DCGMBackgroundRecorder")
def test_make_gpu_recorder_host_selects_dcgm(
    mock_dcgm: MagicMock, mock_local: MagicMock
) -> None:
    """A configured gpu_metrics_host picks the DCGM recorder, not local NVML."""
    recorder = _make_gpu_recorder(
        collect_gpu_stats=True, gpu_metrics_host="10.0.0.5,10.0.0.6"
    )

    mock_dcgm.assert_called_once_with("10.0.0.5,10.0.0.6")
    mock_local.assert_not_called()
    assert recorder is mock_dcgm.return_value


@patch("max.profiler.gpu.BackgroundRecorder")
@patch("max.benchmark.benchmark_serving.DCGMBackgroundRecorder")
@patch("max.benchmark.benchmark_serving.PrometheusGPURecorder")
def test_make_gpu_recorder_prometheus_wins_over_dcgm(
    mock_prometheus: MagicMock, mock_dcgm: MagicMock, mock_local: MagicMock
) -> None:
    """Prometheus narrows to the engine's GPUs, so it outranks a DCGM host."""
    recorder = _make_gpu_recorder(
        collect_gpu_stats=True,
        gpu_metrics_host="10.0.0.5",
        gpu_metrics_prometheus_url="http://prom:9090",
        gpu_metrics_selector='namespace="bench"',
    )

    mock_prometheus.assert_called_once_with(
        "http://prom:9090", 'namespace="bench"'
    )
    mock_dcgm.assert_not_called()
    mock_local.assert_not_called()
    assert recorder is mock_prometheus.return_value


def test_make_gpu_recorder_prometheus_requires_a_selector() -> None:
    """Without a selector the mean would span every GPU Prometheus scrapes."""
    with pytest.raises(ValueError, match="selector"):
        _make_gpu_recorder(
            collect_gpu_stats=True,
            gpu_metrics_host="",
            gpu_metrics_prometheus_url="http://prom:9090",
        )


@patch("max.profiler.gpu.BackgroundRecorder")
@patch("max.benchmark.benchmark_serving.DCGMBackgroundRecorder")
def test_make_gpu_recorder_empty_host_selects_local(
    mock_dcgm: MagicMock, mock_local: MagicMock
) -> None:
    """An empty host preserves the local NVML recorder path."""
    recorder = _make_gpu_recorder(collect_gpu_stats=True, gpu_metrics_host="")

    mock_local.assert_called_once_with()
    mock_dcgm.assert_not_called()
    assert recorder is mock_local.return_value


@patch("max.profiler.gpu.BackgroundRecorder")
@patch("max.benchmark.benchmark_serving.DCGMBackgroundRecorder")
def test_make_gpu_recorder_disabled_selects_neither(
    mock_dcgm: MagicMock, mock_local: MagicMock
) -> None:
    """collect_gpu_stats=False disables both recorders."""
    recorder = _make_gpu_recorder(
        collect_gpu_stats=False, gpu_metrics_host="10.0.0.5"
    )

    assert recorder is None
    mock_dcgm.assert_not_called()
    mock_local.assert_not_called()


def test_resolve_skip_counts() -> None:
    def rsc(
        orig_skip_first: int | None = None,
        orig_skip_last: int | None = None,
        request_rate: float = float("inf"),
        max_concurrency: int | None = None,
        ignore_first_turn_stats: bool = False,
        warmup_multiturn: bool = False,
    ) -> tuple[int, int]:
        return resolve_skip_counts(
            orig_skip_first=orig_skip_first,
            orig_skip_last=orig_skip_last,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            ignore_first_turn_stats=ignore_first_turn_stats,
            warmup_multiturn=warmup_multiturn,
        )

    # finite rate: auto-None → 0; explicit values are preserved
    assert rsc(request_rate=2.0) == (0, 0)
    assert rsc(orig_skip_first=3, orig_skip_last=5, request_rate=2.0) == (3, 5)
    assert rsc(orig_skip_last=5, request_rate=2.0) == (0, 5)

    # infinite rate, max_concurrency > 1: auto-None → concurrency value
    assert rsc(max_concurrency=4) == (4, 4)
    assert rsc(orig_skip_first=2, max_concurrency=4) == (2, 4)
    assert rsc(orig_skip_last=2, max_concurrency=4) == (4, 2)
    assert rsc(orig_skip_first=2, orig_skip_last=3, max_concurrency=4) == (2, 3)

    # infinite rate, max_concurrency=1: sequential, no ramp-up → 0
    assert rsc(max_concurrency=1) == (0, 0)

    # infinite rate, max_concurrency=None: unbounded → 0
    assert rsc(max_concurrency=None) == (0, 0)

    # ignore_first_turn_stats resets skip_first unless warmup_multiturn
    assert rsc(max_concurrency=4, ignore_first_turn_stats=True) == (0, 4)
    assert rsc(
        max_concurrency=4,
        ignore_first_turn_stats=True,
        warmup_multiturn=True,
    ) == (4, 4)
    assert rsc(max_concurrency=4, ignore_first_turn_stats=False) == (4, 4)

    # ignore_first_turn_stats with skip_first=0 (falsy): no override
    assert rsc(request_rate=2.0, ignore_first_turn_stats=True) == (0, 0)


@pytest.mark.parametrize(
    ("passed", "always_save_result", "kept"),
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_retain_result(
    passed: bool, always_save_result: bool, kept: bool
) -> None:
    """A result is kept when it passed or the caller always saves results."""
    result = BenchmarkResult(task_type="text", max_concurrency=1)
    retained = retain_result(
        result, passed=passed, always_save_result=always_save_result
    )
    assert retained is (result if kept else None)


def test_parse_args_pre_parses_concurrency_and_request_rate_sweeps() -> None:
    """CLI strings for concurrency and request-rate sweeps become parsed lists."""
    cfg = parse_args(
        [
            "--model",
            "myorg/model",
            "--max-concurrency",
            "1,none,4",
            "--request-rate",
            "inf,2.5",
        ]
    )
    assert list(cfg.max_concurrency) == [1, None, 4]
    assert cfg.request_rate[0] == float("inf")
    assert cfg.request_rate[1] == pytest.approx(2.5)


def test_parse_args_steady_state_filtering_defaults_on() -> None:
    """--no-steady-state-filtering turns off the default steady-state window."""
    assert parse_args(["--model", "myorg/model"]).steady_state_filtering
    cfg = parse_args(["--model", "myorg/model", "--no-steady-state-filtering"])
    assert cfg.steady_state_filtering is False


def test_benchmark_serving_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the benchmark serving help function."""
    # Mock sys.argv to simulate running with --help flag
    test_args = ["benchmark_serving.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_args()

        # Verify it exited with code 0 (success)
        assert excinfo.value.code == 0

        # Capture and verify the help output
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()


def _one_request_sample() -> tuple[SampledRequest, RequestSamples]:
    request = make_sampled_request()
    return request, RequestSamples(requests=[request], shared_contexts=[])


def test_response_format_injection_clears_ignore_eos() -> None:
    """A constrained response ends where its schema is satisfied, so the drawn
    output length caps it rather than pinning it. Leaving ``ignore_eos`` set
    asks the server to keep generating past the end of the response, which it
    can only do unconstrained."""
    request, samples = _one_request_sample()
    args = parse_args(
        [
            "--model",
            "myorg/model",
            "--response-format",
            '{"type": "json_object"}',
        ]
    )

    with patch(
        "max.benchmark.benchmark_serving.sample_requests", return_value=samples
    ):
        _sample_for_seed(args, "text-generation", None, False, 0)

    assert request.response_format is not None
    assert not request.ignore_eos


def test_ignore_eos_survives_without_a_response_format() -> None:
    """Control: an unconstrained request still honours its drawn length."""
    request, samples = _one_request_sample()
    args = parse_args(["--model", "myorg/model"])

    with patch(
        "max.benchmark.benchmark_serving.sample_requests", return_value=samples
    ):
        _sample_for_seed(args, "text-generation", None, False, 0)

    assert request.response_format is None
    assert request.ignore_eos


def _one_chat_sample() -> tuple[ChatSession, ChatSamples]:
    session = ChatSession(
        id=0,
        messages=[
            SessionMessage(source="user", content="hi", num_tokens=1),
            SessionMessage(source="assistant", content="ok", num_tokens=1),
            SessionMessage(source="user", content="again", num_tokens=1),
            SessionMessage(source="assistant", content="ok", num_tokens=1),
        ],
    )
    return session, ChatSamples(chat_sessions=[session], shared_contexts=[])


def test_response_format_injection_marks_every_user_turn() -> None:
    """The mark lands on the user turns, which are the ones that carry a
    request; an assistant message is replayed history and never asks the
    server for anything."""
    session, samples = _one_chat_sample()
    args = parse_args(
        [
            "--model",
            "myorg/model",
            "--response-format",
            '{"type": "json_object"}',
        ]
    )

    with patch(
        "max.benchmark.benchmark_serving.sample_requests", return_value=samples
    ):
        _sample_for_seed(args, "text-generation", None, True, 0)

    by_source: dict[str, list[ResponseFormat | None]] = {
        "user": [],
        "assistant": [],
    }
    for message in session.messages:
        by_source[message.source].append(message.response_format)
    assert all(rf is not None for rf in by_source["user"])
    assert all(rf is None for rf in by_source["assistant"])


def test_response_format_lands_on_the_last_turn_left_after_a_cut() -> None:
    session, samples = _one_chat_sample()
    args = parse_args(
        [
            "--model",
            "myorg/model",
            "--response-format",
            '{"type": "json_object"}',
            "--response-format-turn",
            "last",
            "--image-fraction",
            "1",
            "--image-num-turns",
            "1",
            "--image-long-side",
            "16",
        ]
    )

    with patch(
        "max.benchmark.benchmark_serving.sample_requests", return_value=samples
    ):
        _sample_for_seed(args, "text-generation", None, True, 0)

    user_turns = [m for m in session.messages if m.source == "user"]
    assert len(user_turns) == 1
    assert user_turns[0].response_format is not None


def test_chat_turns_unmarked_without_a_response_format() -> None:
    """Control: no flag, no constrained turns."""
    session, samples = _one_chat_sample()
    args = parse_args(["--model", "myorg/model"])

    with patch(
        "max.benchmark.benchmark_serving.sample_requests", return_value=samples
    ):
        _sample_for_seed(args, "text-generation", None, True, 0)

    assert all(m.response_format is None for m in session.messages)


def test_run_benchmark_sweep_yields_labeled_failure_without_result() -> None:
    """A failed iteration without --always-save-result yields labels but no result."""
    args = parse_args(
        [
            "--model",
            "myorg/model",
            "--tokenizer",
            "myorg/tokenizer",
            "--max-concurrency",
            "4",
            "--request-rate",
            "2.5",
        ]
    )
    args.flush_prefix_cache = False
    session = BenchmarkSession(
        benchmark_task="text-generation",
        endpoint="/v1/completions",
        api_url="http://localhost:8000/v1/completions",
        base_url="http://localhost:8000",
        model_id="myorg/model",
        tokenizer_id="myorg/session-tokenizer",
        tokenizer=None,
        samples=RequestSamples(requests=[]),
        lora_manager=None,
        trace_path=None,
        orig_skip_first=None,
        orig_skip_last=None,
    )

    loops: list[asyncio.AbstractEventLoop] = []

    async def failed_benchmark(
        *args: object, **kwargs: object
    ) -> tuple[BenchmarkResult, bool]:
        loops.append(asyncio.get_running_loop())
        return BenchmarkResult(task_type="text", max_concurrency=4), False

    with (
        patch(
            "max.benchmark.benchmark_serving.benchmark",
            side_effect=failed_benchmark,
        ),
        patch(
            "max.benchmark.benchmark_serving._sample_for_seed",
            return_value=RequestSamples(requests=[]),
        ),
    ):
        end_condition = EndCondition(
            num_prompts=1, num_prompts_multiplier=None, max_duration_s=None
        )
        runs = list(_run_benchmark_sweep(args, session, end_condition))

    assert len(runs) == 1
    (run,) = runs
    assert run.max_concurrency == 4
    assert run.request_rate == 2.5
    assert run.status == "fail"
    assert run.result is None
    assert run.labels is not None
    assert run.labels.tokenizer_id == "myorg/session-tokenizer"
    assert run.labels.model_id == "myorg/model"
    assert isinstance(loops[-1], uvloop.Loop)


def test_run_benchmark_sweep_samples_every_level_from_the_base_seed() -> None:
    """Every concurrency level samples the same workload."""
    args = parse_args(
        [
            "--model",
            "myorg/model",
            "--tokenizer",
            "myorg/tokenizer",
            "--max-concurrency",
            "4,8",
            "--seed",
            "42",
        ]
    )
    args.flush_prefix_cache = False
    session = BenchmarkSession(
        benchmark_task="text-generation",
        endpoint="/v1/completions",
        api_url="http://localhost:8000/v1/completions",
        base_url="http://localhost:8000",
        model_id="myorg/model",
        tokenizer_id="myorg/tokenizer",
        tokenizer=None,
        samples=RequestSamples(requests=[]),
        lora_manager=None,
        trace_path=None,
        orig_skip_first=None,
        orig_skip_last=None,
    )

    async def passing_benchmark(
        *args: object, **kwargs: object
    ) -> tuple[BenchmarkResult, bool]:
        return BenchmarkResult(task_type="text", max_concurrency=4), True

    with (
        patch(
            "max.benchmark.benchmark_serving.benchmark",
            side_effect=passing_benchmark,
        ),
        patch(
            "max.benchmark.benchmark_serving._sample_for_seed",
            return_value=RequestSamples(requests=[]),
        ) as sample,
    ):
        end_condition = EndCondition(
            num_prompts=1, num_prompts_multiplier=None, max_duration_s=None
        )
        runs = list(_run_benchmark_sweep(args, session, end_condition))

    assert [run.max_concurrency for run in runs] == [4, 8]
    assert [call.args[-1] for call in sample.call_args_list] == [42, 42]


def _thinking_samples(count: int) -> ChatSamples:
    """``count`` sessions of 3 turns that think 60 s: 120 s of interest each."""
    sessions = []
    for sid in range(count):
        messages: list[SessionMessage] = []
        for turn in range(3):
            messages.append(
                SessionMessage(source="user", content=f"u{turn}", num_tokens=5)
            )
            messages.append(
                SessionMessage(
                    source="assistant",
                    content="a",
                    num_tokens=5,
                    delay_until_next_message=(60_000.0 if turn < 2 else None),
                )
            )
        sessions.append(ChatSession(id=sid, messages=messages))
    return ChatSamples(chat_sessions=sessions, shared_contexts=[])


def _preview(samples: ChatSamples, argv: list[str], rate: float) -> int:
    """Run one dry-run preview point and return the runs it reported."""
    args = parse_args(["--model", "myorg/model", *argv])
    args.print_workload_stats = True
    return len(
        list(
            _run_dry_run_sweep(
                args,
                BenchmarkSession(
                    benchmark_task="text-generation",
                    endpoint="/v1/chat/completions",
                    api_url="http://localhost:8000/v1/chat/completions",
                    base_url="http://localhost:8000",
                    model_id="myorg/model",
                    tokenizer_id="myorg/session-tokenizer",
                    tokenizer=None,
                    samples=samples,
                    lora_manager=None,
                    trace_path=None,
                    orig_skip_first=None,
                    orig_skip_last=None,
                ),
                EndCondition(
                    num_prompts=10,
                    num_prompts_multiplier=None,
                    max_duration_s=None,
                ),
                [None],
                [rate],
                WorkloadStats(
                    requests=0,
                    context_limit=None,
                    turns_dropped_at_context_limit=0,
                    input_len=None,
                    input_len_first_turn=None,
                    input_len_later_turn=None,
                    output_len=None,
                    tool_request_fraction=None,
                    tools_per_request=None,
                    image_request_fraction=None,
                    constrained_request_fraction=None,
                ),
            )
        )
    )


def test_dry_run_preview_reports_the_plan_the_run_will_use(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The preview samples the population and pool split the run itself uses.

    A preview that measured the chain from a different ``main_pool_target``
    reported a different candidate count and factor than the run, which is the
    one thing a preview exists to get right.
    """
    # Five sessions of 120 s think time hold ceil(0.01 x 120) = 2 open, and the
    # measured pool keeps the rest because the dataset has nothing to spare.
    with caplog.at_level(logging.INFO, logger="max.benchmark"):
        assert (
            _preview(
                _thinking_samples(5),
                ["--request-rate", "0.01", "--num-chat-sessions", "5"],
                0.01,
            )
            == 1
        )
    assert "M=2" in caplog.text
    assert "main_pool=3" in caplog.text
    assert "factor=1" in caplog.text

    # A workload with no think time derives a zero population, so the preview
    # says the run warms nothing instead of sampling every session as a warm-up.
    caplog.clear()
    with caplog.at_level(
        logging.INFO, logger="max.benchmark.benchmark_serving"
    ):
        assert (
            _preview(
                _one_chat_sample()[1],
                ["--request-rate", "1.0", "--num-chat-sessions", "1"],
                1.0,
            )
            == 1
        )
    assert "will not pre-warm any sessions" in caplog.text
