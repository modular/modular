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
"""Tests for the shared eval scaffolding."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import eval_common
import httpx
import pytest
from openai import BadRequestError
from openai.types.chat import ChatCompletion

# An endpoint that rejects every request: the score is 0.0 not because the model
# was wrong but because nothing was measured.
ALL_ERRORED = {"accuracy": 0.0, "correct": 0, "total": 120, "errors": 120}

# A real AIME25 run whose only failures were gateway timeouts on the two
# longest-generating problems. Well inside the budget; must still score.
GATEWAY_TIMEOUTS = {
    "accuracy": 0.9021,
    "correct": 433,
    "total": 480,
    "errors": 3,
}


def test_error_budget_rejects_an_all_errored_run() -> None:
    with pytest.raises(SystemExit):
        eval_common.enforce_error_budget(ALL_ERRORED)


def test_error_budget_allows_incidental_errors() -> None:
    eval_common.enforce_error_budget(GATEWAY_TIMEOUTS)


@pytest.mark.parametrize(
    ("errors", "rejected"),
    [(9, False), (10, False), (11, True)],
)
def test_error_budget_boundary(errors: int, rejected: bool) -> None:
    summary = {"total": 100, "errors": errors}
    if rejected:
        with pytest.raises(SystemExit):
            eval_common.enforce_error_budget(summary)
    else:
        eval_common.enforce_error_budget(summary)


def test_error_budget_reads_the_judged_eval_key() -> None:
    """AA-Omniscience names its error count ``errored``, not ``errors``."""
    with pytest.raises(SystemExit):
        eval_common.enforce_error_budget({"total": 600, "errored": 600})


@pytest.mark.parametrize(
    "summary",
    [
        {"total": 600},  # no error count reported
        {"accuracy": 0.4},  # no total either (SciCode-style)
        {"total": 0, "errors": 0},  # nothing submitted
    ],
)
def test_error_budget_skips_summaries_it_cannot_judge(
    summary: dict[str, Any],
) -> None:
    eval_common.enforce_error_budget(summary)


def test_dump_score_writes_before_rejecting(tmp_path: Path) -> None:
    """A rejected run still leaves score.json behind to diagnose from."""
    out = tmp_path / "results"
    with pytest.raises(SystemExit):
        eval_common.dump_score(str(out), ALL_ERRORED)
    written = json.loads((out / "score.json").read_text())
    assert written == ALL_ERRORED


def test_gated_dataset_denial_fails_the_step() -> None:
    """A lane that cannot load its dataset must not report success."""

    def denied() -> list[dict[str, Any]]:
        raise RuntimeError("403 Client Error: gated repo, enable access")

    with pytest.raises(SystemExit) as excinfo:
        eval_common.load_gated(denied, label="GPQA", dataset_id="some/dataset")
    assert excinfo.value.code != 0


def test_gated_loader_passes_rows_through() -> None:
    rows = [{"q": 1}]
    assert (
        eval_common.load_gated(
            lambda: rows, label="GPQA", dataset_id="some/dataset"
        )
        is rows
    )


def test_gated_loader_propagates_unrelated_errors() -> None:
    """Only access denials are translated; real bugs keep their traceback."""

    def broken() -> list[dict[str, Any]]:
        raise ValueError("malformed row")

    with pytest.raises(ValueError):
        eval_common.load_gated(broken, label="GPQA", dataset_id="some/dataset")


# A run where a quarter of completed responses hit the token cap. Below any
# reasonable floor, but must pass while the gate is off (the default).
LOW_STOP = {
    "total": 100,
    "errors": 0,
    "finish_stop": 75,
    "finish_length": 25,
    "stop_ratio": 0.75,
}


def test_finish_stats_counts_and_ratio() -> None:
    rows = (
        [{"finish_reason": "stop"}] * 3
        + [{"finish_reason": "length"}]
        + [{"error": "boom"}]  # errored rows carry no finish reason
    )
    s = eval_common.finish_stats(rows)
    # These rows carry no ``correct`` key, so nothing counts as correct.
    assert s == {
        "finish_stop": 3,
        "finish_length": 1,
        "stop_ratio": 0.75,
        "correct_given_stop": 0,
        "accuracy_given_stop": 0.0,
        "correct_given_length": 0,
        "accuracy_given_length": 0.0,
    }


def test_finish_stats_with_nothing_completed() -> None:
    assert eval_common.finish_stats([{"error": "x"}])["stop_ratio"] is None


def test_stop_ratio_gate_is_off_by_default() -> None:
    eval_common.enforce_stop_ratio(LOW_STOP)


def test_stop_ratio_gate_rejects_below_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_common, "MIN_STOP_RATIO", 0.9)
    with pytest.raises(SystemExit):
        eval_common.enforce_stop_ratio(LOW_STOP)


def test_stop_ratio_gate_passes_at_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_common, "MIN_STOP_RATIO", 0.75)
    eval_common.enforce_stop_ratio(LOW_STOP)


def test_stop_ratio_gate_skips_summaries_without_the_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_common, "MIN_STOP_RATIO", 0.9)
    eval_common.enforce_stop_ratio({"total": 100, "errors": 0})
    eval_common.enforce_stop_ratio({"stop_ratio": None})


def test_exact_match_score_reports_stop_ratio() -> None:
    rows = [
        {"correct": True, "finish_reason": "stop", "completion_tokens": 10},
        {"correct": False, "finish_reason": "length", "completion_tokens": 99},
    ]
    s = eval_common.exact_match_score(rows, total=2, errors=0)
    assert (s["finish_stop"], s["finish_length"], s["stop_ratio"]) == (
        1,
        1,
        0.5,
    )


def _completed_accuracy(stats: dict[str, object]) -> float:
    """Recombines the two conditional accuracies into completed-row accuracy."""
    stop_ratio = stats["stop_ratio"]
    assert isinstance(stop_ratio, float)
    given_stop = stats["accuracy_given_stop"] or 0.0
    given_length = stats["accuracy_given_length"] or 0.0
    assert isinstance(given_stop, float) and isinstance(given_length, float)
    return given_stop * stop_ratio + given_length * (1 - stop_ratio)


def test_finish_stats_reports_the_accuracy_decomposition() -> None:
    """Completed-row accuracy splits by finish reason, weighted by stop_ratio.

    Reporting the factors rather than their sum keeps a shift in how often the
    model fails to terminate from reading as a change in answer quality.
    """
    results = [
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": False},
        {"finish_reason": "length", "correct": False},
    ]
    stats = eval_common.finish_stats(results)
    assert stats["finish_stop"] == 4
    assert stats["finish_length"] == 1
    assert stats["stop_ratio"] == 0.8
    assert stats["correct_given_stop"] == 3
    assert stats["accuracy_given_stop"] == 0.75
    assert stats["correct_given_length"] == 0
    assert stats["accuracy_given_length"] == 0.0

    # Every row here completed, so completed-row accuracy is overall accuracy.
    overall = sum(1 for r in results if r["correct"]) / len(results)
    assert overall == pytest.approx(_completed_accuracy(stats))


def test_finish_stats_decomposition_excludes_errored_rows() -> None:
    """The product is completed-row accuracy, above accuracy over everything.

    :func:`eval_common.exact_match_score` keeps errored rows in the denominator
    of the headline ``accuracy`` so dropping the hardest problems cannot inflate
    it. Those rows carry no finish reason, so they are outside this product --
    the two agree only when nothing errored.
    """
    results = [
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "length", "correct": False},
        {"error": "boom", "correct": False},  # no finish reason
    ]
    stats = eval_common.finish_stats(results)
    # 1 correct of the 2 completed.
    assert _completed_accuracy(stats) == pytest.approx(0.5)

    summary = eval_common.exact_match_score(results, total=3, errors=1)
    assert summary["accuracy"] == pytest.approx(1 / 3)
    # The gap is exactly the completion factor the summary already reports.
    completion = summary["answered"] / summary["total"]
    assert summary["accuracy"] == pytest.approx(
        _completed_accuracy(stats) * completion
    )


def test_finish_stats_decomposition_survives_a_correct_truncated_row() -> None:
    """A ``length`` row that grades correct keeps the identity exact.

    Nothing makes truncation imply a wrong grade: the graders parse whatever
    content came back without consulting ``finish_reason``, and ``strip_think``
    only removes a ``<think>`` span that was actually closed. A response cut off
    mid-reasoning therefore reaches the parser as raw reasoning text, out of
    which ``gpqa_eval`` can read a standalone option letter and ``aime_eval``
    the last integer. Weighting both conditional accuracies keeps the
    decomposition exact when that happens, where a stop-only product would
    silently drop the row.
    """
    results = [
        {"finish_reason": "stop", "correct": True},
        {"finish_reason": "stop", "correct": False},
        {"finish_reason": "length", "correct": True},  # lucky parse
        {"finish_reason": "length", "correct": False},
    ]
    stats = eval_common.finish_stats(results)
    assert stats["accuracy_given_stop"] == 0.5
    assert stats["correct_given_length"] == 1
    assert stats["accuracy_given_length"] == 0.5

    overall = sum(1 for r in results if r["correct"]) / len(results)
    assert overall == pytest.approx(_completed_accuracy(stats))
    # The stop-side factor alone would have understated it by the dropped row.
    assert stats["accuracy_given_stop"] * stats["stop_ratio"] == pytest.approx(
        0.25
    )


def test_finish_stats_counts_recombine_exactly_despite_rounding() -> None:
    """The counts, unlike the reported ratios, carry no rounding error.

    ``accuracy_given_stop`` and ``stop_ratio`` are each rounded to four places,
    so a caller needing the identity to the last bit adds the counts instead.
    """
    results = [
        {"finish_reason": "stop", "correct": i % 3 == 0} for i in range(7)
    ]
    results += [{"finish_reason": "length", "correct": False}] * 2
    stats = eval_common.finish_stats(results)

    assert stats["accuracy_given_stop"] == 0.4286  # 3/7, rounded
    correct_completed = (
        stats["correct_given_stop"] + stats["correct_given_length"]
    )
    completed = stats["finish_stop"] + stats["finish_length"]
    assert correct_completed / completed == pytest.approx(3 / 9)
    # The rounded ratios only reconstruct it approximately.
    assert _completed_accuracy(stats) == pytest.approx(3 / 9, abs=1e-4)


def test_finish_stats_takes_an_eval_specific_correctness_reader() -> None:
    """Evals that grade into another field pass their own reader.

    AA-Omniscience grades into a four-way ``verdict`` and never sets
    ``correct``; without this its stopped-row metrics would read a missing key
    and report 0 on every run.
    """
    results = [
        {"finish_reason": "stop", "verdict": "CORRECT"},
        {"finish_reason": "stop", "verdict": "INCORRECT"},
        {"finish_reason": "stop", "verdict": "NOT_ATTEMPTED"},
        {"finish_reason": "length", "verdict": "NOT_ATTEMPTED"},
    ]
    stats = eval_common.finish_stats(
        results, is_correct=lambda r: r["verdict"] == "CORRECT"
    )
    assert stats["correct_given_stop"] == 1
    assert stats["accuracy_given_stop"] == 0.3333  # 1 of 3 stopped, rounded

    # The default reader finds no ``correct`` key on these rows at all.
    assert eval_common.finish_stats(results)["correct_given_stop"] == 0


def test_finish_stats_accuracy_given_stop_is_none_without_stopped_rows() -> (
    None
):
    """No stopped rows means the ratio has no denominator, not a zero."""
    stats = eval_common.finish_stats(
        [{"finish_reason": "length", "correct": False}]
    )
    assert stats["accuracy_given_stop"] is None
    assert stats["stop_ratio"] == 0.0


def _run(
    items: list[Any],
    out_dir: Path,
    resume: bool,
    fail: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[list[dict[str, Any]], int, list[tuple[int, int]]]:
    """Runs ``run_parallel``, recording each ``(repeat, prompt)`` it asks for.

    Items whose ``(repeat, prompt)`` is in ``fail`` raise.
    """
    called: list[tuple[int, int]] = []

    def fn(item: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
        rep, idx, _ = item
        called.append((rep, idx))
        if (rep, idx) in fail:
            raise RuntimeError("timeout")
        return {"repeat_index": rep, "prompt_index": idx, "correct": True}

    def on_error(
        item: tuple[int, int, dict[str, Any]], exc: Exception
    ) -> dict[str, Any]:
        rep, idx, _ = item
        return {"repeat_index": rep, "prompt_index": idx, "error": str(exc)}

    results, errors = eval_common.run_parallel(
        items, fn, on_error, 2, "test", out_dir=str(out_dir), resume=resume
    )
    return results, errors, called


def _lines(out_dir: Path) -> list[dict[str, Any]]:
    text = (out_dir / "results.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


ITEMS = eval_common.expand_repeats(
    [(0, {"q": "a"}), (1, {"q": "b"}), (2, {"q": "c"})], 2
)


def test_resume_skips_finished_rows_and_retries_errored_ones(
    tmp_path: Path,
) -> None:
    _, errors, _ = _run(ITEMS, tmp_path, resume=False, fail=frozenset({(1, 2)}))
    assert errors == 1
    # A kill can leave the last line half-written.
    with open(tmp_path / "results.jsonl", "a") as f:
        f.write('{"repeat_index": 1, "prompt_')

    results, errors, called = _run(ITEMS, tmp_path, resume=True)

    assert called == [(1, 2)]
    assert errors == 0
    assert len(results) == len(ITEMS)
    assert not any("error" in r for r in results)
    assert len(_lines(tmp_path)) == len(ITEMS)


def test_every_row_records_when_its_request_was_sent(tmp_path: Path) -> None:
    before = datetime.now(timezone.utc)
    results, _, _ = _run(
        ITEMS, tmp_path, resume=False, fail=frozenset({(1, 2)})
    )
    after = datetime.now(timezone.utc)

    for row in results:
        sent = datetime.fromisoformat(row["timestamp"])
        assert sent.tzinfo == timezone.utc
        assert before <= sent <= after
    assert {r["timestamp"] for r in _lines(tmp_path)} == {
        r["timestamp"] for r in results
    }

    resumed, _, _ = _run(ITEMS, tmp_path, resume=True)
    # Finished rows keep the time they were first sent.
    kept = {r["timestamp"] for r in results if "error" not in r}
    assert kept <= {r["timestamp"] for r in resumed}


def test_resume_counts_duplicate_rows(tmp_path: Path) -> None:
    # --row-ids may repeat an index; each copy is its own sample.
    items = eval_common.expand_repeats([(4, {"q": "x"}), (4, {"q": "x"})], 1)
    _run(items, tmp_path, resume=False, fail=frozenset({(0, 4)}))

    rows = _lines(tmp_path)
    assert sum("error" in r for r in rows) == 2

    (tmp_path / "results.jsonl").write_text(
        json.dumps({"repeat_index": 0, "prompt_index": 4, "correct": True})
        + "\n"
    )
    results, _, called = _run(items, tmp_path, resume=True)

    assert called == [(0, 4)]
    assert len(results) == 2


def test_resume_drops_rows_no_item_asks_for(tmp_path: Path) -> None:
    _run(ITEMS, tmp_path, resume=False)

    results, _, called = _run(ITEMS[:2], tmp_path, resume=True)

    assert called == []
    assert len(results) == 2
    assert len(_lines(tmp_path)) == 2


def test_resume_matches_rows_without_a_repeat_index(tmp_path: Path) -> None:
    # Single-pass evals (HLE, AA-Omniscience) pass (prompt_index, sample) items
    # and write rows without a repeat_index.
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"prompt_index": 0, "correct": True}) + "\n"
    )
    called: list[Any] = []

    def fn(item: tuple[int, str]) -> dict[str, Any]:
        called.append(item)
        return {"prompt_index": item[0], "correct": True}

    results, _ = eval_common.run_parallel(
        [(0, "a"), (1, "b")],
        fn,
        lambda item, exc: {"prompt_index": item[0], "error": str(exc)},
        1,
        "test",
        out_dir=str(tmp_path),
        resume=True,
    )

    assert called == [(1, "b")]
    assert len(results) == 2


def test_without_resume_a_rerun_starts_over(tmp_path: Path) -> None:
    _run(ITEMS, tmp_path, resume=False)

    _, _, called = _run(ITEMS, tmp_path, resume=False)

    assert len(called) == len(ITEMS)
    assert len(_lines(tmp_path)) == len(ITEMS)


def test_resume_needs_an_out_dir() -> None:
    with pytest.raises(ValueError, match="out_dir"):
        eval_common.run_parallel(
            [], lambda item: {}, lambda item, exc: {}, 1, "test", resume=True
        )


CONFIG = {"model": "m", "seed": 7, "temperature": 1.0, "row_ids": None}


def test_prepare_out_dir_accepts_the_same_config(tmp_path: Path) -> None:
    eval_common.prepare_out_dir(str(tmp_path), CONFIG, resume=False)
    (tmp_path / "results.jsonl").write_text("")

    eval_common.prepare_out_dir(str(tmp_path), dict(CONFIG), resume=True)


def test_prepare_out_dir_rejects_a_different_config(tmp_path: Path) -> None:
    eval_common.prepare_out_dir(str(tmp_path), CONFIG, resume=False)
    (tmp_path / "results.jsonl").write_text("")

    with pytest.raises(ValueError, match="seed"):
        eval_common.prepare_out_dir(
            str(tmp_path), {**CONFIG, "seed": 8}, resume=True
        )


def test_prepare_out_dir_rejects_results_without_a_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "results.jsonl").write_text("")

    with pytest.raises(ValueError, match=r"no run_config\.json"):
        eval_common.prepare_out_dir(str(tmp_path), CONFIG, resume=True)


def test_prepare_out_dir_resume_on_a_fresh_dir_records_the_config(
    tmp_path: Path,
) -> None:
    eval_common.prepare_out_dir(str(tmp_path / "new"), CONFIG, resume=True)

    recorded = json.loads((tmp_path / "new" / "run_config.json").read_text())
    assert recorded == CONFIG


def test_resumable_config_ignores_operational_options() -> None:
    params = {
        **CONFIG,
        "base_url": "http://a",
        "workers": 16,
        "out_dir": "/tmp/x",
        "resume": True,
        "metric_prefix": "GPQA",
    }

    assert eval_common.resumable_config(params) == CONFIG


def test_build_chat_kwargs_omits_extra_body_when_nothing_rides_it() -> None:
    """No ``chat_template_kwargs`` or token IDs must not add an empty one."""
    kwargs = eval_common.build_chat_kwargs(
        "m",
        [{"role": "user", "content": "hi"}],
        eval_common.GenParams(return_token_ids=False),
    )
    assert "extra_body" not in kwargs


def test_build_chat_kwargs_forwards_chat_template_kwargs_via_extra_body() -> (
    None
):
    """``chat_template_kwargs`` rides ``extra_body``"""
    params = eval_common.GenParams(
        chat_template_kwargs={"reasoning_effort": "low"},
        return_token_ids=False,
    )
    kwargs = eval_common.build_chat_kwargs(
        "m", [{"role": "user", "content": "hi"}], params
    )
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"reasoning_effort": "low"}
    }


def test_build_chat_kwargs_requests_token_ids_via_extra_body() -> None:
    params = eval_common.GenParams(
        chat_template_kwargs={"reasoning_effort": "low"},
        return_token_ids=True,
    )
    kwargs = eval_common.build_chat_kwargs(
        "m", [{"role": "user", "content": "hi"}], params
    )
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "return_token_ids": True,
    }


def test_response_fields_reads_token_ids_off_the_choice() -> None:
    """MAX returns ``token_ids`` beside the message, an extra SDK field."""
    completion = ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Answer: A",
                        "reasoning": "hmm",
                    },
                    "token_ids": [151644, 0, 7],
                }
            ],
        }
    )
    assert eval_common.response_fields(completion.choices[0]) == {
        "response": "Answer: A",
        "reasoning": "hmm",
        "completion_token_ids": [151644, 0, 7],
    }


def test_response_fields_without_token_ids() -> None:
    choice = SimpleNamespace(message=SimpleNamespace(content="Answer: A"))
    assert eval_common.response_fields(choice) == {
        "response": "Answer: A",
        "reasoning": None,
        "completion_token_ids": None,
    }


def test_judge_does_not_request_token_ids() -> None:
    sent: dict[str, Any] = {}

    def create(**kwargs: Any) -> SimpleNamespace:
        sent.update(kwargs)
        message = SimpleNamespace(content="correct: yes")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert eval_common.self_judge(client, "m", "prompt")
    assert "extra_body" not in sent


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "http://host/v1/chat/completions")
    return BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


class _RejectingCompletions:
    """Fails requests that ask for token IDs with ``error``, like a strict server."""

    def __init__(self, error: BadRequestError) -> None:
        self.error = error
        self.sent: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> str:
        self.sent.append(kwargs)
        if "return_token_ids" in kwargs.get("extra_body", {}):
            raise self.error
        return "response"


def test_chat_completion_drops_token_ids_a_server_rejects() -> None:
    client = _RejectingCompletions(
        _bad_request(
            "Error code: 400 - Unrecognized argument: return_token_ids"
        )
    )
    params = eval_common.GenParams(return_token_ids=True)
    messages = [{"role": "user", "content": "hi"}]

    assert eval_common.chat_completion(client, "m", messages, params) == (
        "response"
    )
    assert "extra_body" not in client.sent[1]
    # Later requests of the run share params and no longer ask.
    assert not params.return_token_ids
    eval_common.chat_completion(client, "m", messages, params)
    assert len(client.sent) == 3


def test_chat_completion_raises_other_rejections() -> None:
    client = _RejectingCompletions(_bad_request("Error code: 400 - bad model"))
    params = eval_common.GenParams(return_token_ids=True)
    with pytest.raises(BadRequestError):
        eval_common.chat_completion(
            client, "m", [{"role": "user", "content": "hi"}], params
        )
    assert params.return_token_ids
    assert len(client.sent) == 1
