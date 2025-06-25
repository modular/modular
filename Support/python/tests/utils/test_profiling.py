# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import copy
from typing import Any

from _pytest.logging import LogCaptureFixture
from modular.utils.profiling import TimeTrace

trace_fast: dict[str, Any] = {
    "traceEvents": [
        {
            "pid": 1,
            "ts": 0,
            "ph": "M",
            "name": "process_name",
            "args": {"name": "trace_fast"},
        },
        {
            "pid": 1,
            "ph": "M",
            "name": "thread_name",
            "args": {"name": "mt"},
        },
        {
            "pid": 1,
            "ph": "M",
            "name": "thread_name",
            "args": {"name": "🔥 Thread1"},
        },
        {"pid": 1, "ph": "X", "ts": 10, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 20, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 30, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 40, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 15, "dur": 3, "name": "foo.mojo.matmul"},
        {
            "pid": 1,
            "ph": "X",
            "ts": 15,
            "dur": 3,
            "name": "foo.mojo.matmul.task",
        },
        {"pid": 1, "ph": "X", "ts": 15, "dur": 3, "name": "bar.mojo.elemwise"},
    ],
    "filename": "trace_fast.json",
    "versionInfo": {
        "modular-git-sha": "82d954ef",
        "modular-build-type": "release",
        "modular-profiling-level": "01001",
    },
}


def test_from_dict() -> None:
    TimeTrace.from_dict(trace_fast)


def test_from_dict_bad_build_type(caplog: LogCaptureFixture) -> None:
    trace_dbg = copy.deepcopy(trace_fast)
    trace_dbg["versionInfo"]["modular-build-type"] = "debug"
    TimeTrace.from_dict(trace_dbg, check_build_type="release")

    expected = "Trace was not generated from a 'release' build"
    assert len(caplog.records) == 1
    assert expected in caplog.records[0].msg


def test_from_dict_bad_profiling_level(caplog: LogCaptureFixture) -> None:
    trace_no_prof = copy.deepcopy(trace_fast)
    trace_no_prof["versionInfo"]["modular-profiling-level"] = "00000"
    TimeTrace.from_dict(trace_no_prof, check_profiling_levels="01111")

    expected = "Insufficient profiling level"
    assert len(caplog.records) == 1
    assert expected in caplog.records[0].msg


def test_process_name() -> None:
    x = TimeTrace.from_dict(trace_fast)
    assert x.process_name == "trace_fast"


def test_get_runs() -> None:
    runs = TimeTrace.from_dict(trace_fast).get_runs()
    assert len(runs) == 4


def test_execution_interval_all() -> None:
    start_time, end_time = TimeTrace.from_dict(
        trace_fast
    ).get_execution_interval()
    assert start_time == 10
    assert end_time == 50


def test_execution_interval_selected() -> None:
    start_time, end_time = TimeTrace.from_dict(
        trace_fast
    ).get_execution_interval(1)
    assert start_time == 20
    assert end_time == 30


def test_num_threads_multiple() -> None:
    assert TimeTrace.from_dict(trace_fast).num_threads == 2


def test_trim() -> None:
    trimmed = TimeTrace.from_dict(trace_fast).trim(start_time=20, end_time=35)
    assert len(trimmed.get_runs()) == 2
    # check that metadata calls still work
    assert trimmed.num_threads == 2
    assert trimmed.process_name == "trace_fast"


def test_filter_events() -> None:
    full = TimeTrace.from_dict(trace_fast)
    filtered = full.filter_events(
        include_fragments=["matmul"], exclude_fragments=["task"]
    )
    # only the parent matmul should survive the filter
    assert len([x for x in filtered.events if "mojo" in x["name"]]) == 1


def test_standardize_names() -> None:
    unchanged = ["foo", "__gen_ins_P0_P1_call_spam.mojo.matmul.task"]

    for invariant in unchanged:
        assert invariant == TimeTrace.standardize_operator_name(invariant)

    to_standardize = {
        "CST0_0": "CST0_?",
        "CST123_456": "CST123_?",
        "_CST1_0_CST2_4_CST4_8_": "_CST1_?_CST2_?_CST4_?_",
    }
    for orig, std in to_standardize.items():
        assert std == TimeTrace.standardize_operator_name(orig)
        # standardization should be idempotent
        assert std == TimeTrace.standardize_operator_name(std)
