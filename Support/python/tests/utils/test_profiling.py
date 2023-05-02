# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import copy

from _pytest.logging import LogCaptureFixture
from modular.utils.profiling import TimeTrace

trace_fast = {
    "traceEvents": [
        {
            "pid": 1,
            "ts": 0,
            "ph": "M",
            "name": "process_name",
            "args": {"name": "trace_fast"},
        },
        {"pid": 1, "ph": "X", "ts": 10, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 20, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 30, "dur": 10, "name": "executeModel.run"},
        {"pid": 1, "ph": "X", "ts": 40, "dur": 10, "name": "executeModel.run"},
    ],
    "filename": "trace_fast.json",
    "versionInfo": {
        "modular-git-sha": "82d954ef",
        "modular-build-type": "release",
        "modular-profiling-level": "01001",
    },
}


def test_from_dict():
    TimeTrace.from_dict(trace_fast)


def test_from_dict_bad_build_type(caplog: LogCaptureFixture):
    trace_dbg = copy.deepcopy(trace_fast)
    trace_dbg["versionInfo"]["modular-build-type"] = "debug"
    TimeTrace.from_dict(trace_dbg, check_build_type="release")

    expected = "Trace was not generated from a 'release' build"
    assert len(caplog.records) == 1
    assert expected in caplog.records[0].msg


def test_from_dict_bad_profiling_level(caplog: LogCaptureFixture):
    trace_no_prof = copy.deepcopy(trace_fast)
    trace_no_prof["versionInfo"]["modular-profiling-level"] = "00000"
    TimeTrace.from_dict(trace_no_prof, check_profiling_levels="01111")

    expected = "Insufficient profiling level"
    assert len(caplog.records) == 1
    assert expected in caplog.records[0].msg


def test_process_name():
    x = TimeTrace.from_dict(trace_fast)
    assert x.process_name == "trace_fast"


def test_get_runs():
    runs = TimeTrace.from_dict(trace_fast).get_runs()
    assert len(runs) == 4
