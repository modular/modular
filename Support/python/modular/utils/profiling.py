# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Profiling Utility Library

This is a module for parsing time-trace files generated from mt and other
performance analysis-related helpers.
"""

import json
from copy import deepcopy as dcopy
from functools import cached_property
from pathlib import Path
from re import sub
from typing import Any, Callable, Dict, List, Optional

from modular.utils.logging import warning

EventDict = Dict[str, Any]
TraceDict = Dict[str, Any]
VersionInfoDict = Dict[str, Any]
EventList = List[EventDict]

# Perfetto Trace Event Format doc:
# https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU

RUN_NAMES = {"benchmarkModelOnce", "executeModel.run"}


class TimeTrace:
    """
    This is a helper class that wraps a time-trace json object and provides
    convenience functions.
    """

    def __init__(self, trace: TraceDict):
        self.trace = trace

    @classmethod
    def from_dict(
        cls,
        trace: TraceDict,
        check_build_type: Optional[str] = "rel",
        check_profiling_levels: Optional[str] = "00001",
        deepcopy: bool = True,
    ) -> "TimeTrace":
        """
        Factory function to create a TimeTrace from a dictionary.
        Args:
            trace: The dictionary of events to wrap.
            check_build_type: Substring that is expected within the build-type
                              If None, then no check is performed.
                              The check is not case-sensitive.
            check_profiling_levels: Octal string of minimum required levels.
                                    If None, then no check is performed.
            deepcopy: make a deep copy of the input data (default: true)
        Returns:
            The constructed TimeTrace.
        """
        if deepcopy:
            trace = dcopy(trace)

        # Construct TimeTrace
        time_trace = cls(trace)
        fname = trace.get("filename", "Trace")

        # Check build type, if requested, and print warning.
        if check_build_type:
            try:
                time_trace.check_build_type(check_build_type)
            except RuntimeError as e:
                warning(f"{fname}: {str(e)}")

        # Check profiling levels, if requested, and print warning.
        if check_profiling_levels:
            try:
                time_trace.check_profiling_levels(check_profiling_levels)
            except RuntimeError as e:
                warning(f"{fname}: Insufficient profiling level ({str(e)})")

        return time_trace

    @classmethod
    def from_file(cls, tracefile: Path, **kwargs) -> "TimeTrace":
        """
        Factory function to create a TimeTrace from a filename.
        Args:
            tracefile: Path to the time-trace file.
        Returns:
            The constructed TimeTrace.
        """
        with open(tracefile, "r") as infile:
            trace = json.load(infile)
        trace.update({"filename": tracefile})
        return cls.from_dict(trace, **kwargs)

    def to_file(self, outfile: Path, **kwargs):
        """Write the trace to a file as json."""
        with open(outfile, "w") as f:
            json.dump(self.trace, f)

    @property
    def events(self) -> EventList:
        """Return all events in the trace"""
        return self.trace["traceEvents"]

    def standardize_names(self) -> "TimeTrace":
        """Standardize the names of operators in-place."""
        for e in self.events:
            e["name"] = TimeTrace.standardize_operator_name(e["name"])
        return self

    @staticmethod
    def standardize_operator_name(name: str) -> str:
        """Standardize the name of an operator.

        The goal here is to remove elements of the operator name that don't
        indicate differences in its computational signature, so that they
        can be more meaningfully aggregated by name downstream.  Current
        standardizations:
          - Replace constant value references with "?"

        """
        return sub(
            r"(CST[\d+]+\_[\d+]+)",
            lambda match: match.group().partition("_")[0] + "_?",
            name,
        )

    @cached_property
    def num_threads(self) -> int:
        """Get the number of threads used in the profile."""

        # note: there aren't many thread events
        thread_events = [
            x["args"]["name"] for x in self.events if "dur" not in x
        ]

        thread_ids = [
            int(x.rpartition("Thread")[-1].strip())
            for x in thread_events
            if "Thread" in x
        ] + [
            0
        ]  # main thread doesn't have "Thread" in it
        return max(thread_ids) + 1

    def trim(
        self, start_time: int, end_time: int, include_metadata: bool = True
    ) -> "TimeTrace":
        """Trim the timeline of the events based on time timestamps (in us)"""

        def __include(x: EventDict):
            if x["ph"] == "M" and include_metadata:
                return True
            elif x["ph"] == "M" and not include_metadata:
                return False
            else:
                return x["ts"] >= start_time and x["ts"] <= end_time

        self.trace["traceEvents"] = list(filter(__include, self.events))
        return self

    def annotate(
        self, func: Callable[[EventDict], Dict[str, Any]]
    ) -> "TimeTrace":
        """Apply a function and inline outputs into 'args'."""
        for ev in self.trace["traceEvents"]:
            for k, v in (func(ev) or {}).items():
                ev["args"][k] = v
        return self

    def filter_events(
        self,
        include_fragments: Optional[List[str]] = None,
        exclude_fragments: Optional[List[str]] = None,
    ) -> "TimeTrace":
        """Select and remove events with names that match the given fragments.
        """
        local_include = (
            # default to the empty string, which is in everything
            include_fragments if include_fragments
            is not None else [""]
        )
        local_exclude = exclude_fragments if exclude_fragments else []

        def __include(ev: EventDict):
            name = ev["name"]
            return all(x not in name for x in local_exclude) and any(
                x in name for x in local_include
            )

        self.trace["traceEvents"] = list(filter(__include, self.events))
        return self

    @property
    def version_info(self) -> VersionInfoDict:
        return self.trace["versionInfo"]

    def get_process_name_event(self) -> EventDict:
        """Find the 'process_name' event and return it"""
        names = list(filter(lambda x: x["name"] == "process_name", self.events))
        assert len(names) == 1, "More than one process in tracefile?"
        return names[0]

    @property
    def process_name(self) -> str:
        """Return the name of the traced process"""
        return self.get_process_name_event()["args"]["name"]

    @property
    def process_id(self) -> int:
        """Return the PID of the traced process"""
        return self.get_process_name_event()["pid"]

    @classmethod
    def _get_level_list(cls, level_string: str) -> List[int]:
        """
        Convert given level string to level list.
        Args:
            level_string: Octal string of profiling levels.
        Raises:
            ValueError: if given string does not contain leading zero.
        Returns:
            List[int]: Level list in reverse order.
        """
        if level_string[0] != "0":
            raise ValueError(
                "Given level value ({level_string}) "
                "does not match expected octal format"
            )
        return [int(x) for x in level_string[-1:1:-1]]

    def check_build_type(self, expected_build_type: str) -> bool:
        """
        Check trace build type contains the given string as a substring.
        The check is not case-sensitive.
        Args:
            expected_build_type: Substring to search for.
        Raises:
            RuntimeError: If an insufficient build type is detected.
        Returns:
            boolean indicating whether or not the check succeeded.
        """
        trace_build_type = self.version_info["modular-build-type"].lower()
        if expected_build_type.lower() not in trace_build_type:
            exp = expected_build_type.lower()
            raise RuntimeError(f"Trace was not generated from a '{exp}' build.")
        return True

    def check_profiling_levels(self, expected_levels: str) -> bool:
        """
        Check trace profiling levels match or exceed given requirements.
        Args:
            expected_levels: Octal string of minimum required levels
        Raises:
            ValueError: if expected or recorded levels has invalid format.
            RuntimeError: if an insufficient profiling level is detected.
        Returns:
            boolean indicating whether or not the check succeeded.
        """
        # List of known profiling categories, in reverse order
        # These are only used for error messages.
        labels = ["Other", "AsyncRT", "Mem", "Mojo"]
        # Retrieve recorded profiling levels
        trace_levels = self.version_info["modular-profiling-level"]
        # Convert expected & recorded levels to list of ints in reverse order
        trace_level_list = self._get_level_list(trace_levels)
        expected_level_list = self._get_level_list(expected_levels)

        # Check levels
        for i, expected in enumerate(expected_level_list):
            if trace_level_list[i] < expected:
                # Found an error.
                # If we have a label, use it. Otherwise, fall back on 'unknown'.
                label = labels[i] if i < len(labels) else "Unknown"
                raise RuntimeError(f"{label} level {expected} required")
        return True

    def get_events_by_name(self, name: str) -> EventList:
        """Return all trace events matching 'name'"""
        return list(filter(lambda x: x["name"] == name, self.events))

    def get_runs(self) -> EventList:
        """Return ordered list of run events from trace"""
        self.check_profiling_levels("00001")  # Need at least 'Other' profiling
        exec_runs = self.get_events_by_name("executeModel.run")
        bench_runs = self.get_events_by_name("benchmarkModelOnce")
        assert len(exec_runs) != 0 or len(bench_runs) != 0, "No runs?"
        assert len(exec_runs) == 0 or len(bench_runs) == 0, "Mixed runs?"
        runs = exec_runs if len(exec_runs) != 0 else bench_runs
        return sorted(runs, key=lambda x: x["ts"])

    def get_execution_interval(self, run_number: Optional[int] = None):
        """
        Return the (start_time, end_time) pair for all or the selected run
        """
        runs = self.get_runs()
        if run_number is not None:
            assert 0 <= run_number < len(runs), "Invalid --run-number"
            start_time = runs[run_number]["ts"]
            end_time = start_time + runs[run_number]["dur"]
        else:
            start_time = runs[0]["ts"]
            end_time = runs[-1]["ts"] + runs[-1]["dur"]
        return (start_time, end_time)

    @staticmethod
    def parse_details(as_str: str) -> Dict[str, str]:
        """Parse the details string into a (possibly empty) dict."""
        if as_str is None:
            return {}

        detail_part, _, task_part = as_str.partition(" (")

        # drop type info from shapes
        detail_part = detail_part.replace("xf32", "")
        items = [x.split("=") for x in detail_part.split(";") if len(x) > 0]

        if "task_id" in task_part:
            items.append(["task_id", task_part.split(" ")[1].rstrip(")")])

        return {kv[0]: kv[1] for kv in items if len(kv) == 2}

    @staticmethod
    def parse_name(event_name: str) -> Dict[str, Any]:
        """Parse the kernel name to extract the kernel type."""
        # tasks have the format /task:{id}
        parent_name = event_name.split("/")[0]
        toks = parent_name.split(".")

        if toks[0] == "mojo":
            kernel_type = toks[-1]
        elif parent_name.startswith("__gen__PDEF"):
            kernel_type = "param_def"
        else:
            kernel_type = None

        return {"kernel_type": kernel_type, "task": "/task" in event_name}
