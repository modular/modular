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

from pathlib import Path
from modular.utils.logging import warning
from modular.utils.typing import Any, Dict, List, Optional

EventDict = Dict[str, Any]
TraceDict = Dict[str, Any]
VersionInfoDict = Dict[str, Any]
EventList = List[EventDict]

# Perfetto Trace Event Format doc:
# https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU


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
    ):
        """
        Factory function to create a TimeTrace from a dictionary.
        Args:
            trace: The dictionary of events to wrap.
            check_build_type: Substring that is expected within the build-type
                              If None, then no check is performed.
                              The check is not case-sensitive.
            check_profiling_levels: Octal string of minimum required levels.
                                    If None, then no check is performed.
        Returns:
            The constructed TimeTrace.
        """
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
    def from_file(cls, tracefile: Path, **kwargs):
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

    @property
    def events(self) -> EventList:
        """Return all events in the trace"""
        return self.trace["traceEvents"]

    @property
    def version_info(self) -> VersionInfoDict:
        return self.trace["versionInfo"]

    @property
    def num_threads(self) -> int:
        """Return # of threads used in the trace"""
        return sum(x["name"] == "thread_name" for x in self.events)

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
        labels = ["Other", "LLCL", "Mem", "Mojo"]
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
