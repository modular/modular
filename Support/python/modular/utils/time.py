# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Utility Library for Parsing Output from /usr/bin/time
"""

from dataclasses import dataclass
from pathlib import Path
from platform import system
from typing import Iterable, Optional

THIS_OS = system()


@dataclass
class RUsageResult:
    """Class to hold statistics printed by /usr/bin/time -v"""

    time_real_ns: Optional[int] = None
    time_user_ns: Optional[int] = None
    time_sys_ns: Optional[int] = None
    peak_rss_bytes: Optional[int] = None
    ctx_switches_voluntary: Optional[int] = None
    ctx_switches_involuntary: Optional[int] = None
    page_faults_major: Optional[int] = None
    page_faults_minor: Optional[int] = None

    @classmethod
    def from_file(
        cls, time_log_path: Path, os: str = THIS_OS
    ) -> "RUsageResult":
        """
        Parses the given time log and returns results.
        Args:
            time_log_path (Path): Path to logfile to be parsed.
            os (str): name of operating system (indicates log format).
        """
        with open(time_log_path, "r") as log:
            return RUsageResult.from_lines(log, os)

    @classmethod
    def from_lines(
        cls, time_log_lines: Iterable[str], os: str = THIS_OS
    ) -> "RUsageResult":
        """
        Parses the given time log and returns results.

        Args:
            time_log_lines (Iterable[str]): Contents of logfile to be parsed.
            os (str): name of operating system (indicates log format).
        """
        if os == "Darwin":
            return cls._from_lines_macos(time_log_lines)
        elif os == "Linux":
            return cls._from_lines_linux(time_log_lines)
        else:
            raise ValueError(f"Unknown operating system: {os}")

    @classmethod
    def _from_lines_macos(cls, time_log_lines: Iterable[str]) -> "RUsageResult":
        def get_int(val: str, idx: int = 0):
            return int(val.strip().split()[idx].strip())

        def get_float(val: str, idx: int = 0):
            return float(val.strip().split()[idx].strip())

        result = RUsageResult()
        for line in time_log_lines:
            if "real" in line:
                result.time_real_ns = int(get_float(line, 0) * 1e9)
            if "user" in line:
                result.time_user_ns = int(get_float(line, 2) * 1e9)
            if "sys" in line:
                result.time_sys_ns = int(get_float(line, 4) * 1e9)
            if "maximum resident set size" in line:
                result.peak_rss_bytes = get_int(line)
            elif "involuntary context switches" in line:
                result.ctx_switches_involuntary = get_int(line)
            elif "voluntary context switches" in line:
                result.ctx_switches_voluntary = get_int(line)
            elif "page faults" in line:
                result.page_faults_major = get_int(line)
        return result

    @classmethod
    def _from_lines_linux(cls, time_log_lines: Iterable[str]) -> "RUsageResult":
        def get_int(line: str):
            return int(line.split(":")[1].strip())

        def get_float(line: str):
            return float(line.split(":")[1].strip())

        def get_time(line: str):
            val = line.split("):")[1].strip()
            if val.count(":") == 1:
                hrs = 0
                mins, secs = val.split(":")
            elif val.count(":") == 2:
                hrs, mins, secs = val.split(":")
            else:
                raise ValueError(f"Unexpected time format: '{line}'")
            return (float(hrs) * 60 + float(mins)) * 60 + float(secs)

        result = RUsageResult()
        for line in time_log_lines:
            if "Elapsed (wall clock) time (h:mm:ss or m:ss)" in line:
                result.time_real_ns = int(get_time(line) * 1e9)
            if "User time (seconds)" in line:
                result.time_user_ns = int(get_float(line) * 1e9)
            if "System time (seconds)" in line:
                result.time_sys_ns = int(get_float(line) * 1e9)
            if "Maximum resident set size (kbytes)" in line:
                result.peak_rss_bytes = get_int(line) * 1000
            if "Voluntary context switches" in line:
                result.ctx_switches_voluntary = get_int(line)
            if "Involuntary context switches" in line:
                result.ctx_switches_involuntary = get_int(line)
            if "Major (requiring I/O) page faults" in line:
                result.page_faults_major = get_int(line)
            if "Minor (reclaiming a frame) page faults" in line:
                result.page_faults_minor = get_int(line)
        return result
