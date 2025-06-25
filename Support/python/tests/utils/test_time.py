# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform

import pytest
from modular.utils.subprocess import run_shell_command
from modular.utils.time import RUsageResult

time_log_linux = """
        Command being timed: "sha256sum stable-diffusion-v1.5-unet.tar"
        User time (seconds): 36.01
        System time (seconds): 2.06
        Percent of CPU this job got: 74%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:51.40
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 1920
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 132
        Voluntary context switches: 21563
        Involuntary context switches: 83
        Swaps: 0
        File system inputs: 13433840
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
"""

time_log_macos = """
       24.36 real        22.79 user         0.83 sys
             1589248  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                 214  page reclaims
                   1  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                  17  voluntary context switches
               12323  involuntary context switches
        242953331230  instructions retired
         74930184152  cycles elapsed
             1410048  peak memory footprint
"""


def test_from_lines_static_linux() -> None:
    rusage = RUsageResult.from_lines(time_log_linux.split("\n"), "Linux")
    assert rusage.time_real_ns == 51400000000
    assert rusage.time_user_ns == 36010000000
    assert rusage.time_sys_ns == 2060000000
    assert rusage.peak_rss_bytes == 1920000
    assert rusage.ctx_switches_voluntary == 21563
    assert rusage.ctx_switches_involuntary == 83
    assert rusage.page_faults_minor == 132
    assert rusage.page_faults_major == 0


def test_from_lines_static_macos() -> None:
    rusage = RUsageResult.from_lines(time_log_macos.split("\n"), "Darwin")
    assert rusage.time_real_ns == 24360000000
    assert rusage.time_user_ns == 22790000000
    assert rusage.time_sys_ns == 830000000
    assert rusage.peak_rss_bytes == 1589248
    assert rusage.ctx_switches_voluntary == 17
    assert rusage.ctx_switches_involuntary == 12323
    assert rusage.page_faults_minor is None
    assert rusage.page_faults_major == 1


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/time"), reason="binary unavailable"
)
def test_from_lines_dynamic() -> None:
    switch = "-l" if platform.system() == "Darwin" else "-v"
    command = f"/usr/bin/time {switch} echo Modular's Magnificent Menagerie"
    proc = run_shell_command(command.split(), capture_output=True)
    lines = proc.stderr.decode("utf-8").rstrip().split("\n")
    rusage = RUsageResult.from_lines(lines)
    assert rusage.time_real_ns is not None
    assert rusage.time_user_ns is not None
    assert rusage.time_sys_ns is not None
    assert rusage.peak_rss_bytes is not None
    assert rusage.ctx_switches_voluntary is not None
    assert rusage.ctx_switches_involuntary is not None
    assert rusage.page_faults_major is not None
