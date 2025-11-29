# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import time
from typing import Any

import psutil


def collect_pids_for_port(port: int) -> list[int]:
    """Collects the PIDs of processes and child processes listening on a given port.

    Args:
        port (int): The port number to check.

    Returns:
        list[int]: A list of PIDs of processes listening on the specified port.
    """

    pids = set()

    def add_child_pids(pid: int) -> None:
        pids.add(pid)
        for proc in psutil.process_iter(["pid", "ppid"]):
            if proc.info["ppid"] == pid and proc.info["pid"] not in pids:
                add_child_pids(proc.info["pid"])

    for conn in psutil.net_connections(kind="inet"):
        if (
            conn.laddr.port == port
            and conn.status == psutil.CONN_LISTEN
            and conn.pid
        ):
            add_child_pids(conn.pid)

    return list(pids)


class CpuMetricsCollector:
    def __init__(self, pids: list[int]):
        self.pids: list[int] = pids
        self.clock_start: float | None = None
        self.clock_end: float | None = None
        self.cpu_times_start: dict[int, Any] = {}
        self.cpu_times_end: dict[int, Any] = {}

    def start(self) -> None:
        self.clock_start = time.monotonic()
        for pid in self.pids:
            try:
                proc = psutil.Process(pid)
                self.cpu_times_start[pid] = proc.cpu_times()
            except psutil.NoSuchProcess:
                self.cpu_times_start[pid] = None

    def stop(self) -> None:
        self.clock_end = time.monotonic()
        for pid in self.pids:
            try:
                proc = psutil.Process(pid)
                self.cpu_times_end[pid] = proc.cpu_times()
            except psutil.NoSuchProcess:
                self.cpu_times_end[pid] = None

    def dump_stats(self) -> dict[str, float]:
        if not self.clock_start or not self.clock_end:
            raise RuntimeError("Must call start and stop before dump_stats")

        # right now we just combine cpu time for all pids
        # because there's no easy way to know which pid is doing what
        user = 0.0
        system = 0.0
        elapsed = self.clock_end - self.clock_start
        if elapsed <= 0:
            raise RuntimeError("Elapsed time must be positive")

        for pid in self.pids:
            if self.cpu_times_start[pid] and self.cpu_times_end[pid]:
                user += (
                    self.cpu_times_end[pid].user
                    - self.cpu_times_start[pid].user
                )
                system += (
                    self.cpu_times_end[pid].system
                    - self.cpu_times_start[pid].system
                )
        return {
            "user": user,
            "user_percent": (user / elapsed) * 100,
            "system": system,
            "system_percent": (system / elapsed) * 100,
            "elapsed": elapsed,
        }
