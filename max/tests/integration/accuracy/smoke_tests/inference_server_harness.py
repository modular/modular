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

import logging
import os
import signal
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import Popen, TimeoutExpired

import requests

logger = logging.getLogger(__name__)


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


def _server_is_ready() -> bool:
    try:
        return (
            requests.get("http://127.0.0.1:8000/health", timeout=1).status_code
            == 200
        )
    except requests.exceptions.RequestException:
        return False


def _gracefully_stop(process: Popen[bytes]) -> None:
    start_time = time.time()
    process.send_signal(signal.SIGINT)
    try:
        process.wait(25)
        shutdown_seconds = int(time.time() - start_time)
        logger.info(f"Server shutdown took {shutdown_seconds} seconds")
    except TimeoutExpired:
        logger.warning("Server did not stop after ctrl-c, trying SIGTERM")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(5)
        except ProcessLookupError:
            pass
        except TimeoutExpired:
            logger.warning("Process did not terminate gracefully, forcing kill")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(5)


@dataclass
class RunningServer:
    process: Popen[bytes]
    startup_time: float


@contextmanager
def start_server(
    cmd: list[str], timeout: int
) -> Generator[RunningServer, None, None]:
    env = os.environ.copy()

    if not _inside_bazel():
        # SGLang depends on ninja which is in the serve environment
        env["PYTHONSAFEPATH"] = "1"  # Avoids root dir `max` shadowing
        venv_bin = os.path.abspath(".venv-serve/bin")
        prev_path = env.get("PATH")
        env["PATH"] = f"{venv_bin}:{prev_path}" if prev_path else venv_bin

    start = time.monotonic()
    proc = Popen(cmd, start_new_session=True, env=env)
    try:
        deadline = start + timeout
        while time.monotonic() < deadline:
            if _server_is_ready():
                break
            if proc.poll() is not None:
                raise RuntimeError("Server process terminated unexpectedly")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"Server did not start in {timeout} seconds")
        yield RunningServer(proc, time.monotonic() - start)
    finally:
        if proc.poll() is None:
            try:
                _gracefully_stop(proc)
            except Exception:
                logger.exception("Failed to stop server")
