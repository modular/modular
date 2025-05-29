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

"""Multi-process queue based on ZeroMQ. Tested for SPSC case."""

from __future__ import annotations

import logging
import queue
import tempfile
import uuid
import weakref
from collections import deque
from typing import Generic, Optional, TypeVar

import psutil
import zmq
import zmq.asyncio
import zmq.constants
from max.profiler import traced

logger = logging.getLogger("max.serve")

T = TypeVar("T")


def generate_zmq_ipc_path() -> str:
    """Generate a unique ZMQ IPC path."""
    base_rpc_path = tempfile.gettempdir()
    return f"ipc://{base_rpc_path}/{uuid.uuid4()}"


# Adapted from:
#  - vllm: https://github.com/vllm-project/vllm/blob/46c759c165a5a985ce62f019bf684e4a6109e41c/vllm/utils.py#L2093
#  - sglang: https://github.com/sgl-project/sglang/blob/efc52f85e2d5c9b31545d4092f2b361b6ff04d67/python/sglang/srt/utils.py#L783
def _open_zmq_socket(
    zmq_ctx: zmq.Context,
    path: str,
    mode: int,
) -> zmq.Socket:
    """Open a ZMQ socket with the proper bind/connect semantics."""
    mem = psutil.virtual_memory()
    socket = zmq_ctx.socket(mode)

    # Calculate buffer size based on system memory
    GIB = 1024**3
    total_mem_gb = mem.total / GIB
    available_mem_gb = mem.available / GIB
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem_gb > 32 and available_mem_gb > 16:
        buf_size = int(0.5 * GIB)
    else:
        buf_size = -1

    if mode == zmq.constants.PULL:
        socket.setsockopt(zmq.constants.RCVHWM, 0)
        socket.setsockopt(zmq.constants.RCVBUF, buf_size)
        socket.connect(path)
    elif mode == zmq.constants.PUSH:
        socket.setsockopt(zmq.constants.SNDHWM, 0)
        socket.setsockopt(zmq.constants.SNDBUF, buf_size)
        socket.bind(path)
    else:
        raise ValueError(f"Unknown Socket Mode: {mode}")

    return socket


class ZmqPushSocket(Generic[T]):
    def __init__(
        self,
        zmq_ctx: zmq.Context,
        zmq_endpoint: Optional[str] = None,
    ):
        self.zmq_endpoint = (
            zmq_endpoint
            if zmq_endpoint is not None
            else generate_zmq_ipc_path()
        )
        self.push_socket = _open_zmq_socket(
            zmq_ctx, self.zmq_endpoint, mode=zmq.PUSH
        )

        self._finalize = weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        self.push_socket.close(linger=0)

    def put_nowait(self, *args, **kwargs):
        self.put(*args, **kwargs, flags=zmq.NOBLOCK)

    def put(self, *args, **kwargs) -> None:
        while True:
            try:
                self.push_socket.send_pyobj(*args, **kwargs)

                # Exit since we succeeded
                break
            except zmq.ZMQError as e:
                # If we get EAGAIN, we just try again.
                # This could be due to:
                #   - the pull socket not being opened yet
                #   - a full queue
                if e.errno == zmq.EAGAIN:
                    continue

                # Unknown error, log it and let caller handle it
                logger.error(
                    f"Failed to send message on ZMQ socket for unknown reason: {e}"
                )
                raise e


class ZmqPullSocket(Generic[T]):
    def __init__(
        self, zmq_ctx: zmq.Context, zmq_endpoint: Optional[str] = None
    ):
        self.zmq_endpoint = (
            zmq_endpoint
            if zmq_endpoint is not None
            else generate_zmq_ipc_path()
        )
        self.pull_socket = _open_zmq_socket(
            zmq_ctx, self.zmq_endpoint, mode=zmq.PULL
        )
        self.local_queue: deque[T] = deque()

        self._finalize = weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        self.pull_socket.close(linger=0)

    def put_front_nowait(self, item: T):
        """A new method that allows us to add requests to the front of the queue."""
        self.local_queue.append(item)

    def _pull_from_socket(self, *args, **kwargs) -> T:
        try:
            return self.pull_socket.recv_pyobj(*args, **kwargs)

        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                raise queue.Empty()

            logger.error(
                f"Failed to receive message on ZMQ socket for unknown reason: {e}"
            )
            raise e

    def get(self, *args, **kwargs) -> T:
        if self.local_queue:
            return self.local_queue.pop()

        return self._pull_from_socket(*args, **kwargs)

    @traced
    def get_nowait(self) -> T:
        return self.get(flags=zmq.NOBLOCK)

    def qsize(self) -> int:
        """Return the size of the queue by repeatedly polling the ZmqSocket and
        adding the items to the local queue."""
        while True:
            try:
                item = self._pull_from_socket(flags=zmq.NOBLOCK)
                self.local_queue.appendleft(item)
            except queue.Empty:
                break

        return len(self.local_queue)

    def empty(self) -> bool:
        return self.qsize() == 0
