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

import logging
import queue
import tempfile
import threading
import uuid
import weakref
from collections import deque
from typing import Any, Optional, Union

import psutil
import zmq
import zmq.asyncio
import zmq.constants
from max.profiler import traced

logger = logging.getLogger("max.serve")


def _generate_zmq_ipc_path() -> str:
    base_rpc_path = tempfile.gettempdir()
    return f"ipc://{base_rpc_path}/{uuid.uuid4()}"


# Adapted from:
#  - vllm: https://github.com/vllm-project/vllm/blob/46c759c165a5a985ce62f019bf684e4a6109e41c/vllm/utils.py#L2093
#  - sglang: https://github.com/sgl-project/sglang/blob/efc52f85e2d5c9b31545d4092f2b361b6ff04d67/python/sglang/srt/utils.py#L783
def _open_zmq_socket(
    zmq_ctx: zmq.Context,
    path: str,
    mode: int,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:
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


class ZmqQueue:
    # One zmq context should be created per process AFTER process forks.
    # The Python GC is responsible for cleaning up the zmq context when the
    # process exits. A lock is needed to ensure that the zmq context is only
    # created once across all Queue instances.
    mutex: threading.Lock = threading.Lock()
    zmq_ctx: Optional[zmq.Context] = None

    def __init__(self):
        # Generate a unique path for the ZMQ socket.
        self.zmq_ipc_path = _generate_zmq_ipc_path()

        # These sockets are lazily initialized when needed.
        # They are initially None to allow this Queue class to be trivially
        # picklable.
        self.zmq_pull_socket: Optional[zmq.Socket] = None
        self.zmq_push_socket: Optional[zmq.Socket] = None

        # Register a finalizer to clean up resource handles.
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self) -> None:
        # https://github.com/vllm-project/vllm/blob/72a8639b68964ba50a019856f2fabd3c4fdbaa3f/vllm/v1/engine/core_client.py#L221
        # ZMQ context termination can hang if the sockets aren't explicitly closed first.
        if self.zmq_pull_socket is not None:
            self.zmq_pull_socket.close(linger=0)
            self.zmq_pull_socket = None

        if self.zmq_push_socket is not None:
            self.zmq_push_socket.close(linger=0)
            self.zmq_push_socket = None

    def _get_or_init_zmq_ctx(self) -> zmq.Context:
        with ZmqQueue.mutex:
            if ZmqQueue.zmq_ctx is None:
                ZmqQueue.zmq_ctx = zmq.Context(io_threads=2)
        return ZmqQueue.zmq_ctx

    def _get_or_init_pull_socket(self) -> zmq.Socket:
        zmq_ctx = self._get_or_init_zmq_ctx()
        if self.zmq_pull_socket is None:
            self.zmq_pull_socket = _open_zmq_socket(
                zmq_ctx, self.zmq_ipc_path, zmq.constants.PULL
            )

        return self.zmq_pull_socket

    def _get_or_init_push_socket(self) -> zmq.Socket:
        zmq_ctx = self._get_or_init_zmq_ctx()
        if self.zmq_push_socket is None:
            self.zmq_push_socket = _open_zmq_socket(
                zmq_ctx, self.zmq_ipc_path, zmq.constants.PUSH
            )

        return self.zmq_push_socket

    def get_nowait(self) -> Any:
        return self.get(flags=zmq.NOBLOCK)

    def put_nowait(self, item: Any) -> None:
        self.put(item, flags=zmq.NOBLOCK)

    def get(self, *args, **kwargs) -> Any:
        pull_socket = self._get_or_init_pull_socket()

        try:
            return pull_socket.recv_pyobj(*args, **kwargs)
        except zmq.ZMQError as e:
            # If we get EAGAIN, this is probably:
            #   - the queue is empty
            #   - the push socket is not yet ready
            if e.errno == zmq.EAGAIN:
                raise queue.Empty()

            # For all other unknown errors, log it and let caller handle it
            logger.error(
                f"Failed to receive message on ZMQ socket for unknown reason: {e}"
            )
            raise e

    def put(self, *args, **kwargs) -> None:
        push_socket = self._get_or_init_push_socket()

        while True:
            try:
                push_socket.send_pyobj(*args, **kwargs)

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


class ZmqDeque:
    """A wrapper around the ZmqQueue that adds a local queue.

    This allows the user to add requests to the front of the queue and check
    the queue size. This should probably only be used on the PULL side.
    """

    def __init__(self, queue: ZmqQueue):
        self.queue = queue
        self.local_queue: deque[Any] = deque()

    def put_front_nowait(self, item: Any):
        """A new method that allows us to add requests to the front of the queue."""
        self.local_queue.append(item)

    def put(self, *args, **kwargs) -> None:
        return self.queue.put(*args, **kwargs)

    def put_nowait(self, item: Any) -> None:
        return self.queue.put_nowait(item)

    def get(self, *args, **kwargs) -> Any:
        if self.local_queue:
            return self.local_queue.pop()
        return self.queue.get(*args, **kwargs)

    @traced
    def get_nowait(self) -> Any:
        if self.local_queue:
            return self.local_queue.pop()
        return self.queue.get_nowait()

    def qsize(self) -> int:
        """Return the size of the queue by repeatedly polling the ZmqQueue and
        adding the items to the local queue."""
        while True:
            try:
                item = self.queue.get_nowait()
                self.local_queue.appendleft(item)
            except queue.Empty:
                break

        return len(self.local_queue)

    def empty(self) -> bool:
        return self.qsize() == 0
