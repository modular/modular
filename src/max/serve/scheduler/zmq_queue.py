# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Multi-process queue based on ZeroMQ. Tested for SPSC case."""

import logging
import multiprocessing
import queue
import tempfile
import threading
import uuid
import weakref
from typing import Any, Optional, Union

import psutil
import zmq
import zmq.asyncio
import zmq.constants
from max.serve.scheduler.max_queue import AtomicInt, MaxQueue

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


class ZmqQueue(MaxQueue):
    # One zmq context should be created per process AFTER process forks.
    # The Python GC is responsible for cleaning up the zmq context when the
    # process exits. A lock is needed to ensure that the zmq context is only
    # created once across all Queue instances.
    mutex: threading.Lock = threading.Lock()
    zmq_ctx: Optional[zmq.Context] = None

    def __init__(
        self,
        ctx: multiprocessing.context.BaseContext,
    ):
        # Generate a unique path for the ZMQ socket.
        self.zmq_ipc_path = _generate_zmq_ipc_path()

        # These sockets are lazily initialized when needed.
        # They are initially None to allow this Queue class to be trivially
        # picklable.
        self.zmq_pull_socket: Optional[zmq.Socket] = None
        self.zmq_push_socket: Optional[zmq.Socket] = None

        # This counter is used to track the number of items in the queue.
        # This is a best effort estimate that may not be accurate.
        self.counter = AtomicInt(ctx)

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

        # Decrement the counter prior to recv
        count = self.counter.dec()
        if count == 0:
            raise queue.Empty()

        try:
            return pull_socket.recv_pyobj(*args, **kwargs)
        except zmq.ZMQError as e:
            # Since it failed, we increment the counter
            self.counter.inc()

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

                # Increment the counter after send
                self.counter.inc()

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

    def qsize(self) -> int:
        _ = self._get_or_init_pull_socket()
        return self.counter.value

    def empty(self) -> bool:
        _ = self._get_or_init_pull_socket()
        return self.qsize() == 0
