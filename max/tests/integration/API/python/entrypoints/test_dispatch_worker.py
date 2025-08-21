# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import multiprocessing
import os
import signal
import socket
import time
from typing import Any

import pytest
from max.entrypoints.workers.dispatch_worker import _dispatch_process_fn
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.kvcache_agent.dispatcher_transport import TransportMessage
from max.serve.process_control import ProcessControl


def _make_dispatcher_factory() -> DispatcherFactory:
    # Allocate an ephemeral TCP port for ZMQ ROUTER to avoid conflicts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()

    bind_addr = f"tcp://127.0.0.1:{port}"
    config = DispatcherConfig(
        transport=TransportType.DYNAMIC_ZMQ,
        transport_config=TransportFactory.DynamicZmqTransportConfig(
            bind_address=bind_addr
        ),
    )

    return DispatcherFactory[TransportMessage[dict]](
        config, transport_payload_type=TransportMessage[dict]
    )


def _wait_until(
    predicate: Any, timeout_s: float = 5.0, poll_s: float = 0.05
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


@pytest.mark.asyncio
async def test_dispatch_worker_start_and_cancel_via_monitor() -> None:
    settings = Settings()
    factory = _make_dispatcher_factory()

    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, name="TEST_DISPATCH_START_CANCEL")

    proc = ctx.Process(
        target=_dispatch_process_fn,
        args=(settings, pc, factory),
        daemon=True,
        name="TEST_DISPATCH_START_CANCEL_PROC",
    )
    proc.start()

    # Ensure startup completed
    assert _wait_until(pc.is_started, timeout_s=5.0)

    # Use ProcessMonitor to shut down gracefully
    from max.serve.process_control import (
        ProcessMonitor,  # local import to avoid cycles
    )

    monitor = ProcessMonitor(pc, proc)
    await monitor.shutdown()

    assert not proc.is_alive()
    assert proc.exitcode == 0


def test_dispatch_worker_handles_keyboard_interrupt() -> None:
    # Spawn the worker process directly so we can send it SIGINT
    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, name="TEST_DISPATCH_KBINT")

    factory = _make_dispatcher_factory()

    proc = ctx.Process(
        target=_dispatch_process_fn,
        args=(Settings(), pc, factory),
        daemon=True,
        name="TEST_DISPATCH_KBINT_PROC",
    )

    proc.start()

    # Wait until the worker reports started
    assert _wait_until(pc.is_started, timeout_s=5.0), (
        "Worker failed to report started"
    )

    # Send SIGINT to trigger KeyboardInterrupt in the child process
    pid = proc.pid
    assert pid is not None
    os.kill(pid, signal.SIGINT)

    # The worker should exit cleanly after SIGINT due to KeyboardInterrupt handling
    proc.join(timeout=5.0)

    assert not proc.is_alive(), "Worker did not exit after SIGINT"
    # When KeyboardInterrupt is caught and not re-raised, exitcode should be 0
    assert proc.exitcode == 0, (
        f"Unexpected exit code after SIGINT: {proc.exitcode}"
    )
