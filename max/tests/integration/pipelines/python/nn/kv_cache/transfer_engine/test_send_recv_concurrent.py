# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import KVTransferEngine, XferReqData, available_port

"""
This test launches 256 concurrent transfers at once.
"""


def transfer_routine_sender(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    xfer_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
    GB: float,
) -> None:
    device = Accelerator(0)

    blocks_np = np.full(total_bytes, 42, dtype=np.int8)
    blocks = Tensor.from_numpy(blocks_np).to(device)

    # Create engine
    engine = KVTransferEngine(
        "engine_1",
        [blocks],
        total_num_pages=total_num_pages,
        listen_port=available_port(),
    )

    # Connect with peer
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    t0 = time.time()
    xfer_reqs: list[XferReqData] = []

    for idx in range(total_num_pages):
        xfer_req = engine.initiate_send_xfer(remote_md, [idx], [idx])
        xfer_queue.put(xfer_req)
        xfer_reqs.append(xfer_req)

    for xfer_req in xfer_reqs:
        engine.sync_and_release(xfer_req)

    t1 = time.time()
    bw = total_bytes / (t1 - t0) / GB
    ms = (t1 - t0) * 1000

    print(
        f"[SENDER] Transferring {total_bytes / GB:.2f} GB took {ms:.2f} ms ({bw:.2f} GB/s)"
    )

    # Verify results
    assert (blocks.to_numpy() == 42).all()

    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    xfer_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
) -> None:
    device = Accelerator(1)

    blocks_np = np.full(total_bytes, 99, dtype=np.int8)
    blocks = Tensor.from_numpy(blocks_np).to(device)

    # Create engine
    engine = KVTransferEngine(
        "engine_2",
        [blocks],
        total_num_pages=total_num_pages,
        listen_port=available_port(),
    )

    # Connect with peer
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    for _ in range(total_num_pages):
        xfer_req = xfer_queue.get()
        engine.sync_and_release(xfer_req)

    # TODO: Verify results
    # assert (blocks.to_numpy() == 42).all()

    engine.cleanup()


def test_send_recv_basic() -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()
    receiver_md_queue: mp.Queue = ctx.Queue()
    xfer_queue: mp.Queue = ctx.Queue()

    # Transfer parameters
    GB = 1024 * 1024 * 1024
    total_bytes = int(6 * GB)
    total_num_pages = 256

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            xfer_queue,
            total_num_pages,
            total_bytes,
            GB,
        ),
    )
    receiver_proc = ctx.Process(
        target=transfer_routine_receiver,
        args=(
            sender_md_queue,
            receiver_md_queue,
            xfer_queue,
            total_num_pages,
            total_bytes,
        ),
    )

    sender_proc.start()
    receiver_proc.start()

    sender_proc.join()
    receiver_proc.join()

    assert sender_proc.exitcode == 0, (
        f"Sender process failed with exit code {sender_proc.exitcode}"
    )
    assert receiver_proc.exitcode == 0, (
        f"Receiver process failed with exit code {receiver_proc.exitcode}"
    )
