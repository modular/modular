# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

import multiprocessing as mp
import os
import re
import time

import numpy as np
import pytest
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import KVTransferEngine


def transfer_routine_sender(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    xfer_queue: mp.Queue,
    total_num_pages: int,
    src_idxs: list[int],
    dst_idxs: list[int],
    total_bytes: int,
    GB: float,
) -> None:
    # Enabling UCX debug logging only for sender
    os.environ["UCX_LOG_LEVEL"] = "debug"

    device = Accelerator(1)

    blocks_np = np.full(total_bytes, 42, dtype=np.int8)
    blocks = Tensor.from_numpy(blocks_np).to(device)

    # Create engine
    engine = KVTransferEngine(
        "engine_1",
        blocks,
        total_num_pages=total_num_pages,
    )

    # Connect with peer
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    t0 = time.time()
    xfer_req = engine.initiate_send_xfer(remote_md, src_idxs, dst_idxs)
    xfer_queue.put(xfer_req)
    engine.sync_and_release(xfer_req)
    t1 = time.time()
    bw = total_bytes / (t1 - t0) / GB
    ms = (t1 - t0) * 1000

    # This print statement is consumed by capfd. Unfortunately, we can't serialize
    # the capfd object into child to call capfd.disabled().
    print(
        f"[Sender] Transferring {total_bytes / GB:.2f} GB took {ms:.2f} ms ({bw:.2f} GB/s)"
    )

    # Check that the transfer speed is at least 1 GB/s
    # We found that CUDA_COPY yields ~.3GB/s while CUDA_IPC yields 100+GB/s
    # Note that CUDA_IPC requires memory to be allocated via `cuMemAlloc` and not
    # `cuMemAllocAsync`.
    assert bw > 1.0, f"Transfer speed is too low: {bw:.2f} GB/s"

    # Verify results
    assert (blocks.to_numpy() == 42).all()

    # Release resources is skipped since it causes `get_transfer_status` to raise NIXL_ERR_BACKEND :(
    # TODO(E2EOPT-299) Reenable cleanup
    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    xfer_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
) -> None:
    device = Accelerator(0)

    blocks_np = np.full(total_bytes, 99, dtype=np.int8)
    blocks = Tensor.from_numpy(blocks_np).to(device)

    # Create engine
    engine = KVTransferEngine(
        "engine_2",
        blocks,
        total_num_pages=total_num_pages,
    )

    # Connect with peer
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    xfer_req = xfer_queue.get()
    engine.sync_and_release(xfer_req)

    # Verify results
    assert (blocks.to_numpy() == 42).all()

    # Release resources is skipped since it causes `get_transfer_status` to raise NIXL_ERR_BACKEND
    # TODO(E2EOPT-299) Reenable cleanup
    engine.cleanup()


def test_send_recv_basic(capfd: pytest.CaptureFixture[str]) -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()
    receiver_md_queue: mp.Queue = ctx.Queue()
    xfer_queue: mp.Queue = ctx.Queue()

    # Transfer parameters
    GB = 1024 * 1024 * 1024
    total_bytes = int(8 * GB)
    total_num_pages = 2
    src_idxs = [0, 1]
    dst_idxs = [1, 0]

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            xfer_queue,
            total_num_pages,
            src_idxs,
            dst_idxs,
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

    out, err = capfd.readouterr()

    # Let some print statements actually be printed
    with capfd.disabled():
        print()
        print("-" * 80)
        for line in out.split("\n"):
            if "[Sender]" in line:
                print(line)
        print("-" * 80)

    # Check stdout
    assert re.search(r"Transferring .* GB took .* ms \(.* GB/s\)", out)
    assert "UCX  DEBUG register host memory on: cma, cuda_cpy, self, tcp" in out
    assert "UCX  DEBUG register cuda memory on: cuda_cpy, cuda_ipc" in out
    assert "UCX  DEBUG register cuda-managed memory on: cuda_cpy" in out
    assert "UCX  DEBUG no memory domain supports registering rocm memory" in out
    assert (
        "UCX  DEBUG cuMemcpyDtoDAsync_v2(dst, src, iov[0].length, stream) -> 0"
        in out
    )
