# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
This test currently passes but prints the following warnings:
```
    event_set.c:176  UCX  ERROR epoll_ctl(event_fd=73, DEL, fd=69) failed: Bad file descriptor
        async.c:585  UCX  WARN  failed to remove async handler 0x12be6d00 [id=69 ref 1] ???() : Input/output error
```
"""

import multiprocessing as mp

import numpy as np
import pytest
from common import get_unique_port
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    XferReqData,
)


def transfer_routine_sender(
    sender_md_queue,
    receiver_md_queue,
    xfer_queue,
    total_num_pages,
    src_idxs,
    dst_idxs,
) -> None:
    device = Accelerator()

    blocks_np = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])
    blocks = Tensor.from_numpy(blocks_np).to(device)

    # Create engine
    engine = KVTransferEngine(
        "engine_1", blocks, total_num_pages, listen_port=get_unique_port()
    )

    # Connect with peer
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    xfer_req = engine.initiate_send_xfer(remote_md, src_idxs, dst_idxs)
    xfer_queue.put(xfer_req)
    engine.send_xfer_sync(xfer_req)

    # Verify results
    expected_blocks = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])
    assert np.array_equal(blocks.to_numpy(), expected_blocks)

    # Release resources is skipped since it causes `get_transfer_status` to raise NIXL_ERR_BACKEND :(
    # TODO(E2EOPT-299) Reenable cleanup
    # engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue,
    receiver_md_queue,
    xfer_queue,
    total_num_pages,
) -> None:
    device = Accelerator()

    blocks_np = np.array([80, 81, 82, 83, 84, 85, 86, 87, 88])
    blocks = Tensor.from_numpy(blocks_np).to(device)

    # Create engine
    engine = KVTransferEngine(
        "engine_2", blocks, total_num_pages, listen_port=get_unique_port()
    )

    # Connect with peer
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Perform transfer
    xfer_req = xfer_queue.get()
    engine.recv_xfer_sync(xfer_req)

    # Verify results
    expected_blocks = np.array([16, 17, 18, 16, 17, 18, 86, 87, 88])
    assert np.array_equal(blocks.to_numpy(), expected_blocks)

    # Release resources is skipped since it causes `get_transfer_status` to raise NIXL_ERR_BACKEND
    # TODO(E2EOPT-299) Reenable cleanup
    # engine.cleanup()


@pytest.mark.skip(reason="❄️")
def test_send_recv_basic() -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue[KVTransferEngineMetadata] = ctx.Queue()
    receiver_md_queue: mp.Queue[KVTransferEngineMetadata] = ctx.Queue()
    xfer_queue: mp.Queue[XferReqData] = ctx.Queue()

    # Transfer parameters
    total_num_pages = 3
    src_idxs = [2, 2]
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
        ),
    )
    receiver_proc = ctx.Process(
        target=transfer_routine_receiver,
        args=(
            sender_md_queue,
            receiver_md_queue,
            xfer_queue,
            total_num_pages,
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
