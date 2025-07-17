# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import multiprocessing as mp
import time

import numpy as np
from common import get_unique_port
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    XferReqData,
)


def transfer_routine_sender(
    sender_md_queue,  # noqa: ANN001
    receiver_md_queue,  # noqa: ANN001
    xfer_queue,  # noqa: ANN001
    total_num_pages,  # noqa: ANN001
    src_idxs,  # noqa: ANN001
    dst_idxs,  # noqa: ANN001
    total_bytes,  # noqa: ANN001
    GB,  # noqa: ANN001
) -> None:
    device = Accelerator(1)

    blocks_np = np.full(total_bytes, 42, dtype=np.int8)
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
    t0 = time.time()
    xfer_req = engine.initiate_send_xfer(remote_md, src_idxs, dst_idxs)
    xfer_queue.put(xfer_req)
    engine.send_xfer_sync(xfer_req)
    t1 = time.time()
    bw = total_bytes / (t1 - t0) / GB
    print(
        f"Transferring {total_bytes / GB:.2f} GB took {t1 - t0:.2f} seconds ({bw:.2f} GB/s)",
        flush=True,
    )

    # Verify results
    expected_blocks = np.full(total_bytes, 42, dtype=np.int8)
    assert np.array_equal(blocks.to_numpy(), expected_blocks)

    # Release resources is skipped since it causes `get_transfer_status` to raise NIXL_ERR_BACKEND :(
    # TODO(E2EOPT-299) Reenable cleanup
    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue,  # noqa: ANN001
    receiver_md_queue,  # noqa: ANN001
    xfer_queue,  # noqa: ANN001
    total_num_pages,  # noqa: ANN001
    total_bytes,  # noqa: ANN001
) -> None:
    device = Accelerator(0)

    blocks_np = np.full(total_bytes, 99, dtype=np.int8)
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
    expected_blocks = np.full(total_bytes, 42, dtype=np.int8)
    assert np.array_equal(blocks.to_numpy(), expected_blocks)

    # Release resources is skipped since it causes `get_transfer_status` to raise NIXL_ERR_BACKEND
    # TODO(E2EOPT-299) Reenable cleanup
    engine.cleanup()


def test_send_recv_basic() -> None:
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue[KVTransferEngineMetadata] = ctx.Queue()
    receiver_md_queue: mp.Queue[KVTransferEngineMetadata] = ctx.Queue()
    xfer_queue: mp.Queue[XferReqData] = ctx.Queue()

    # Transfer parameters
    GB = 1024 * 1024 * 1024
    total_bytes = int(0.5 * GB)
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
