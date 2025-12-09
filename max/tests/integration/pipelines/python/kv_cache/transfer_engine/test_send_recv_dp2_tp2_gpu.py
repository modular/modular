# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test DP=2, TP=2 transfers using GPUs.

This test validates that transfer engine correctly handles:

- Multiple DP replicas (DP=2)
- Multiple TP shards per replica (TP=2)
- Per-replica transfers with src_replica_idx and dst_replica_idx parameters
"""

import multiprocessing as mp
import time

import numpy as np
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.kv_cache import KVTransferEngine


def full(total_bytes: int, value: int, accelerator_idx: int) -> Tensor:
    return Tensor.from_numpy(np.full(total_bytes, value, dtype=np.int8)).to(
        Accelerator(accelerator_idx)
    )


def transfer_routine_sender(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    transfer_queue_0: mp.Queue,
    transfer_queue_1: mp.Queue,
    sender_done_queue: mp.Queue,
    receiver_done_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
    GB: float,
) -> None:
    """Sender routine for DP=2, TP=2 transfer."""
    # DP=2, TP=2: 4 GPUs total for sender
    # Replica 0: GPU 0, 1
    # Replica 1: GPU 2, 3
    replica_0_tensors = [
        full(total_bytes, value=10, accelerator_idx=0),
        full(total_bytes, value=11, accelerator_idx=1),
    ]
    replica_1_tensors = [
        full(total_bytes, value=20, accelerator_idx=2),
        full(total_bytes, value=21, accelerator_idx=3),
    ]

    # Create engine with DP=2, TP=2
    engine = KVTransferEngine(
        "sender_engine",
        [replica_0_tensors, replica_1_tensors],
        total_num_pages=total_num_pages,
    )

    # Connect with receiver
    sender_md_queue.put(engine.metadata)
    remote_md = receiver_md_queue.get()
    engine.connect(remote_md)

    # Transfer from replica 0
    start_time_0 = time.time()
    transfer_req_0 = engine.initiate_send_transfer(
        remote_md,
        src_idxs=[0, 1],
        dst_idxs=[0, 1],
        src_replica_idx=1,
        dst_replica_idx=0,
    )
    transfer_queue_0.put(transfer_req_0)
    engine.sync_and_release(transfer_req_0)
    end_time_0 = time.time()

    # Transfer from replica 1
    start_time_1 = time.time()
    transfer_req_1 = engine.initiate_send_transfer(
        remote_md,
        src_idxs=[0, 1],
        dst_idxs=[0, 1],
        src_replica_idx=0,
        dst_replica_idx=1,
    )
    transfer_queue_1.put(transfer_req_1)
    engine.sync_and_release(transfer_req_1)
    end_time_1 = time.time()

    # Calculate bandwidth for each replica
    total_bytes_transferred = total_bytes * 2  # 2 TP shards

    bw_0 = total_bytes_transferred / (end_time_0 - start_time_0) / GB
    bw_1 = total_bytes_transferred / (end_time_1 - start_time_1) / GB

    print(
        f"[Sender] Replica 0 -> Replica 0: {total_bytes_transferred / GB:.4f} GB "
        f"in {(end_time_0 - start_time_0) * 1000:.2f} ms ({bw_0:.2f} GB/s)"
    )
    print(
        f"[Sender] Replica 0 -> Replica 1: {total_bytes_transferred / GB:.4f} GB "
        f"in {(end_time_1 - start_time_1) * 1000:.2f} ms ({bw_1:.2f} GB/s)"
    )

    # Verify bandwidth is reasonable
    assert bw_0 > 1.0, f"Replica 0 transfer too slow: {bw_0:.2f} GB/s"
    assert bw_1 > 1.0, f"Replica 1 transfer too slow: {bw_1:.2f} GB/s"

    sender_done_queue.put(None)
    receiver_done_queue.get()
    engine.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    transfer_queue_0: mp.Queue,
    transfer_queue_1: mp.Queue,
    sender_done_queue: mp.Queue,
    receiver_done_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
) -> None:
    """Receiver routine for DP=2, TP=2 transfer."""
    # DP=2, TP=2: 4 GPUs total for receiver
    # Replica 0: GPU 1, 3
    # Replica 1: GPU 2, 2
    replica_0_tensors = [
        full(total_bytes, value=99, accelerator_idx=1),
        full(total_bytes, value=98, accelerator_idx=3),
    ]
    replica_1_tensors = [
        full(total_bytes, value=97, accelerator_idx=2),
        full(total_bytes, value=96, accelerator_idx=2),
    ]

    # Create engine with DP=2, TP=2
    engine = KVTransferEngine(
        "receiver_engine",
        [replica_0_tensors, replica_1_tensors],
        total_num_pages=total_num_pages,
    )

    # Connect with sender
    receiver_md_queue.put(engine.metadata)
    remote_md = sender_md_queue.get()
    engine.connect(remote_md)

    # Receive transfer for replica 0
    transfer_req_0 = transfer_queue_0.get()
    engine.sync_and_release(transfer_req_0)

    # Receive transfer for replica 1
    transfer_req_1 = transfer_queue_1.get()
    engine.sync_and_release(transfer_req_1)

    # Verify received data
    assert (replica_0_tensors[0].to_numpy() == 20).all()
    assert (replica_0_tensors[1].to_numpy() == 21).all()
    assert (replica_1_tensors[0].to_numpy() == 10).all()
    assert (replica_1_tensors[1].to_numpy() == 11).all()

    receiver_done_queue.put(None)
    sender_done_queue.get()
    engine.cleanup()


def test_dp2_tp2_transfer_multiprocessing() -> None:
    """Test DP=2, TP=2 transfer using 8 GPUs (4 for sender, 4 for receiver).

    This test validates:
    - Engine construction with DP=2, TP=2
    - Metadata structure for 2D tensor layout
    - Per-replica transfers using src_replica_idx and dst_replica_idx
    - Correct data transfer for each replica independently
    """
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()
    receiver_md_queue: mp.Queue = ctx.Queue()
    transfer_queue_0: mp.Queue = ctx.Queue()  # For replica 0
    transfer_queue_1: mp.Queue = ctx.Queue()  # For replica 1
    sender_done_queue: mp.Queue = ctx.Queue()
    receiver_done_queue: mp.Queue = ctx.Queue()

    GB = 1024 * 1024 * 1024
    total_bytes = int(0.5 * GB)
    total_num_pages = 2

    sender_proc = ctx.Process(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            transfer_queue_0,
            transfer_queue_1,
            sender_done_queue,
            receiver_done_queue,
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
            transfer_queue_0,
            transfer_queue_1,
            sender_done_queue,
            receiver_done_queue,
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
