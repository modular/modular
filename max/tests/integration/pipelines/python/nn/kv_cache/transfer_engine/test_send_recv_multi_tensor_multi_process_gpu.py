# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import multiprocessing as mp
import time

import numpy as np
import pytest
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import (
    KVTransferEngine,
    available_port,
)


def transfer_routine_sender(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    xfer_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
    src_idxs: list[int],
    dst_idxs: list[int],
    GB: float,
) -> None:
    """Multiprocessing sender routine for multi-tensor transfer."""
    device_0 = Accelerator(0)
    device_1 = Accelerator(1)

    # Create large tensors with distinct values for each tensor
    tensor_0_data = np.full(total_bytes, 42, dtype=np.int8)
    tensor_1_data = np.full(total_bytes, 84, dtype=np.int8)
    tensors_1 = [
        Tensor.from_numpy(tensor_0_data).to(device_0),
        Tensor.from_numpy(tensor_1_data).to(device_1),
    ]

    engine_1 = KVTransferEngine(
        "engine_1",
        tensors_1,
        total_num_pages=total_num_pages,
        listen_port=available_port(),
    )

    sender_md_queue.put(engine_1.metadata)
    remote_md = receiver_md_queue.get()
    engine_1.connect(remote_md)

    # Perform transfers
    start_time = time.time()
    xfer_req = engine_1.initiate_send_xfer(remote_md, src_idxs, dst_idxs)
    xfer_queue.put(xfer_req)
    engine_1.send_xfer_sync(xfer_req)
    end_time = time.time()

    transfer_time = end_time - start_time
    # Calculate actual bytes transferred (both tensors for the specified pages)
    bytes_per_tensor = total_bytes * len(src_idxs) // total_num_pages
    total_bytes_transferred = bytes_per_tensor * len(tensors_1)
    bw = total_bytes_transferred / transfer_time / GB
    ms = transfer_time * 1000

    print(
        f"[Sender MP] Transferred {total_bytes_transferred / GB:.4f} GB in {ms:.2f} ms ({bw:.2f} GB/s)"
    )

    # Check that the transfer speed is at least 1 GB/s
    # We found that CUDA_COPY yields ~.3GB/s while CUDA_IPC yields 100+GB/s
    assert bw > 1.0, f"Transfer speed is too low: {bw:.2f} GB/s"

    # Verify sender data unchanged
    assert np.array_equal(tensors_1[0].to_numpy(), tensor_0_data)
    assert np.array_equal(tensors_1[1].to_numpy(), tensor_1_data)

    engine_1.cleanup()


def transfer_routine_receiver(
    sender_md_queue: mp.Queue,
    receiver_md_queue: mp.Queue,
    xfer_queue: mp.Queue,
    total_num_pages: int,
    total_bytes: int,
) -> None:
    """Multiprocessing receiver routine for multi-tensor transfer."""
    device_2 = Accelerator(2)
    device_3 = Accelerator(3)

    # Create large tensors initialized to different values
    tensor_0_data = np.full(total_bytes, 99, dtype=np.int8)
    tensor_1_data = np.full(total_bytes, 77, dtype=np.int8)
    tensors_2 = [
        Tensor.from_numpy(tensor_0_data).to(device_2),
        Tensor.from_numpy(tensor_1_data).to(device_3),
    ]

    engine_2 = KVTransferEngine(
        "engine_2",
        tensors_2,
        total_num_pages=total_num_pages,
        listen_port=available_port(),
    )

    receiver_md_queue.put(engine_2.metadata)
    remote_md = sender_md_queue.get()
    engine_2.connect(remote_md)

    xfer_req = xfer_queue.get()
    engine_2.recv_xfer_sync(xfer_req)

    # Verify data received correctly - should now have sender values (42 and 84)
    expected_tensor_0 = np.full(total_bytes, 42, dtype=np.int8)
    expected_tensor_1 = np.full(total_bytes, 84, dtype=np.int8)

    assert np.array_equal(tensors_2[0].to_numpy(), expected_tensor_0), (
        f"Expected tensor 0 to have value 42, got {tensors_2[0].to_numpy()[:10]}"
    )
    assert np.array_equal(tensors_2[1].to_numpy(), expected_tensor_1), (
        f"Expected tensor 1 to have value 84, got {tensors_2[1].to_numpy()[:10]}"
    )

    engine_2.cleanup()


def test_multi_tensor_transfer_multiprocessing(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Test transfer between multiple tensors using multiprocessing."""
    # Use multiprocessing.Queue for inter-process communication
    ctx = mp.get_context("spawn")
    sender_md_queue: mp.Queue = ctx.Queue()
    receiver_md_queue: mp.Queue = ctx.Queue()
    xfer_queue: mp.Queue = ctx.Queue()

    GB = 1024 * 1024 * 1024
    total_bytes = int(12 * GB)
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
            total_bytes,
            src_idxs,
            dst_idxs,
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

    # Capture and display output from subprocesses
    out, err = capfd.readouterr()

    # Display the bandwidth information
    with capfd.disabled():
        print("\n" + "=" * 80)
        print("Multi-tensor multiprocessing test completed successfully!")
        for line in out.split("\n"):
            if "[Sender MP]" in line:
                print(line)
        print("=" * 80)
