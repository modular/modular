# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test to ensure transfer completion notifications are delivered promptly.

Now the progress thread is responsible for sending out completion notifications.
"""

import time
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
from max.driver import Accelerator, Tensor
from max.nn.kv_cache import KVTransferEngine


def test_notification_delivery_is_prompt():
    TIMEOUT_SEND_S = 10
    TIMEOUT_RECV_S = 12
    MAX_ACCEPTABLE_LATENCY_S = 8
    GB = 1024 * 1024 * 1024

    num_blocks = 3
    bytes_per_block = int(0.25 * GB)

    # Create transfer engines
    sender_md_queue: Queue[Any] = Queue()
    receiver_md_queue: Queue[Any] = Queue()
    xfer_queue: Queue[Any] = Queue()
    done_queue: Queue[Any] = Queue()

    # Exit codes
    exit_codes = [-1, -1]

    def sender():
        acc = Accelerator(0)
        blocks = Tensor.from_numpy(
            np.ones((num_blocks, bytes_per_block), dtype=np.int8)
        ).to(acc)

        engine = KVTransferEngine(
            name="latency_sender",
            tensors=[blocks],
            total_num_pages=blocks.shape[0],
        )

        # Connect with receiver
        sender_md_queue.put(engine.metadata)
        remote_md = receiver_md_queue.get()
        engine.connect(remote_md)

        # Initiate transfer
        src_idxs = [0, 1, 2]
        dst_idxs = [0, 1, 2]
        xfer_req = engine.initiate_send_xfer(remote_md, src_idxs, dst_idxs)
        xfer_queue.put(xfer_req)

        # Notification should be delivered even though sender is asleep at the wheel.
        for i in range(TIMEOUT_SEND_S):
            print(f"Sender is sleeping... {i}s")
            time.sleep(1)
            if not done_queue.empty():
                assert done_queue.get() == "I am done!"
                break

        assert engine.is_complete(xfer_req), (
            "Transfer should be complete within 10 seconds"
        )

        exit_codes[0] = 0

    def receiver():
        acc = Accelerator(1)
        blocks = Tensor.from_numpy(
            np.ones((num_blocks, bytes_per_block), dtype=np.int8)
        ).to(acc)

        engine = KVTransferEngine(
            name="latency_receiver",
            tensors=[blocks],
            total_num_pages=blocks.shape[0],
        )

        # Connect with sender
        receiver_md_queue.put(engine.metadata)
        remote_md = sender_md_queue.get()
        engine.connect(remote_md)

        # Measure notification latency
        xfer_req = xfer_queue.get()
        start_time = time.time()
        is_done = False
        while not is_done and time.time() - start_time < TIMEOUT_RECV_S:
            is_done = engine.is_complete(xfer_req)
            print(f"Recv transfer status: {is_done}")
            time.sleep(0.25)

        if not is_done:
            raise TimeoutError(
                f"Transfer completion notification was not received after {TIMEOUT_RECV_S}s"
            )

        latency = time.time() - start_time

        done_queue.put("I am done!")
        transfer_bytes = bytes_per_block * num_blocks
        transfer_gb = transfer_bytes / GB
        bw = transfer_gb / latency
        # Note that because this test uses multi-thread instead of multi-process,
        # we use CUDA_COPY instead of CUDA_IPC. This has lower transfer bandwidth.
        print(
            f"Transfer of {transfer_gb:.2f} GB took {latency:.2f}s ({bw:.2f} GB/s)"
        )

        # Assert that latency is within acceptable bounds
        assert latency < MAX_ACCEPTABLE_LATENCY_S, (
            f"Notification latency {latency:.3f}s exceeds maximum acceptable "
            f"latency of {MAX_ACCEPTABLE_LATENCY_S}s. This suggests the UCX "
            "progress thread is not sending notifs promptly."
        )

        exit_codes[1] = 0

    # Run test
    sender_thread = Thread(target=sender)
    receiver_thread = Thread(target=receiver)

    sender_thread.start()
    receiver_thread.start()

    sender_thread.join()
    receiver_thread.join()

    assert exit_codes[0] == 0, "Sender thread failed"
    assert exit_codes[1] == 0, "Receiver thread failed"
