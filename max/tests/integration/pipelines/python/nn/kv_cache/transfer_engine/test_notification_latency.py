# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test to ensure transfer completion notifications are delivered promptly.

This test verifies that the UCX backend fix for immediate notification delivery
works correctly and prevents regressions of the 3+ second notification delay.
"""

import time
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
from max.driver import Tensor
from max.nn.kv_cache import KVTransferEngine, available_port


def test_notification_delivery_is_prompt():
    MAX_ACCEPTABLE_LATENCY_S = 8.0
    num_blocks = 3
    bytes_per_block = 3

    # Create transfer engines
    sender_md_queue: Queue[Any] = Queue()
    receiver_md_queue: Queue[Any] = Queue()
    xfer_queue: Queue[Any] = Queue()
    done_queue: Queue[Any] = Queue()

    # Exit codes
    exit_codes = [-1, -1]

    def sender():
        blocks = Tensor.from_numpy(
            np.ones((num_blocks, bytes_per_block), dtype=np.int32)
        )

        engine = KVTransferEngine(
            name="latency_sender",
            tensors=[blocks],
            total_num_pages=blocks.shape[0],
            listen_port=available_port(),
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
        for i in range(10):
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
        blocks = Tensor.from_numpy(
            np.ones((num_blocks, bytes_per_block), dtype=np.int32)
        )

        engine = KVTransferEngine(
            name="latency_receiver",
            tensors=[blocks],
            total_num_pages=blocks.shape[0],
            listen_port=available_port(),
        )

        # Connect with sender
        receiver_md_queue.put(engine.metadata)
        remote_md = sender_md_queue.get()
        engine.connect(remote_md)

        # Measure notification latency
        xfer_req = xfer_queue.get()
        start_time = time.time()
        is_done = False
        while not is_done:
            is_done = engine.is_complete(xfer_req)
            print(f"Recv transfer status: {is_done}")
            time.sleep(0.1)

        latency = time.time() - start_time

        print(f"Transfer completion notification latency: {latency:.3f}s")
        done_queue.put("I am done!")

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
