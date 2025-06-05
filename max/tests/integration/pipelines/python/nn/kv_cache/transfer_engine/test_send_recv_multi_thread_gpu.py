# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from queue import Queue
from threading import Thread

import numpy as np
from common import get_unique_port
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    XferReqData,
)


def test_send_recv_basic():
    # Queues for communication between threads
    sender_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    receiver_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    xfer_queue: Queue[XferReqData] = Queue()

    # Transfer parameters
    total_num_pages = 3
    src_idxs = [2, 2]
    dst_idxs = [1, 0]

    def transfer_routine_sender():
        device = Accelerator()

        blocks_np = np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
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
        expected_blocks = np.array(
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
        assert np.array_equal(blocks.to_numpy(), expected_blocks)

        # Release resources is skipped since it causes the following error:
        # `flush.c:58   UCX  ERROR req 0x7f274411a280: error during flush: Endpoint timeout`

        # TODO(E2EOPT-299) Reenable cleanup
        # engine.cleanup()

    def transfer_routine_receiver():
        device = Accelerator()

        blocks_np = np.array(
            [80, 81, 82, 83, 84, 85, 86, 87, 88],
        )
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
        expected_blocks = np.array(
            [16, 17, 18, 16, 17, 18, 86, 87, 88],
        )
        assert np.array_equal(blocks.to_numpy(), expected_blocks)

        # Release resources is skipped since it causes the following error:
        # `flush.c:58   UCX  ERROR req 0x7f274411a280: error during flush: Endpoint timeout`

        # TODO(E2EOPT-299) Reenable cleanup
        # engine.cleanup()

    thread_1 = Thread(target=transfer_routine_sender)
    thread_2 = Thread(target=transfer_routine_receiver)

    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()
