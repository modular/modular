# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# NIXL transfers can fail when multiple transfers are in the same bazel pytest target.
# This appears as a invalid file descriptor error.
# As such, this transfer test is alone in this file.

from queue import Queue
from threading import Thread

import numpy as np
import pytest
from max.driver import CPU, Device
from max.driver.tensor import Tensor
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    XferReqData,
)
from max.nn.kv_cache.paged_cache.transfer_engine import available_port


def transfer_routine_sender(
    engine: KVTransferEngine,
    remote: KVTransferEngineMetadata,
    queue: Queue,
    src_idxs: list[int],
    dst_idxs: list[int],
) -> None:
    xfer_req = engine.initiate_send_xfer(remote, src_idxs, dst_idxs)
    queue.put(xfer_req)
    engine.send_xfer_sync(xfer_req)


def transfer_routine_receiver(engine: KVTransferEngine, queue: Queue) -> None:
    xfer_req = queue.get()
    engine.recv_xfer_sync(xfer_req)


@pytest.mark.parametrize("device", [CPU()])
def test_send_recv_basic(device: Device) -> None:
    total_num_pages = 3
    elts_per_page = 3
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Tensor.from_numpy(np.arange(num_elts, dtype=np.int16) + 10).to(
        device
    )
    blocks_2 = Tensor.from_numpy(np.arange(num_elts, dtype=np.int16) + 80).to(
        device
    )

    engine_1 = KVTransferEngine(
        "engine_1",
        blocks_1,
        total_num_pages=total_num_pages,
        listen_port=available_port(),
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        blocks_2,
        total_num_pages=total_num_pages,
        listen_port=available_port(),
    )

    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    queue: Queue[XferReqData] = Queue()
    src_idxs = [2, 2]
    dst_idxs = [1, 0]
    thread_1 = Thread(
        target=transfer_routine_sender,
        args=(engine_1, engine_2.metadata, queue, src_idxs, dst_idxs),
    )
    thread_2 = Thread(target=transfer_routine_receiver, args=(engine_2, queue))

    # This is done via threads so wait_send_complete and wait_recv_complete
    # can progress in parallel. Doing this with single thread may cause hangs.
    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()

    expected_blocks_1 = np.array(
        [10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    expected_blocks_2 = np.array(
        [16, 17, 18, 16, 17, 18, 86, 87, 88],
    )
    assert np.array_equal(blocks_1.to_numpy(), expected_blocks_1)
    assert np.array_equal(blocks_2.to_numpy(), expected_blocks_2)

    engine_2.cleanup()
    engine_1.cleanup()
