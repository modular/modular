# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from queue import Queue
from threading import Thread

import numpy as np
from max.driver import Accelerator
from max.driver.tensor import Tensor
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    XferReqData,
    available_port,
)

total_num_pages = 10


def transfer_routine_sender(
    sender_md_queue: Queue[KVTransferEngineMetadata],
    receiver_md_queue: Queue[KVTransferEngineMetadata],
    xfer_queue_0: Queue[XferReqData],
    xfer_queue_1: Queue[XferReqData],
) -> None:
    device_0 = Accelerator(0)
    device_1 = Accelerator(1)

    t0 = np.arange(100, dtype=np.float32)
    t1 = np.arange(100, dtype=np.float32) + 1000
    tensors_1 = [
        Tensor.from_numpy(t0).to(device_0),
        Tensor.from_numpy(t1).to(device_1),
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

    xfer_0 = engine_1.initiate_send_xfer(
        remote_md,
        src_idxs=[0],
        dst_idxs=[0],
    )
    xfer_queue_0.put(xfer_0)

    xfer_1 = engine_1.initiate_send_xfer(
        remote_md,
        src_idxs=[3, 4],
        dst_idxs=[3, 4],
    )
    xfer_queue_1.put(xfer_1)

    engine_1.send_xfer_sync(xfer_0)
    engine_1.send_xfer_sync(xfer_1)

    assert np.array_equal(
        tensors_1[0].to_numpy(), np.arange(100, dtype=np.float32)
    )
    assert np.array_equal(
        tensors_1[1].to_numpy(), np.arange(100, dtype=np.float32) + 1000
    )

    engine_1.cleanup()


def transfer_routine_receiver(
    sender_md_queue: Queue[KVTransferEngineMetadata],
    receiver_md_queue: Queue[KVTransferEngineMetadata],
    xfer_queue_0: Queue[XferReqData],
    xfer_queue_1: Queue[XferReqData],
) -> None:
    device_2 = Accelerator(2)
    device_3 = Accelerator(3)

    t0 = np.zeros((100,), dtype=np.float32)
    t1 = np.zeros((100,), dtype=np.float32)
    tensors_2 = [
        Tensor.from_numpy(t0).to(device_2),
        Tensor.from_numpy(t1).to(device_3),
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

    xfer_0 = xfer_queue_0.get()
    engine_2.recv_xfer_sync(xfer_0)

    xfer_1 = xfer_queue_1.get()
    engine_2.recv_xfer_sync(xfer_1)

    assert np.array_equal(
        tensors_2[0].to_numpy()[:10], np.arange(10, dtype=np.float32)
    ), f"Expected arange(10) in first page, got {tensors_2[0].to_numpy()[:10]}"
    assert np.array_equal(
        tensors_2[1].to_numpy()[:10], np.arange(10, dtype=np.float32) + 1000
    ), (
        f"Expected arange(10)+1000 in first page, got {tensors_2[1].to_numpy()[:10]}"
    )

    elts_per_page = tensors_2[0].num_elements // total_num_pages
    expected_0 = np.arange(100, dtype=np.float32)[
        3 * elts_per_page : 5 * elts_per_page
    ]
    result_0 = tensors_2[0].to_numpy()[3 * elts_per_page : 5 * elts_per_page]
    assert np.array_equal(result_0, expected_0), (
        f"Expected {expected_0} for tensor 0 pages 3-4, got {result_0}"
    )

    expected_1 = (
        np.arange(100, dtype=np.float32)[3 * elts_per_page : 5 * elts_per_page]
        + 1000
    )
    result_1 = tensors_2[1].to_numpy()[3 * elts_per_page : 5 * elts_per_page]
    assert np.array_equal(result_1, expected_1), (
        f"Expected {expected_1} for tensor 1 pages 3-4, got {result_1}"
    )

    engine_2.cleanup()


def test_multi_tensor_transfer_threaded() -> None:
    """Test transfer between multiple tensors using threading."""
    sender_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    receiver_md_queue: Queue[KVTransferEngineMetadata] = Queue()
    xfer_queue_0: Queue[XferReqData] = Queue()
    xfer_queue_1: Queue[XferReqData] = Queue()

    sender_thread = Thread(
        target=transfer_routine_sender,
        args=(
            sender_md_queue,
            receiver_md_queue,
            xfer_queue_0,
            xfer_queue_1,
        ),
    )
    receiver_thread = Thread(
        target=transfer_routine_receiver,
        args=(
            sender_md_queue,
            receiver_md_queue,
            xfer_queue_0,
            xfer_queue_1,
        ),
    )

    sender_thread.start()
    receiver_thread.start()

    sender_thread.join()
    receiver_thread.join()

    print("\n" + "=" * 80)
    print("Multi-tensor threading test completed successfully!")
    print("=" * 80)
