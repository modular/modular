# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
from common import get_unique_port
from max.driver import CPU, Accelerator
from max.driver.tensor import Tensor
from max.dtype import DType
from max.nn.kv_cache import KVTransferEngine


def test_constructor():
    tensor = Tensor(DType.int8, (10, 10), device=CPU())

    # ok
    _ = KVTransferEngine(
        "abc",
        tensor,
        total_num_pages=2,
        listen_port=get_unique_port(),
    )
    _ = KVTransferEngine(
        "abc",
        tensor.to(Accelerator()),
        total_num_pages=2,
        listen_port=get_unique_port(),
    )

    # total_num_pages is 0
    with pytest.raises(ValueError):
        _ = KVTransferEngine(
            "abc",
            tensor,
            total_num_pages=0,
            listen_port=get_unique_port(),
        )

    # bytes is not divisible by total_num_pages
    with pytest.raises(ValueError):
        _ = KVTransferEngine(
            "abc",
            tensor,
            total_num_pages=3,
            listen_port=get_unique_port(),
        )


def test_initiate_send_xfer():
    device = CPU()
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
        "engine_1", blocks_1, total_num_pages, listen_port=get_unique_port()
    )
    engine_2 = KVTransferEngine(
        "engine_2", blocks_2, total_num_pages, listen_port=get_unique_port()
    )

    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    # ok
    _ = engine_1.initiate_send_xfer(
        engine_2.metadata, src_idxs=[2, 1], dst_idxs=[1, 0]
    )

    # oob src_idx
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_xfer(
            engine_2.metadata, src_idxs=[100], dst_idxs=[1]
        )

    # oob dst_idx
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_xfer(
            engine_2.metadata, src_idxs=[2, 0], dst_idxs=[100, 0]
        )

    # oob dst_idx
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_xfer(
            engine_2.metadata, src_idxs=[2], dst_idxs=[-1]
        )

    # mismatch lengths
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_xfer(
            engine_2.metadata, src_idxs=[2], dst_idxs=[0, 1]
        )

    # write to same dst page
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_xfer(
            engine_2.metadata, src_idxs=[2, 1], dst_idxs=[0, 0]
        )

    engine_1.cleanup()
    engine_2.cleanup()
