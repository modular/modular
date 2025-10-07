# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.comm.ep import EPCommInitializer, EPConfig
from test_common.graph_utils import is_b100_b200, is_h100_h200


@pytest.mark.skipif(
    not (is_h100_h200() or is_b100_b200()),
    reason="NVSHMEM library requires H100 or H200 or B200",
)
@pytest.mark.parametrize("n_devices", [4])
def test_init_ep(n_devices: int) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    # Initialize the device-contexts
    host = CPU(0)
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_with_host = [host, *devices]
    session = InferenceSession(devices=devices_with_host)

    config = EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=7168,
        top_k=8,
        n_experts=min(256, n_devices * 32),
        max_tokens_per_rank=128,
        n_gpus_per_node=n_devices,
        n_nodes=1,
    )
    ep_initializer = EPCommInitializer(config)
    ep_initializer.ep_init(session)

    all_tensors = ep_initializer.model_inputs()
    # check if the returned device pointers are not zero
    for tensor in all_tensors[2 * n_devices :]:
        assert np.all(tensor.to_numpy() != 0)
