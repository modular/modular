# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import ctypes
import ctypes.util
import dataclasses
import os

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.nn import ep_config as ep_config_module
from max.nn.ep_config import EPConfig
from max.nn.legacy.comm.ep import EPCommInitializer
from test_common.graph_utils import gpu_warp_size


@pytest.mark.parametrize("n_devices", [4])
def test_init_ep(n_devices: int) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    n_nodes = int(os.environ.get("SHMEM_TOTAL_NODES", "1"))

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
        n_experts=min(256, n_devices * (1024 // gpu_warp_size())),
        max_tokens_per_rank=128,
        n_gpus_per_node=n_devices,
        n_nodes=n_nodes,
    )
    ep_initializer = EPCommInitializer(config)
    ep_initializer.ep_init(session)

    all_tensors = ep_initializer.model_inputs()
    # check if the returned device pointers are not zero
    group_0_send_buf_ptrs = all_tensors[2 * n_devices]
    assert np.all(group_0_send_buf_ptrs.to_numpy() != 0)

    # The combine send buffer is only allocated when there are multiple nodes.
    if n_nodes > 1:
        group_1_send_buf_ptrs = all_tensors[2 * n_devices + 1]
        assert np.all(group_1_send_buf_ptrs.to_numpy() != 0)

    group_0_recv_buf_ptrs = all_tensors[2 * n_devices + 2]
    assert np.all(group_0_recv_buf_ptrs.to_numpy() != 0)
    group_1_recv_buf_ptrs = all_tensors[2 * n_devices + 3]
    assert np.all(group_1_recv_buf_ptrs.to_numpy() != 0)
    group_0_recv_count_ptrs = all_tensors[2 * n_devices + 4]
    assert np.all(group_0_recv_count_ptrs.to_numpy() != 0)
    group_1_recv_count_ptrs = all_tensors[2 * n_devices + 5]
    assert np.all(group_1_recv_count_ptrs.to_numpy() != 0)


_HIP_MEMCPY_DEVICE_TO_HOST = 2
_P2P_SMALL_DISPATCH_BYTES = ep_config_module._EP_P2P_SMALL_DISPATCH_BYTES
_P2P_SMALL_COMBINE_BYTES = ep_config_module._EP_P2P_SMALL_COMBINE_BYTES


def _read_device_bytes(address: int, n_bytes: int) -> bytes:
    """Copies ``n_bytes`` at a device address to the host with the HIP runtime."""
    hip = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
    host = ctypes.create_string_buffer(n_bytes)
    status = hip.hipMemcpy(
        host,
        ctypes.c_void_p(address),
        ctypes.c_size_t(n_bytes),
        _HIP_MEMCPY_DEVICE_TO_HOST,
    )
    assert status == 0, f"hipMemcpy failed with status {status}"
    return host.raw


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the P2P small path is AMD-only"
)
@pytest.mark.parametrize("n_devices", [4])
def test_init_ep_p2p_small_headers(n_devices: int) -> None:
    """With the small path on, ``ep.init`` sizes the counters, publishes each
    receive buffer past its flag header and zeroes the header; the flags rely
    on that zero as their epoch-0 sentinel."""
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )
    devices = [Accelerator(id) for id in range(n_devices)]
    session = InferenceSession(devices=[CPU(0), *devices])

    config = EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=7168,
        top_k=8,
        n_experts=min(256, n_devices * (1024 // gpu_warp_size())),
        max_tokens_per_rank=128,
        n_gpus_per_node=n_devices,
        n_nodes=1,
        fused_shared_expert=True,
        tokens_split_evenly=True,
    )
    assert config.uses_p2p_small_path
    ep_initializer = EPCommInitializer(config)
    ep_initializer.ep_init(session)
    inputs = ep_initializer.model_inputs()

    plain = EPCommInitializer(
        dataclasses.replace(config, tokens_split_evenly=False)
    )
    assert (
        inputs[0].num_elements
        == plain.atomic_counter_size + ep_config_module._EP_P2P_SMALL_WORDS
    )

    dispatch_recv = inputs[2 * n_devices + 2].to_numpy()
    combine_recv = inputs[2 * n_devices + 3].to_numpy()
    for ptr in dispatch_recv:
        header = _read_device_bytes(
            int(ptr) - _P2P_SMALL_DISPATCH_BYTES, _P2P_SMALL_DISPATCH_BYTES
        )
        assert not any(header)
    for ptr in combine_recv:
        header = _read_device_bytes(
            int(ptr) - _P2P_SMALL_COMBINE_BYTES, _P2P_SMALL_COMBINE_BYTES
        )
        assert not any(header)
