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

import os

import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.nn.comm.ep import EPCommInitializer, EPConfig
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
