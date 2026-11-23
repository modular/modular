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
"""Which data-parallel degrees GLM-5.3-Flash constructs.

Data parallelism runs one attention replica per device: each device runs
whole KDA and sparse-MLA heads on its own replica's rows. A degree between
one and the device count would need tensor parallelism inside each replica,
which the graph does not build, so the config refuses it rather than letting
the caches split the batch one way and the graph another.
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig,
)

HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_HEADS = 8
PAGE_SIZE = 128
MAX_SEQ_LEN = 512


def _config(devices: list[DeviceRef], data_parallel_degree: int) -> None:
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HIDDEN_SIZE,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=devices,
        num_q_heads=NUM_HEADS,
    )
    Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=MAX_SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        data_parallel_degree=data_parallel_degree,
    )


def test_tensor_parallel_degree_is_accepted() -> None:
    """Tensor-parallel attention over every device constructs."""
    _config([DeviceRef.GPU(0), DeviceRef.GPU(1)], data_parallel_degree=1)


@pytest.mark.parametrize("degree", [2, 8])
def test_one_replica_per_device_is_accepted(degree: int) -> None:
    """The production DP8 shape, and its two-device miniature, construct."""
    devices = [DeviceRef.GPU(i) for i in range(degree)]
    _config(devices, data_parallel_degree=degree)


@pytest.mark.parametrize(("degree", "num_devices"), [(2, 4), (4, 8), (3, 1)])
def test_replicas_of_several_devices_are_refused(
    degree: int, num_devices: int
) -> None:
    """A degree that is neither one nor the device count is refused."""
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    with pytest.raises(ValueError, match="one data-parallel replica per"):
        _config(devices, data_parallel_degree=degree)
