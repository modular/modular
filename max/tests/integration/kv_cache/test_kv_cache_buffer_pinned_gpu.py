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
"""Tests that a KVCacheBuffer does not mix pinned and non-pinned shards."""

import pytest
from max.driver import CPU, Accelerator, Buffer, Device, Usage
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheBuffer


def _shard(device: Device, usage: Usage) -> Buffer:
    return Buffer(shape=(4, 8), dtype=DType.uint8, device=device, usage=usage)


def _kv_buffer(values: list[Buffer]) -> KVCacheBuffer:
    return KVCacheBuffer(
        leaf_id="leaf", replicates_kv_across_tp=False, values=values
    )


@pytest.mark.parametrize(
    "usage", [Usage.DEFAULT, Usage.STAGING, Usage.STAGING | Usage.UNTRACKED]
)
def test_uniform_usage_is_accepted(usage: Usage) -> None:
    gpu = Accelerator()
    _kv_buffer([_shard(gpu, usage), _shard(gpu, usage)])


def test_mixed_pinning_is_rejected() -> None:
    gpu = Accelerator()
    default, staging = _shard(gpu, Usage.DEFAULT), _shard(gpu, Usage.STAGING)
    assert staging.pinned and not default.pinned
    with pytest.raises(ValueError, match="all pinned or all non-pinned"):
        _kv_buffer([default, staging])


def test_host_staging_is_not_pinned() -> None:
    # Staging on a host device falls back to plain host memory, so it mixes
    # freely with default host buffers.
    _kv_buffer([_shard(CPU(), Usage.DEFAULT), _shard(CPU(), Usage.STAGING)])
