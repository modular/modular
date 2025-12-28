# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


import pytest
from conftest import MiB, alloc_pinned


def test_oom(memory_manager_config: None) -> None:
    # We expect a OOM because we cannot allocate 101MiB when the limit is 100MiB.
    with pytest.raises(
        ValueError,
        match=r"\[Use only memory manager mode\]: No room left in memory manager: .*\[0 - host\] on .* \(size: 101MB ; free: 0B ; cache_size: 0B ; max_cache_size: 100MB\)",
    ):
        _ = alloc_pinned(101 * MiB)
