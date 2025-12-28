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


@pytest.mark.skip(
    reason="GEX-2980: Host memory manager is not working as expected"
)
def test_enumerate(memory_manager_config: None) -> None:
    # allocate 1MiB, 2MiB, 3MiB, 4MiB, 5MiB in increasing order
    # the sum of the sizes is far less than 100MiB
    for i in range(1, 6):
        size = i * MiB
        # Fails due to:
        #   ValueError: [Use only memory manager mode]: No room left in memory manager: cuda[0 - host] on 0x421a2c00 (size: 3MB ; free: 1MB ; cache_size: 4MB ; max_cache_size: 100MB)
        _ = alloc_pinned(size)
