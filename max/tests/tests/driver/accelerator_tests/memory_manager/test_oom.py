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


import pytest
from conftest import MemType, MiB


@pytest.mark.parametrize("mem_type", [MemType.PINNED, MemType.DEVICE])
def test_oom(memory_manager_config: None, mem_type: MemType) -> None:
    if mem_type == MemType.PINNED:
        pytest.skip("PINNED memory is not supported in the memory manager")

    # In memory-manager-only mode this must fail rather than falling back to a
    # raw device allocation, because the request exceeds the 100MiB manager.
    with pytest.raises(
        ValueError,
        match=r"^Memory manager cannot satisfy allocation \([^)]*\): .* on \[.*\]@.* \(request: 101MB ; free: 0B ; free_blocks: \d+ ; available: .* ; chunk_size: .* ; chunk_count: \d+ ; cache_size: 0B ; max_cache_size: 100MB\)",
    ):
        _ = mem_type.alloc(101 * MiB)
