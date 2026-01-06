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
from conftest import MemType, MiB
from max.driver import Accelerator


@pytest.mark.parametrize("mem_type", [MemType.PINNED, MemType.DEVICE])
def test_limit(memory_manager_config: None, mem_type: MemType) -> None:
    # The cache has 100MiB so we try to alloc/free 100MiB a bunch of times.
    for _ in range(321):
        t = mem_type.alloc(100 * MiB)
        # This `del t` is needed.
        # Otherwise the Garbage Collector may delay the free until after the sync.
        del t

        # Synchronizing is necessary to ensure that allocated memory is returned
        # to the buffer cache for pinned memory.
        if mem_type == MemType.PINNED:
            Accelerator().synchronize()
