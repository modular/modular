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


@pytest.mark.parametrize("mem_type", [MemType.PINNED, MemType.DEVICE])
def test_enumerate(memory_manager_config: None, mem_type: MemType) -> None:
    # allocate 1MiB, 2MiB, 3MiB, 4MiB, 5MiB in increasing order
    # the sum of the sizes is far less than 100MiB
    for i in range(1, 6):
        size = i * MiB
        _ = mem_type.alloc(size)
