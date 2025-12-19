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
from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.support import to_human_readable_bytes

KiB = 1024
MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024


# The buffer cache is persistent in a process so it is recommended to run tests
# one at time in its own process. We split each test into its own file to isolate them.
# TODO: Figure out a way to consolidate all the tests into a single file.


@pytest.fixture
def buffer_cache_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Max cache size is 100MiB
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_HOST_BUFFER_CACHE_SIZE", f"{100 * MiB}"
    )
    # 2% of 100MiB is 2MiB per chunk
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_HOST_BUFFER_CACHE_CHUNK_PERCENT", "2"
    )
    # This toggles asserts in the buffer cache
    monkeypatch.setenv("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SELF_CHECK", "True")
    # This can be manually enabled for debugging
    monkeypatch.setenv("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_LOG", "False")


def alloc_pinned(size: int) -> Tensor:
    print(f"Allocating pinned buffer of size {to_human_readable_bytes(size)}")
    return Tensor(
        shape=(size,), dtype=DType.int8, device=Accelerator(), pinned=True
    )


def align_down(size: int, alignment: int = 256 * KiB) -> int:
    multiples = size // alignment
    return multiples * alignment
