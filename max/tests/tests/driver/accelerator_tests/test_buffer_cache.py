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

# FYI that the buffer cache is persisted across tests so it is recommended to
# run tests one at time. Running multiple tests in parallel may cause failures.


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


@pytest.mark.skip(
    reason="GEX-2980: Host buffer cache is not working as expected"
)
def test_host_buffer_cache_enumerate(buffer_cache_config: None) -> None:
    print("====== test_host_buffer_cache_enumerate")

    # allocate 1MiB, 2MiB, 3MiB, 4MiB, 5MiB in increasing order
    # the sum of the sizes is far less than 100MiB
    for i in range(1, 6):
        size = i * MiB
        # Fails due to:
        #   ValueError: [Use only buffer cache mode]: No room left in buffer cache: cuda[0 - host] on 0x421a2c00 (size: 3MB ; free: 1MB ; cache_size: 4MB ; max_cache_size: 100MB)
        _ = alloc_pinned(size)


def test_host_buffer_cache_pyramid(buffer_cache_config: None) -> None:
    print("====== test_host_buffer_cache_pyramid")

    # allocate 1x90MiB, 2x45MiB, 3x30MiB, ..., 30x3MiB in a pyramid pattern.
    total_size = 90 * MiB
    for num_buffers in range(1, 31):
        # Compute the size of each buffer, aligning down to the nearest multiple of 256 KiB.
        # Unaligned allocations may result in memory fragmentation.
        size_per_buffer = align_down(total_size // num_buffers)
        size_per_buffer_str = to_human_readable_bytes(size_per_buffer)

        print(
            f"====== Allocating {num_buffers} buffers of size {size_per_buffer_str}"
        )
        bufs = [alloc_pinned(size_per_buffer) for _ in range(num_buffers)]

        print("====== Freeing buffers")
        # Each of the below lines is necessary to force the buffers to be freed
        # prior to the next iteration...

        # Clear the list of buffers to decr ref_cnt of DeviceBuffers.
        # As now ref_cnt==0, this calls enqueueEventHandler() for async free.
        bufs.clear()

        # Calls checkPendingWork() which checks for ready events and runs the
        # above handlers we enqueued earlier. This actually frees the buffers.
        Accelerator().synchronize()

        print("====== Done")
        print()
