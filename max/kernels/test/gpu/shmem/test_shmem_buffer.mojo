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
# RUN: %mojo-build %s -o %t
# RUN: %t

from math import iota

from shmem import SHMEMBuffer, SHMEMContext, shmem_launch
from testing import assert_equal


def test_buffer_copy(ctx: SHMEMContext):
    alias length = 1024

    var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](length)
    var host_buffer_2 = ctx.enqueue_create_host_buffer[DType.float32](length)
    var shmem_buffer = ctx.enqueue_create_buffer[DType.float32](length)

    iota(host_buffer.unsafe_ptr(), length)

    shmem_buffer.enqueue_copy_from(host_buffer)
    shmem_buffer.enqueue_copy_to(host_buffer_2)

    ctx.synchronize()

    for i in range(length):
        assert_equal(host_buffer[i], host_buffer_2[i])


def main():
    shmem_launch[test_buffer_copy]()
