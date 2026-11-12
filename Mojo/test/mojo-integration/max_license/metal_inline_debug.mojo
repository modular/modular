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

# REQUIRES: APPLE-GPU
# RUN: %mojo -O0 -g %s
# RUN: %mojo -O0 -g1 %s

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal


@always_inline
def foo(a: UInt32) -> UInt32:
    return a * 2


def bar(a: Pointer[UInt32, MutAnyOrigin]):
    a[unsafe_offset=Int(global_idx.x)] = foo(UInt32(global_idx.x))


def main() raises:
    var a = DeviceContext()
    var b = a.enqueue_create_buffer[DType.uint32](4)
    with b.map_to_host() as c:
        for d in range(4):
            c[d] = UInt32(0xDEADBEEF)
    a.enqueue_function[bar](b.unsafe_ptr(), grid_dim=1, block_dim=4)
    a.synchronize()
    with b.map_to_host() as c:
        for d in range(4):
            assert_equal(c[d], UInt32(d * 2))
