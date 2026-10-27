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

from std.memory.alloc import unsafe_alloc
from std.sys._assembly import inlined_assembly
from std.testing import assert_equal

from max.gpu.host import DeviceContext
from max.gpu.memory import CacheEviction
from max.gpu.memory.memory import _tma_cache_hint


def _createpolicy[priority: StaticString]() -> UInt64:
    return inlined_assembly[
        "createpolicy.fractional.L2::" + priority + ".b64 $0;",
        UInt64,
        constraints="=l",
    ]()


def kernel_policies(dst: MutPointer[UInt64, MutAnyOrigin]):
    dst[0] = _createpolicy["evict_normal"]()
    dst[1] = _createpolicy["evict_first"]()
    dst[2] = _createpolicy["evict_last"]()
    dst[3] = _createpolicy["evict_unchanged"]()


def test_tma_cache_hint_matches_createpolicy(ctx: DeviceContext) raises:
    var host = unsafe_alloc[UInt64](4)
    var dev = ctx.enqueue_create_buffer[.uint64](4)
    ctx.enqueue_function[kernel_policies](dev, grid_dim=(1,), block_dim=(1,))
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()

    var want: List[Int64] = [
        _tma_cache_hint[CacheEviction.EVICT_NORMAL](),
        _tma_cache_hint[CacheEviction.EVICT_FIRST](),
        _tma_cache_hint[CacheEviction.EVICT_LAST](),
        _tma_cache_hint[CacheEviction.EVICT_UNCHANGED](),
    ]
    for i in range(4):
        print("policy", i, "createpolicy", hex(host[i]), "hint", hex(want[i]))
    for i in range(4):
        assert_equal(host[i], UInt64(want[i]))
    host.unsafe_free()


def main() raises:
    with DeviceContext() as ctx:
        test_tma_cache_hint_matches_createpolicy(ctx)
