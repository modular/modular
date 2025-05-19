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

from gpu.memory import AddressSpace
from gpu import (
    WARP_SIZE,
)
from gpu.tcgen05 import (
    TensorMemory,
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_st,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from memory import UnsafePointer, stack_allocation
from gpu.sync import barrier
from gpu.host import DeviceContext
from layout._utils import ManagedLayoutTensor
from layout import Layout, LayoutTensor
from testing import assert_almost_equal
from gpu.id import thread_idx

alias M = 128
alias N = 8


fn tcgen05_st_ld_roundtrip_kernel(
    data: LayoutTensor[DType.float32, Layout.row_major(M, N), MutableAnyOrigin]
):
    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = thread_idx.x == 0

    var ptr_tmem_addr = stack_allocation[
        1, UInt32, address_space = AddressSpace.SHARED, alignment=16
    ]()

    alias width = 8
    alias num_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, num_cols)

    barrier()

    tmem_addr = ptr_tmem_addr[0]

    var data_st = SIMD[DType.float32, width]()
    for n in range(N):
        data_st[n] = thread_idx.x * N + n

    tcgen05_st[
        datapaths=16,
        bits=256,
        repeat=2,
        pack=False,
    ](tmem_addr, data_st)

    tcgen05_store_wait()

    var data_ld = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=2,
        type = DType.float32,
        pack=False,
        width=width,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock()
        tcgen05_dealloc[1](tmem_addr, num_cols)

    for n in range(N):
        if data_ld[n] == data_st[n]:
            data[thread_idx.x, n] = data_ld[n]


def test_tcgen05_st_ld_roundtrip(ctx: DeviceContext):
    var data = ManagedLayoutTensor[
        DType.float32,
        Layout.row_major(M, N),
    ](ctx)
    ctx.enqueue_function[tcgen05_st_ld_roundtrip_kernel](
        data.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(M),
    )
    ctx.synchronize()
    data_host = data.tensor()
    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                data_host[m, n],
                m * 8 + n,
                atol=1e-3,
                rtol=1e-4,
            )


def main():
    with DeviceContext() as ctx:
        test_tcgen05_st_ld_roundtrip(ctx)
