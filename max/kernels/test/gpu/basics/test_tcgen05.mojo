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

from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from gpu.memory import AddressSpace
from gpu.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_st,
    tcgen05_cp,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from memory import stack_allocation
from testing import assert_true
from gpu.mma_sm100 import MMASmemDescriptor
from layout import LayoutTensor, Layout, IntTuple


fn alloc_test_fn[cta_group: Int32]():
    var ptr_tmem_addr = UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED
    ]()
    var num_cols: UInt32 = 32
    tcgen05_alloc[cta_group](ptr_tmem_addr, num_cols)


fn test_tcgen05_alloc() raises:
    var asm1 = _compile_code[
        alloc_test_fn[1],
        target = get_gpu_target["sm_100a"](),
    ]().asm
    var asm2 = _compile_code[
        alloc_test_fn[2],
        target = get_gpu_target["sm_100a"](),
    ]().asm
    assert_true(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32" in asm1
    )
    assert_true(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32" in asm2
    )


fn alloc_dealloc_test_fn():
    var ptr_tmem_addr = UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED
    ]()
    var tmem_addr: UInt32 = 0
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    tcgen05_release_allocation_lock[1]()
    tcgen05_dealloc[1](tmem_addr, num_cols)


fn test_tcgen05_dealloc() raises:
    var asm = _compile_code[
        alloc_dealloc_test_fn,
        target = get_gpu_target["sm_100a"](),
    ]().asm
    assert_true(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" in asm
    )
    assert_true("tcgen05.dealloc.cta_group::1.sync.aligned.b32" in asm)


fn ld_test_fn():
    var ptr_tmem_addr = UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED
    ]()
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    var tmem_addr = ptr_tmem_addr[0]
    _ = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=64,
        dtype = DType.float32,
        pack=False,
        width=64,
    ](tmem_addr)
    tcgen05_load_wait()
    tcgen05_dealloc[1](tmem_addr, num_cols)


fn test_tcgen05_ld() raises:
    var asm = _compile_code[
        ld_test_fn,
        target = get_gpu_target["sm_100a"](),
    ]().asm
    assert_true("tcgen05.ld.sync.aligned.32x32b.x64.b32" in asm)
    assert_true("tcgen05.wait::ld.sync.aligned;" in asm)


fn st_test_fn():
    var ptr_tmem_addr = UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED
    ]()
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    var tmem_addr = ptr_tmem_addr[0]
    var data = SIMD[DType.float32, 64](0)
    tcgen05_st[
        datapaths=32,
        bits=32,
        repeat=64,
        pack=False,
    ](tmem_addr, data)
    tcgen05_store_wait()
    tcgen05_dealloc[1](tmem_addr, num_cols)


fn test_tcgen05_st() raises:
    var asm = _compile_code[
        st_test_fn,
        target = get_gpu_target["sm_100a"](),
    ]().asm
    assert_true("tcgen05.st.sync.aligned.32x32b.x64.b32" in asm)
    assert_true("tcgen05.wait::st.sync.aligned;" in asm)


fn cp_test_fn():
    var ptr_tmem_addr = UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED
    ]()
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    var tmem_addr = ptr_tmem_addr[0]

    var smem_tile = LayoutTensor[
        DType.float32,
        Layout(IntTuple(32, 32)),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    var s_desc = MMASmemDescriptor.create[0, 0](smem_tile.ptr)

    tcgen05_cp[
        cta_group=1,
        datapaths=128,
        bits=256,
        src_fmt="b6x16_p32",
        dst_fmt="b8x16",
        multicast="warpx2::01_23",
    ](tmem_addr, s_desc)


fn test_tcgen05_cp() raises:
    var asm = _compile_code[
        cp_test_fn,
        target = get_gpu_target["sm_100a"](),
    ]().asm
    assert_true(
        "tcgen05.cp.cta_group::1.128x256b.warpx2::01_23.b8x16.b6x16_p32" in asm
    )


fn main() raises:
    test_tcgen05_alloc()
    test_tcgen05_dealloc()
    test_tcgen05_ld()
    test_tcgen05_st()
    test_tcgen05_cp()
