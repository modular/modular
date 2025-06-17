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

from collections import OptionalReg
from sys import sizeof

from layout.element import Element
from layout.int_tuple import IntTuple, size
from layout.layout import (
    Layout,
    LayoutList,
    MakeTileLayoutList,
    coalesce,
    complement,
    composition,
    make_layout,
    print_layout,
    right_inverse,
    zipped_divide,
)
from layout.layout_tensor import LayoutTensor, _compute_distribute_layout
from layout.swizzle import Swizzle, make_swizzle
from testing import assert_equal

from utils import StaticTuple


fn print_swizzle(thread_layout: Layout, swizzle: Swizzle):
    for tid in range(thread_layout.size()):
        print(swizzle(tid), end=" ")
        if (tid + 1) % (2**swizzle.shift) == 0:
            print()


# CHECK-LABEL: test_swizzle_basic
fn test_swizzle_basic():
    print("== test_swizzle_basic")

    alias thread_layout = Layout.row_major(8, 8)

    # swizzle every 16 threads by the least significant bit.
    var swizzle_bits1_per16 = Swizzle(1, 0, 4)

    # CHECK: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # CHECK: 17 16 19 18 21 20 23 22 25 24 27 26 29 28 31 30
    # CHECK: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    # CHECK: 49 48 51 50 53 52 55 54 57 56 59 58 61 60 63 62
    print_swizzle(thread_layout, swizzle_bits1_per16)

    # swizzle every 8 threads by 2 least significant bits.
    var swizzle_bits2_per8 = Swizzle(2, 0, 3)

    # CHECK: 0 1 2 3 4 5 6 7
    # CHECK: 9 8 11 10 13 12 15 14
    # CHECK: 18 19 16 17 22 23 20 21
    # CHECK: 27 26 25 24 31 30 29 28
    # CHECK: 32 33 34 35 36 37 38 39
    # CHECK: 41 40 43 42 45 44 47 46
    # CHECK: 50 51 48 49 54 55 52 53
    # CHECK: 59 58 57 56 63 62 61 60
    print_swizzle(thread_layout, swizzle_bits2_per8)

    # swizzle every 16 threads the 2nd and 3rd least significant bits.
    var swizzle_bits2_base1_per8 = Swizzle(2, 1, 3)

    # CHECK: 0 1 2 3 4 5 6 7
    # CHECK: 8 9 10 11 12 13 14 15
    # CHECK: 18 19 16 17 22 23 20 21
    # CHECK: 26 27 24 25 30 31 28 29
    # CHECK: 36 37 38 39 32 33 34 35
    # CHECK: 44 45 46 47 40 41 42 43
    # CHECK: 54 55 52 53 50 51 48 49
    # CHECK: 62 63 60 61 58 59 56 57
    for tid in range(thread_layout.size()):
        # Verify the operator overloaded for different index types.
        var tid_u32 = UInt32(tid)
        print(swizzle_bits2_base1_per8(tid_u32), end=" ")
        if (tid + 1) % 8 == 0:
            print()


fn cat_layout(layout_a: Layout, layout_b: Layout) -> Layout:
    var shape = IntTuple()
    var stride = IntTuple()
    for i in range(len(layout_a.shape)):
        shape.append(layout_a.shape[i])
        stride.append(layout_a.stride[i])
    for i in range(len(layout_b.shape)):
        shape.append(layout_b.shape[i])
        stride.append(layout_b.stride[i])
    return Layout(shape, stride)


fn append_layout(layout_a: Layout, layout_b: Layout) -> Layout:
    var shape = IntTuple()
    var stride = IntTuple()
    shape.append(layout_a.shape)
    stride.append(layout_a.stride)
    for i in range(len(layout_b.shape)):
        shape.append(layout_b.shape[i])
        stride.append(layout_b.stride[i])
    return Layout(shape, stride)


fn vectorize_layout[layout: Layout, *tile_sizes: Int]() -> Layout:
    return zipped_divide(layout, MakeTileLayoutList[*tile_sizes]())


fn vectorize_distribute_layout[
    *element_tile_sizes: Int, data_layout: Layout, thread_layout: Layout
]() -> Layout:
    alias vlayout = vectorize_layout[data_layout, *element_tile_sizes]()
    alias dlayout = _compute_distribute_layout[vlayout[1], thread_layout]()
    return append_layout(vlayout[0], dlayout)


@register_passable
struct WaveFrontSummary(Copyable, Defaultable, Movable):
    var total_wavefronts: Int
    var expected_wavefronts: Int

    fn __init__(out self):
        self.total_wavefronts = 0
        self.expected_wavefronts = 0

    fn excess_wavefronts(self) -> Int:
        return self.total_wavefronts - self.expected_wavefronts

    fn __str__(self) -> String:
        return String(
            "WaveFrontSummary(total_wavefronts=",
            self.total_wavefronts,
            ", expected_wavefronts=",
            self.expected_wavefronts,
            ", excess_wavefronts=",
            self.excess_wavefronts(),
            ")",
        )


fn count_wavefronts[
    *element_tile_sizes: Int,
    thread_layout: Layout,
    data_layout: Layout,
    type_bytes: Int,
    swizzle: Swizzle = Swizzle(0, 0, 1),
    bytes_per_bank: Int = 4,
    num_banks: Int = 32,
    max_memop_bytes: Int = 16,
]() -> WaveFrontSummary:
    constrained[thread_layout.size() == num_banks]()
    alias layout = vectorize_distribute_layout[
        *element_tile_sizes,
        data_layout=data_layout,
        thread_layout=thread_layout,
    ]()
    # layout[0]: elements
    # layout[1]: threads
    # layout[2]: individual memory accesses
    #
    alias coalesced_element = coalesce(layout[0])
    constrained[Int(coalesced_element.stride) == 1]()
    constrained[coalesced_element.rank() == 1]()
    alias element_bytes = coalesced_element.size() * type_bytes
    alias bytes_per_op = element_bytes if element_bytes < max_memop_bytes else max_memop_bytes
    alias ops_per_element = element_bytes // bytes_per_op
    alias vars_per_bank = bytes_per_bank // type_bytes if bytes_per_bank > type_bytes else 1
    alias num_phases = bytes_per_op // bytes_per_bank
    alias bank_group_size = num_banks // num_phases
    var banks = StaticTuple[Int, num_banks]()
    # TODO: 6 bytes should be convertible to 4 + 2 byte ops, or 3x 2-byte ops.
    # Not a high priority, as these will likely result in poor performance anyway.
    constrained[
        element_bytes % bytes_per_op == 0,
        "vectorization should divide evenly by memop size used",
    ]()
    constrained[
        bytes_per_op % bytes_per_bank == 0,
        "for efficiency, we should at least write 4 bytes at a time.",
    ]()
    var wavefronts = WaveFrontSummary()
    wavefronts.expected_wavefronts = (
        layout[2].size() * ops_per_element * num_phases
    )
    wavefronts.total_wavefronts = 0
    alias thread_layout_perm = right_inverse(thread_layout)
    # print(layout)
    # print("num_phases =", num_phases)
    # print("bank_group_size =", bank_group_size)
    # iterate over elements
    for i in range(layout[2].size()):
        var elt_idx_base = layout[2](i)
        # memop per elements
        for j in range(ops_per_element):
            for p in range(num_phases):
                for k in range(banks.size):
                    banks[k] = 0
                for k in range(bank_group_size):
                    var tidx = layout[1](
                        thread_layout_perm(p * bank_group_size + k)
                    )
                    var idx = tidx + elt_idx_base
                    for l in range(num_phases):
                        var sidx = (
                            swizzle(idx + (l + j * num_phases) * vars_per_bank)
                            // vars_per_bank
                        )
                        # print(sidx, end=" ")
                        # print(sidx%32, end=" ")
                        banks[sidx % num_banks] += 1
                var max_wavefronts = 0
                for k in range(banks.size):
                    var w = banks[k]
                    max_wavefronts = (
                        max_wavefronts if max_wavefronts >= w else w
                    )
                wavefronts.total_wavefronts += max_wavefronts
                # print()
    return wavefronts


fn test_swizzle_gemm_store() raises:
    alias WM = 64
    alias WN = 64
    alias MMA_M = 16
    alias MMA_N = 8
    alias noswizzle = Swizzle(0, 0, 1)
    alias swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
    ]()
    var wfs_noswizzle_reg_to_smem_fp32 = count_wavefronts[
        1,
        2,
        thread_layout = Layout.row_major(8, 4),
        data_layout = Layout.row_major(WM, WN),
        type_bytes=4,
        swizzle=noswizzle,
    ]()
    assert_equal(wfs_noswizzle_reg_to_smem_fp32.excess_wavefronts(), 384)
    var wfs_noswizzle_smem_to_gmem_fp32 = count_wavefronts[
        1,
        4,
        thread_layout = Layout.row_major(32 * 8 // WN, WN // 8),
        data_layout = Layout.row_major(WM, WN),
        type_bytes=4,
        swizzle=noswizzle,
    ]()
    assert_equal(wfs_noswizzle_smem_to_gmem_fp32.excess_wavefronts(), 0)
    var wfs_swizzle_reg_to_smem_fp32 = count_wavefronts[
        1,
        2,
        thread_layout = Layout.row_major(8, 4),
        data_layout = Layout.row_major(WM, WN),
        type_bytes=4,
        swizzle=swizzle,
    ]()
    assert_equal(wfs_swizzle_reg_to_smem_fp32.excess_wavefronts(), 0)
    var wfs_swizzle_smem_to_gmem_fp32 = count_wavefronts[
        1,
        8,
        thread_layout = Layout.row_major(32 * 8 // WN, WN // 8),
        data_layout = Layout.row_major(WM, WN),
        type_bytes=4,
        swizzle=swizzle,
    ]()
    assert_equal(wfs_swizzle_smem_to_gmem_fp32.excess_wavefronts(), 128)
    var wfs_swizzle_reg_to_smem_bf16 = count_wavefronts[
        1,
        2,
        thread_layout = Layout.row_major(8, 4),
        data_layout = Layout.row_major(WM, WN),
        type_bytes=2,
        swizzle=swizzle,
    ]()
    assert_equal(wfs_swizzle_reg_to_smem_bf16.excess_wavefronts(), 0)
    var wfs_swizzle_smem_to_gmem_bf16 = count_wavefronts[
        1,
        8,
        thread_layout = Layout.row_major(32 * 8 // WN, WN // 8),
        data_layout = Layout.row_major(WM, WN),
        type_bytes=2,
        swizzle=swizzle,
    ]()
    assert_equal(wfs_swizzle_smem_to_gmem_bf16.excess_wavefronts(), 0)


fn main() raises:
    test_swizzle_basic()
    test_swizzle_gemm_store()
