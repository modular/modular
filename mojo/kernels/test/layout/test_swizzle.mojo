# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s | FileCheck %s

from collections import OptionalReg
from layout.element import Element
from layout.layout import Layout, make_layout
from layout.int_tuple import size
from layout.layout_tensor import LayoutTensor
from layout.swizzle import Swizzle, make_swizzle
from memory import UnsafePointer
from sys import sizeof, is_x86
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


fn thread_fragment[
    type: DType,
    WARP_SIZE: UInt,
    WM: UInt,
    WN: UInt,
    MMA_M: UInt,
    MMA_N: UInt,
](lane_idx: UInt) -> LayoutTensor[
    type,
    Layout.row_major((WM // MMA_M) * (WN // MMA_N), MMA_M * MMA_N // WARP_SIZE),
] as result:
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    alias frag_size = MMA_M * MMA_N // WARP_SIZE
    constrained[
        frag_size == 4, "support for non-4 frag sizes not yet implemented"
    ]()
    # mmas are ordered
    # n_mma * num_m_mmas + m_mma
    # and according to `c`/`d` here:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    # thread position within matrix
    var local_n = lane_idx % (MMA_N // 2)
    var local_m0 = lane_idx // (MMA_N // 2)
    # var local_m1 = local_m0 + MMA_M // 2
    # our c/d reg_tile has 4 elements, at positions
    var local_offset_0 = local_m0 * WN + 2 * local_n
    var local_offset_1 = local_offset_0 + 1
    var local_offset_2 = local_offset_0 + (MMA_M // 2) * WN
    var local_offset_3 = local_offset_2 + 1
    var local_offsets = SIMD[DType.float32, 4](
        local_offset_0, local_offset_1, local_offset_2, local_offset_3
    )
    # local_m0, 2*local_n
    # local_m0, 2*local_n+1
    # local_m1, 2*local_n
    # local_m1, 2*local_n+1
    var frag = LayoutTensor[
        type, Layout.row_major(num_m_mmas * num_n_mmas, frag_size)
    ].stack_allocation()
    # we fill such that our warp tile is equal to
    # arange(LayoutTensor[type,Layout.row_major(WM,WN)].stack_allocation())
    for m_mma in range(num_m_mmas):
        for n_mma in range(num_n_mmas):
            var offset = m_mma * MMA_M * WN + n_mma * MMA_N
            var idx = n_mma * num_m_mmas + m_mma
            frag.ptr.store(
                4 * idx,
                ((SIMD[DType.float32, 4](offset) + local_offsets) / 1).cast[
                    type
                ](),
            )

    return rebind[__type_of(result)](frag)


fn getindex[
    layout: Layout,
    element_layout: Layout,
    swizzle: OptionalReg[Swizzle] = None,
](i: Int, offset: Int,) -> Int:
    @parameter
    if swizzle:
        alias swizzle_fn = swizzle.value()
        var dst_idx = layout(i)
        var dst_idx_base = dst_idx % swizzle_fn.size()
        var dst_idx_diff = dst_idx - dst_idx_base
        return swizzle_fn(offset + dst_idx_base) + dst_idx_diff
    else:
        alias element_size = int(element_layout.size())
        alias cl = make_layout(element_layout, layout)
        return cl(i * element_size) + offset


fn copy_local_to_sram[
    type: DType, swizzle: OptionalReg[Swizzle] = None
](
    smem: LayoutTensor,
    src_frag: LayoutTensor,
    lane_idx: UInt,
    smem_history: LayoutTensor,
):
    var smem_frag = smem.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
        lane_idx
    )
    var smem_frag_offset = smem_frag.distance(smem.ptr)

    # copy frag to smem
    @parameter
    for i in range(__type_of(smem_frag).layout.size()):
        var src_idx = getindex[src_frag.layout, src_frag.element_layout](i, 0)
        var smem_idx = getindex[
            __type_of(smem_frag).layout,
            __type_of(smem_frag).element_layout,
            swizzle,
        ](i, smem_frag_offset)

        var x = Element[type, src_frag.element_layout].load(
            rebind[UnsafePointer[Scalar[type], src_frag.address_space]](
                src_frag.ptr.offset(src_idx)
            )
        )
        Element[type, __type_of(smem_frag).element_layout](
            rebind[
                Element[
                    type, __type_of(smem_frag).element_layout
                ].element_data_type
            ](x.element_data)
        ).store(
            rebind[
                UnsafePointer[Scalar[type], __type_of(smem_frag).address_space]
            ](smem.ptr).offset(smem_idx)
        )
        smem_history[i, lane_idx] = smem_idx
        # if lane_idx == 30 and swizzle:
        #     alias didx = smem_frag.layout(i)
        #     alias didxb = didx % swizzle.value().size()
        #     print("dfo:", smem_frag_offset, "didx:",didx, "didxb:",didxb, "sidx:",(smem_idx//2)%16, sep="\t")


fn copy_sram_to_dram[
    type: DType, WARP_SIZE: UInt, WN: UInt, swizzle: OptionalReg[Swizzle] = None
](
    gmem: LayoutTensor,
    smem: LayoutTensor,
    lane_idx: UInt,
    smem_history: LayoutTensor,
):
    # 16-byte SIMD
    alias simd_size = 16 // sizeof[type]()
    alias thread_layout = Layout.row_major(
        WARP_SIZE * simd_size // WN, WN // simd_size
    )

    var smem_frag = smem.vectorize[1, simd_size]().distribute[thread_layout](
        lane_idx
    )
    var gmem_frag = gmem.vectorize[1, simd_size]().distribute[thread_layout](
        lane_idx
    )
    # gmem_frag.copy_from(smem_frag)

    var smem_frag_offset = smem_frag.distance(smem.ptr)
    var gmem_frag_offset = gmem_frag.distance(gmem.ptr)

    constrained[
        size(smem_history.layout.shape[0]) == __type_of(smem_frag).layout.size()
    ]()

    # copy frag to smem
    @parameter
    for i in range(__type_of(smem_frag).layout.size()):
        var smem_idx = getindex[
            __type_of(smem_frag).layout,
            __type_of(smem_frag).element_layout,
            swizzle,
        ](i, smem_frag_offset)
        var gmem_idx = getindex[
            __type_of(gmem_frag).layout, __type_of(gmem_frag).element_layout
        ](i, gmem_frag_offset)

        var x = Element[type, __type_of(smem_frag).element_layout].load(
            rebind[
                UnsafePointer[Scalar[type], __type_of(smem_frag).address_space]
            ](smem.ptr.offset(smem_idx))
        )
        Element[type, __type_of(gmem_frag).element_layout](
            rebind[
                Element[
                    type, __type_of(gmem_frag).element_layout
                ].element_data_type
            ](x.element_data)
        ).store(
            rebind[
                UnsafePointer[Scalar[type], __type_of(gmem_frag).address_space]
            ](gmem.ptr).offset(gmem_idx)
        )
        smem_history[i, lane_idx] = smem_idx


fn count_excess_wavefronts[
    vector_len: UInt, elbytes: UInt, WARP_SIZE: UInt
](smem_accesses: LayoutTensor) -> UInt:
    alias bank_set_size = vector_len * elbytes // 4
    alias num_bank_sets = WARP_SIZE // bank_set_size
    alias num_vectors_written = size(smem_accesses.layout.shape[0])

    var excess_wavefronts = 0
    var bank_sets = LayoutTensor[
        DType.uint32, Layout.row_major(num_bank_sets)
    ].stack_allocation()
    for n in range(num_vectors_written):
        for b in range(bank_set_size):
            for l in range(num_bank_sets):
                bank_sets[l] = 0
            # NOTE: we assume here that all `widx` are different,
            # Two smem requests to the same 32 bit word don't cause bank conflicts.
            # We'd need to update this code if we intend to model that.
            for l in range(num_bank_sets):
                var i = l + b * num_bank_sets
                var widx = int(smem_accesses[n, i])
                bank_sets[(widx // vector_len) % num_bank_sets] += 1
            var max_use = 0
            for l in range(num_bank_sets):
                if bank_sets[l] > max_use:
                    max_use = int(bank_sets[l])
            excess_wavefronts += max_use - 1
    return excess_wavefronts


fn calc_excess_wavefronts[
    type: DType,
    MMA_M: Int,
    MMA_N: Int,
    WM: Int,
    WN: Int,
    swizzle_w: Swizzle,
    swizzle_r: Swizzle,
]() raises -> StaticTuple[Int, 2]:
    alias WARP_SIZE = 32
    # alias WN = 64
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    alias frag_size = MMA_M * MMA_N // WARP_SIZE

    alias thread_layout = Layout.row_major(8, 4)
    # each thread copies into dst
    alias dst_layout = Layout.row_major(WM, WN)
    # we need to set the alignment as some code assumes vectors are aligned
    var smem = LayoutTensor[type, dst_layout].stack_allocation[alignment=16]()
    var gmem = LayoutTensor[type, dst_layout].stack_allocation[alignment=16]()

    alias simd_smem_write = 2
    alias num_vectors_written = num_m_mmas * num_n_mmas * frag_size // simd_smem_write
    # avoid depending on gpu
    # alias simd_smem_read = simdwidthof[type, target = _get_nvptx_target()]()
    alias simd_smem_read = 16 // sizeof[type]()
    alias num_vectors_read = num_m_mmas * num_n_mmas * frag_size // simd_smem_read

    var smem_writes = LayoutTensor[
        DType.uint32, Layout.row_major(num_vectors_written, WARP_SIZE)
    ].stack_allocation()
    var smem_reads = LayoutTensor[
        DType.uint32, Layout.row_major(num_vectors_read, WARP_SIZE)
    ].stack_allocation()

    for i in range(dst_layout.size()):
        smem.ptr.store(i, -1)
        gmem.ptr.store(i, -1)
    for lid in range(WARP_SIZE):
        var src_frag = thread_fragment[type, WARP_SIZE, WM, WN, MMA_M, MMA_N](
            lid
        ).vectorize[1, 2]().transpose()
        copy_local_to_sram[type, swizzle_w](smem, src_frag, lid, smem_writes)

    for lid in range(WARP_SIZE):
        copy_sram_to_dram[type, WARP_SIZE, WN, swizzle_r](
            gmem, smem, lid, smem_reads
        )

    constrained[sizeof[type]() >= 2, "doesn't yet support < 2 byte types."]()
    alias bank_set_size_write = sizeof[type]() * simd_smem_write // 4
    alias bank_set_size_read = sizeof[type]() * simd_smem_read // 4
    alias num_bank_sets_write = WARP_SIZE // bank_set_size_write
    alias num_bank_sets_read = WARP_SIZE // bank_set_size_read
    # print("smem writes:")
    # for n in range(num_vectors_written):
    #     for l in range(WARP_SIZE):
    #         print(
    #             (smem_writes[n, l] // simd_smem_write) % num_bank_sets_write,
    #             end=" ",
    #         )
    #     print()

    var excess_write_wavefronts = count_excess_wavefronts[
        simd_smem_write, sizeof[type](), WARP_SIZE
    ](smem_writes)
    var excess_read_wavefronts = count_excess_wavefronts[
        simd_smem_read, sizeof[type](), WARP_SIZE
    ](smem_reads)

    for m in range(WM):
        for n in range(WN):
            assert_equal(gmem[m, n], ((m * WM + n) / 1).cast[type]())

    # We have 32 banks, and vectorize by 2.
    # If the element size is 32 bits, we execute in two phases.
    # If the element size is 16 bits, we execute in one phase.
    # for n in range(num_vectors_read):
    #     for l in range(WARP_SIZE):
    #         print(
    #             (smem_reads[n, l] // simd_smem_read) % num_bank_sets_read,
    #             end=" ",
    #         )
    #     print()
    return StaticTuple[Int, 2](excess_write_wavefronts, excess_read_wavefronts)


fn test_swizzle_gemm() raises:
    alias MMA_M = 16
    alias MMA_N = 8
    alias WM = 64
    alias WN = 64
    alias noswizzle = Swizzle(0, 0, 1)
    # print("fp32 no swizzle")
    var conflicts_noswizzle_fp32 = calc_excess_wavefronts[
        DType.float32, MMA_M, MMA_N, WM, WN, noswizzle, noswizzle
    ]()
    assert_equal(conflicts_noswizzle_fp32[0], 384)
    assert_equal(conflicts_noswizzle_fp32[1], 0)

    @parameter
    if is_x86():
        # print("bf16 no swizzle")
        var conflicts_noswizzle_bf16 = calc_excess_wavefronts[
            DType.bfloat16, MMA_M, MMA_N, WM, WN, noswizzle, noswizzle
        ]()
        assert_equal(conflicts_noswizzle_bf16[0], 448)
        assert_equal(conflicts_noswizzle_bf16[1], 0)

    alias matmul_swizzle = make_swizzle[
        num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
    ]()
    # print("fp32 swizzle")
    var conflicts_matmul_swizzle_fp32 = calc_excess_wavefronts[
        DType.float32, MMA_M, MMA_N, WM, WN, matmul_swizzle, matmul_swizzle
    ]()
    assert_equal(conflicts_matmul_swizzle_fp32[0], 0)
    assert_equal(conflicts_matmul_swizzle_fp32[1], 0)

    @parameter
    if is_x86():
        # print("bf16 swizzle")
        var conflicts_matmul_swizzle_bf16 = calc_excess_wavefronts[
            DType.bfloat16, MMA_M, MMA_N, WM, WN, matmul_swizzle, matmul_swizzle
        ]()
        assert_equal(conflicts_matmul_swizzle_bf16[0], 0)
        assert_equal(conflicts_matmul_swizzle_bf16[1], 0)


fn main() raises:
    test_swizzle_basic()
    test_swizzle_gemm()
