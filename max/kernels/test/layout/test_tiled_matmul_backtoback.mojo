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

from math import fma, isclose
from os import abort
from random import rand
from sys import CompilationTarget, argv, simdwidthof, sizeof

import benchmark
from algorithm.functional import vectorize
from layout import Layout, RuntimeLayout, RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE, IntTuple, size
from layout.layout import coalesce, expand_modes_alike, flatten
from layout.layout_tensor import LayoutTensor
from layout.math import outer_product_acc
from memory import memcpy, memset_zero, stack_allocation
from testing import assert_false

from utils import StaticTuple


fn matmul_naive[
    layoutC: Layout, layoutA: Layout, layoutB: Layout, elt: DType
](
    C: LayoutTensor[elt, layoutC, MutableAnyOrigin],
    A: LayoutTensor[elt, layoutA, MutableAnyOrigin],
    B: LayoutTensor[elt, layoutB, MutableAnyOrigin],
):
    constrained[len(layoutC) == 2]()
    constrained[len(layoutA) == 2]()
    constrained[len(layoutB) == 2]()
    alias M: Int = size(layoutC.shape[0])
    alias N: Int = size(layoutC.shape[1])
    alias K: Int = size(layoutA.shape[1])
    constrained[M == size(layoutA.shape[0])]()
    constrained[N == size(layoutB.shape[1])]()
    constrained[K == size(layoutB.shape[0])]()
    for m in range(M):
        for n in range(N):
            C[m, n] = Scalar[elt]()
    for m in range(M):
        for k in range(K):
            for n in range(N):
                C[m, n] += A[m, k] * B[k, n]


alias cacheline_size: Int = 64


# We should be able to support 1-access per cacheline
# even when nr * width < cacheline_size // sizeof[elt]()
# For Apple Silicon, the values for `float32` would be
# 4 * 4 < 128 // 4
# Firestorm core's l1 caches are so large, that we wouldn't
# really need this, though.
# Also: cacheline_size of 64 is currently hard coded.
@always_inline
fn stride[elt: DType](nrw: Int) -> Int:
    if nrw * sizeof[elt]() >= cacheline_size:
        return cacheline_size // sizeof[elt]()
    else:
        return nrw


@always_inline
fn getKr[mode: IntTuple]() -> Int:
    if mode.is_value() or len(mode) == 1:
        return 1
    else:
        return mode[0].value()


# Assumes that we have packed `A` and `B`, `C` also uses a packed layout.
# @always_inline
fn matmul_ukern[
    elt: DType, width: Int, mr: Int, nr: Int, kr: Int, kf: Int
](
    C: UnsafePointer[Scalar[elt], alignment=64],
    A: UnsafePointer[Scalar[elt]],
    B: UnsafePointer[Scalar[elt], alignment=64],
    inc: Bool,
):
    alias Align: Int = sizeof[elt]() * width
    alias Astride: Int = stride[elt](nr * width)
    alias CstoreReps: Int = nr * width // Astride
    constrained[CstoreReps * Astride == nr * width]()
    alias CstoresPer: Int = Astride // width
    constrained[CstoresPer * width == Astride]()
    constrained[CstoresPer * CstoreReps == nr]()
    # for n0 in range(CstoreReps):
    #   for n1 in range(CstoresPer):

    var acc: StaticTuple[SIMD[elt, width], mr * nr] = StaticTuple[
        SIMD[elt, width], mr * nr
    ]()

    @parameter
    for m in range(mr):

        @parameter
        for n in range(nr):
            acc[n + m * nr] = SIMD[elt, width]()
    var Bloads: StaticTuple[SIMD[elt, width], nr] = StaticTuple[
        SIMD[elt, width], nr
    ]()
    # We're assuming 64-byte cachelines
    # This is a trick to repeatedly re-touch A's memory in the microkernel,
    # so that A can stay in the L1-cache, while we stream B through it.

    var Ao: UnsafePointer[Scalar[elt]] = A
    var Bo: UnsafePointer[Scalar[elt], alignment=64] = B
    # TODO: static assert that kf%Astride == 0
    for _ in range(Astride):
        # Aecause we repeatedly call `matmul_ukern` with the same
        # slice of `A`, but different slice of `B`, we wish for `A`
        # to remain in the l1 cache, but freely evict `B`.
        # Repeatedly re-touching the cachelines of `A` helps us achieve this.
        var Atmp: UnsafePointer[Scalar[elt]] = Ao
        Ao = Ao.offset(1)
        for _ in range(kf):

            @parameter
            for _ in range(kr):

                @parameter
                for n in range(nr):
                    Bloads[n] = Bo.load[width=width, alignment=Align](n * width)

                @parameter
                for m in range(mr):
                    var Abroadcast: SIMD[elt, width] = SIMD[elt, width](
                        Atmp.load[width=1, alignment = sizeof[elt]()](
                            m * Astride
                        )
                    )

                    @parameter
                    for n in range(nr):
                        # breakpoint()
                        acc[n + m * nr] = fma(
                            Abroadcast, Bloads[n], acc[n + m * nr]
                        )
                Atmp = Atmp.offset(mr * Astride)
                Bo = Bo.offset(nr * width)
    if inc:
        # Note, `C` would have spilled from the L1 cache by the time
        # we load it again had we loaded before the reduction loop.
        @parameter
        for m in range(mr):

            @parameter
            for n0 in range(CstoreReps):

                @parameter
                for n1 in range(CstoresPer):
                    acc[n1 + n0 * CstoresPer + m * nr] = acc[
                        n1 + n0 * CstoresPer + m * nr
                    ] + C.load[width=width, alignment=Align](
                        (n1 + m * CstoresPer + n0 * (mr * CstoresPer)) * width
                    )

    @parameter
    for m in range(mr):

        @parameter
        for n0 in range(CstoreReps):

            @parameter
            for n1 in range(CstoresPer):
                C.store[alignment=Align](
                    (n1 + m * CstoresPer + n0 * (mr * CstoresPer)) * width,
                    acc[n1 + n0 * CstoresPer + m * nr],
                )


# C = A * B
# C and A have the same layout, to facilitate the next step of
# D = (A * B) * C
# i.e., make it easier to chain matrix multiplies.
# A's shape is (W*Mr, Mc/(W*Mr), M/Mc), (Kc, K/Kc)
# A's strides are (1, W*Mr*Kc, Mc*K), (W*Mr, Mc*Kc)
# B's shape is (Kc*sizeof(elt)/64, 64/sizeof(elt), K/Kc), (nr,Nc/nr,N/Nc)
# B's strides are ((nr*64/sizeof(elt), 1, Nc*Kc), (64/sizeof(elt), nr*Kc, Nc*K)
#
fn matmul[
    elt: DType,
    M: Int,
    N: Int,
    K: Int,
    W: Int,
    Mc: Int,
    Nc: Int,
    Kc: Int,
    Mr: Int,
    Nr: Int,
    Kr: Int,
    layoutC: Layout,
    layoutA: Layout,
    layoutB: Layout,
](
    C: LayoutTensor[elt, layoutC, MutableAnyOrigin],
    A: LayoutTensor[elt, layoutA, MutableAnyOrigin],
    B: LayoutTensor[elt, layoutB, MutableAnyOrigin],
):
    alias WNr = W * Nr
    alias Stride = stride[elt](WNr)

    constrained[len(layoutC) == 2]()
    constrained[len(layoutA) == 2]()
    constrained[len(layoutB) == 2]()
    # I am assuming that the `shape` and `stride` are congruent (i.e., equal length)
    # so that I don't need to check both here.
    constrained[len(layoutC.shape[0]) == 3]()
    constrained[len(layoutC.shape[1]) == 4]()
    constrained[len(layoutA.shape[0]) == 3]()
    constrained[len(layoutA.shape[1]) == 4]()
    constrained[len(layoutB.shape[0]) == 3]()
    constrained[len(layoutB.shape[1]) == 3]()

    # Matrix C
    constrained[size(layoutC.shape[0].tuple()[0]) == Mr]()
    constrained[size(layoutC.shape[0].tuple()[1]) * Mr == Mc]()
    constrained[size(layoutC.shape[0].tuple()[2]) * Mc == M]()

    constrained[size(layoutC.shape[1].tuple()[0]) == Stride]()
    constrained[size(layoutC.shape[1].tuple()[1]) * Stride == WNr]()
    constrained[size(layoutC.shape[1].tuple()[2]) * WNr == Nc]()
    constrained[size(layoutC.shape[1].tuple()[3]) * Nc == N]()

    constrained[size(layoutC.stride[0].tuple()[0]) == Stride]()
    constrained[size(layoutC.stride[0].tuple()[1]) == Mr * Nc]()
    constrained[size(layoutC.stride[0].tuple()[2]) == Mc * N]()

    constrained[size(layoutC.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutC.stride[1].tuple()[1]) == Mr * Stride]()
    constrained[size(layoutC.stride[1].tuple()[2]) == Mr * WNr]()
    constrained[size(layoutC.stride[1].tuple()[3]) == Mc * Nc]()

    # Matrix A
    constrained[size(layoutA.shape[0].tuple()[0]) == Mr]()
    constrained[size(layoutA.shape[0].tuple()[1]) * Mr == Mc]()
    constrained[size(layoutA.shape[0].tuple()[2]) * Mc == M]()

    constrained[size(layoutA.shape[1].tuple()[0]) == Stride]()
    constrained[size(layoutA.shape[1].tuple()[1]) * Stride == WNr]()
    constrained[size(layoutA.shape[1].tuple()[2]) * WNr == Kc]()
    constrained[size(layoutA.shape[1].tuple()[3]) * Kc == K]()

    constrained[size(layoutA.stride[0].tuple()[0]) == Stride]()
    constrained[size(layoutA.stride[0].tuple()[1]) == Mr * Kc]()
    constrained[size(layoutA.stride[0].tuple()[2]) == Mc * K]()

    constrained[size(layoutA.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutA.stride[1].tuple()[1]) == Mr * Stride]()
    constrained[size(layoutA.stride[1].tuple()[2]) == Mr * WNr]()
    constrained[size(layoutA.stride[1].tuple()[3]) == Mc * Kc]()

    # Matrix B
    constrained[size(layoutB.shape[0].tuple()[0]) == Stride]()
    constrained[size(layoutB.shape[0].tuple()[1]) * Stride == Kc]()
    constrained[size(layoutB.shape[0].tuple()[2]) * Kc == K]()

    constrained[size(layoutB.shape[1].tuple()[0]) == WNr]()
    constrained[size(layoutB.shape[1].tuple()[1]) * WNr == Nc]()
    constrained[size(layoutB.shape[1].tuple()[2]) * Nc == N]()

    constrained[size(layoutB.stride[0].tuple()[0]) * Stride == WNr * Kc]()
    constrained[size(layoutB.stride[0].tuple()[1]) == WNr]()
    constrained[size(layoutB.stride[0].tuple()[2]) == Nc * Kc]()

    constrained[size(layoutB.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutB.stride[1].tuple()[1]) == WNr * Kc]()
    constrained[size(layoutB.stride[1].tuple()[2]) == Nc * K]()

    alias Ptr = UnsafePointer[Scalar[elt], alignment=64]
    var pc: UnsafePointer[Scalar[elt]] = C.ptr
    var pa: UnsafePointer[Scalar[elt]] = A.ptr
    # TODO: nontemporal prefetches on the microkernel slices of `B`
    #       as the slice does not get reused at the L2 or L3 level.
    # TODO: prefetches on `A`, to hide latency, as we stream it through
    #       the L1, suffering L2->register latency for each load.
    # NOTE: Read comments within the loop from the inside out.
    for _ in range(M // Mc):
        var pb: UnsafePointer[Scalar[elt], alignment=64] = B.ptr
        var pak: __type_of(pb) = pa
        for _ in range(N // Nc):
            var pck: UnsafePointer[Scalar[elt], alignment=64] = pc
            pak = pa
            for kc in range(K // Kc):
                var pbk: UnsafePointer[Scalar[elt], alignment=64] = pb
                pck = pc
                for _ in range(Mc // Mr):  # mr
                    pbk = pb
                    for _ in range(Nc // WNr):  # nr
                        matmul_ukern[elt, W, Mr, Nr, Kr, Kc // (Stride * Kr)](
                            pck, pak, pbk, kc != 0
                        )
                        pbk = pbk.offset(WNr * Kc)
                        pck = pck.offset(Mr * WNr)
                    pak = pak.offset(Mr * Kc)
                pb = pbk
            pc = pck
        pa = pak


fn alloc_tensor[
    elt: DType, layout: Layout
]() -> LayoutTensor[elt, layout, MutableAnyOrigin]:
    return LayoutTensor[elt, layout, MutableAnyOrigin](
        UnsafePointer[Scalar[elt], alignment=64].alloc(layout.size())
    )


fn alloc_tensor[
    elt: DType, layout: Layout
](rtlayout: RuntimeLayout[layout, **_]) -> LayoutTensor[
    elt, layout, MutableAnyOrigin
]:
    return LayoutTensor[elt, layout, MutableAnyOrigin](
        UnsafePointer[Scalar[elt], alignment=64].alloc(rtlayout.size()),
        rtlayout,
    )


fn max_min_idx_positive(x: List[Int], y: List[Int]) -> Int:
    # this could be implemented more generically, e.g.
    # mapreduce-style?
    # Use `Buffer` for SIMD?
    if len(x) != len(y):
        abort()
    var argmax: Int = -1
    var max: Int = -1
    for i in range(len(x)):
        var m = min(x[i], y[i])
        if m > max:
            argmax = i
            max = m
    return argmax


fn delete_idx(arg: List[Int], idx: Int) -> List[Int]:
    var res = List[Int]()
    res.reserve(len(arg) - 1)
    for i in range(len(arg)):
        if i != idx:
            res.append(arg[i])
    return res


@always_inline
fn strided_load[
    elt: DType, //, W: Int, X: Int
](p: UnsafePointer[Scalar[elt]], i: Int) -> SIMD[elt, W]:
    @parameter
    if X == 1:
        return p.load[width=W](i)
    else:
        return (p + i * X).strided_load[width=W](X)


@always_inline
fn strided_store[
    elt: DType, W: Int, //, X: Int
](p: UnsafePointer[Scalar[elt]], i: Int, x: SIMD[elt, W]):
    @parameter
    if X == 1:
        p.store(i, x)
    else:
        (p + i * X).strided_store(x, X)


@always_inline
fn vectorize_flat[
    elt_a: DType,
    elt_b: DType, //,
    f: fn[width: Int, stride_a: Int, stride_b: Int] (
        UnsafePointer[Scalar[elt_a]], UnsafePointer[Scalar[elt_b]], Int
    ) capturing -> None,
    simd_width: Int,
    unroll_factor: Int,
    shape: List[Int],
    stride_a: List[Int],
    stride_b: List[Int],
](a: UnsafePointer[Scalar[elt_a]], b: UnsafePointer[Scalar[elt_b]]):
    constrained[len(shape) == len(stride_a)]()
    constrained[len(shape) == len(stride_b)]()

    @parameter
    if len(shape) == 1:
        # perform the copy
        alias int_stride_a: Int = stride_a[0]
        alias int_stride_b: Int = stride_b[0]
        alias size = shape[0]

        @always_inline
        @parameter
        fn vf[width: Int](i: Int):
            f[width, int_stride_a, int_stride_b](a, b, i)

        vectorize[
            vf,
            simd_width,
            unroll_factor = min(size // simd_width, unroll_factor),
        ](size)
    else:
        # we find the maximum min stride, subset, and loop over it.
        alias max_idx = max_min_idx_positive(stride_b, stride_a)
        alias subset_shape = delete_idx(shape, max_idx)
        alias subset_stride_b = delete_idx(stride_b, max_idx)
        alias subset_stride_a = delete_idx(stride_a, max_idx)
        for i in range(shape[max_idx]):
            vectorize_flat[
                f,
                simd_width,
                unroll_factor,
                subset_shape,
                subset_stride_a,
                subset_stride_b,
            ](a + i * stride_a[max_idx], b + i * stride_b[max_idx])


fn tolist(x: IntTuple) -> List[Int]:
    var list = List[Int]()
    var flat = flatten(x)
    for y in flat:
        list.append(y.value())
    return list


fn vectorize_layout_tensor[
    elt_a: DType,
    layout_a: Layout,
    elt_b: DType,
    layout_b: Layout, //,
    f: fn[width: Int, stride_a: Int, stride_b: Int] (
        UnsafePointer[Scalar[elt_a]], UnsafePointer[Scalar[elt_b]], Int
    ) capturing -> None,
    simd_width: Int = max(simdwidthof[elt_a](), simdwidthof[elt_b]()),
    unroll_factor: Int = 4,
](
    a: LayoutTensor[elt_a, layout_a, MutableAnyOrigin],
    b: LayoutTensor[elt_b, layout_b, MutableAnyOrigin],
):
    alias expanded = expand_modes_alike(
        layout_a.shape, layout_a.stride, layout_b.shape, layout_b.stride
    )
    alias shape = tolist(expanded[0])
    alias stride_a = tolist(expanded[1])
    alias stride_b = tolist(expanded[2])
    vectorize_flat[f, simd_width, unroll_factor, shape, stride_a, stride_b](
        a.ptr, b.ptr
    )


fn copy_to[
    elt_dst: DType,
    layout_dst: Layout,
    elt_src: DType,
    layout_src: Layout, //,
    simd_width: Int = max(simdwidthof[elt_dst](), simdwidthof[elt_src]()),
    unroll_factor: Int = 4,
](
    dst: LayoutTensor[elt_dst, layout_dst, MutableAnyOrigin],
    src: LayoutTensor[elt_src, layout_src, MutableAnyOrigin],
):
    @always_inline
    @parameter
    fn copy[
        width: Int, stride_a: Int, stride_b: Int
    ](
        dstp: UnsafePointer[Scalar[elt_dst]],
        srcp: UnsafePointer[Scalar[elt_src]],
        i: Int,
    ):
        var vsrc = strided_load[width, stride_b](srcp, i)
        strided_store[stride_a](dstp, i, vsrc.cast[elt_dst]())

    vectorize_layout_tensor[copy, simd_width, unroll_factor](dst, src)


fn check_approx_equal[
    elt_dst: DType,
    layout_dst: Layout,
    elt_src: DType,
    layout_src: Layout, //,
    cmp_elt: DType,
    simd_width: Int = max(simdwidthof[elt_dst](), simdwidthof[elt_src]()),
    *,
    unroll_factor: Int = 4,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
](
    dst: LayoutTensor[elt_dst, layout_dst, MutableAnyOrigin],
    src: LayoutTensor[elt_src, layout_src, MutableAnyOrigin],
) raises:
    var fail: Bool = False

    @always_inline
    @parameter
    fn check[
        width: Int, stride_a: Int, stride_b: Int
    ](
        pa: UnsafePointer[Scalar[elt_dst]],
        pb: UnsafePointer[Scalar[elt_src]],
        i: Int,
    ):
        var va = strided_load[width, stride_a](pa, i).cast[cmp_elt]()
        var vb = strided_load[width, stride_b](pb, i).cast[cmp_elt]()
        if not all(isclose(va, vb, atol=atol, rtol=rtol, equal_nan=equal_nan)):
            fail = True

    vectorize_layout_tensor[check, simd_width, unroll_factor](dst, src)
    assert_false(fail)


# Kc == Nc, so don't need to specify both
fn matmulb2b[
    elt: DType,
    M: Int,
    N: Int,
    K: Int,
    L: Int,
    W: Int,
    Mc: Int,
    Nc: Int,
    Mr: Int,
    Nr: Int,
    Kr: Int,
    layoutD: Layout,
    layoutA: Layout,
    layoutB: Layout,
    layoutC: Layout,
](
    D: LayoutTensor[elt, layoutD, MutableAnyOrigin],
    A: LayoutTensor[elt, layoutA, MutableAnyOrigin],
    B: LayoutTensor[elt, layoutB, MutableAnyOrigin],
    C: LayoutTensor[elt, layoutC, MutableAnyOrigin],
):
    alias WNr = W * Nr
    alias Stride = stride[elt](WNr)
    alias Kc = Nc

    constrained[len(layoutD) == 2]()
    constrained[len(layoutA) == 2]()
    constrained[len(layoutB) == 2]()
    constrained[len(layoutC) == 2]()

    constrained[len(layoutD.shape[0]) == 3]()
    constrained[len(layoutD.shape[1]) == 4]()
    constrained[len(layoutA.shape[0]) == 3]()
    constrained[len(layoutA.shape[1]) == 4]()
    constrained[len(layoutB.shape[0]) == 3]()
    constrained[len(layoutB.shape[1]) == 3]()
    constrained[len(layoutC.shape[0]) == 3]()
    constrained[len(layoutC.shape[1]) == 3]()

    # Matrix D
    constrained[size(layoutD.shape[0].tuple()[0]) == Mr]()
    constrained[size(layoutD.shape[0].tuple()[1]) * Mr == Mc]()
    constrained[size(layoutD.shape[0].tuple()[2]) * Mc == M]()

    constrained[size(layoutD.shape[1].tuple()[0]) == Stride]()
    constrained[size(layoutD.shape[1].tuple()[1]) * Stride == WNr]()
    constrained[size(layoutD.shape[1].tuple()[2]) * WNr == Nc]()
    constrained[size(layoutD.shape[1].tuple()[3]) * Nc == N]()

    constrained[size(layoutD.stride[0].tuple()[0]) == Stride]()
    constrained[size(layoutD.stride[0].tuple()[1]) == Mr * Nc]()
    constrained[size(layoutD.stride[0].tuple()[2]) == Mc * N]()

    constrained[size(layoutD.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutD.stride[1].tuple()[1]) == Mr * Stride]()
    constrained[size(layoutD.stride[1].tuple()[2]) == Mr * WNr]()
    constrained[size(layoutD.stride[1].tuple()[3]) == Mc * Nc]()

    # Matrix A
    constrained[size(layoutA.shape[0].tuple()[0]) == Mr]()
    constrained[size(layoutA.shape[0].tuple()[1]) * Mr == Mc]()
    constrained[size(layoutA.shape[0].tuple()[2]) * Mc == M]()

    constrained[size(layoutA.shape[1].tuple()[0]) == Stride]()
    constrained[size(layoutA.shape[1].tuple()[1]) * Stride == WNr]()
    constrained[size(layoutA.shape[1].tuple()[2]) * WNr == Kc]()
    constrained[size(layoutA.shape[1].tuple()[3]) * Kc == K]()

    constrained[size(layoutA.stride[0].tuple()[0]) == Stride]()
    constrained[size(layoutA.stride[0].tuple()[1]) == Mr * Kc]()
    constrained[size(layoutA.stride[0].tuple()[2]) == Mc * K]()

    constrained[size(layoutA.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutA.stride[1].tuple()[1]) == Mr * Stride]()
    constrained[size(layoutA.stride[1].tuple()[2]) == Mr * WNr]()
    constrained[size(layoutA.stride[1].tuple()[3]) == Mc * Kc]()

    # Matrix B
    constrained[size(layoutB.shape[0].tuple()[0]) == Stride]()
    constrained[size(layoutB.shape[0].tuple()[1]) * Stride == Kc]()
    constrained[size(layoutB.shape[0].tuple()[2]) * Kc == K]()

    constrained[size(layoutB.shape[1].tuple()[0]) == WNr]()
    constrained[size(layoutB.shape[1].tuple()[1]) * WNr == Nc]()
    constrained[size(layoutB.shape[1].tuple()[2]) * Nc == L]()

    constrained[size(layoutB.stride[0].tuple()[0]) * Stride == WNr * Nc]()
    constrained[size(layoutB.stride[0].tuple()[1]) == WNr]()
    constrained[size(layoutB.stride[0].tuple()[2]) == Nc * Kc]()

    constrained[size(layoutB.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutB.stride[1].tuple()[1]) == WNr * Kc]()
    constrained[size(layoutB.stride[1].tuple()[2]) == Nc * K]()

    # Matrix C
    constrained[size(layoutC.shape[0].tuple()[0]) == Stride]()
    constrained[size(layoutC.shape[0].tuple()[1]) * Stride == Kc]()
    constrained[size(layoutC.shape[0].tuple()[2]) * Kc == L]()

    constrained[size(layoutC.shape[1].tuple()[0]) == WNr]()
    constrained[size(layoutC.shape[1].tuple()[1]) * WNr == Nc]()
    constrained[size(layoutC.shape[1].tuple()[2]) * Nc == N]()

    constrained[size(layoutC.stride[0].tuple()[0]) * Stride == WNr * Kc]()
    constrained[size(layoutC.stride[0].tuple()[1]) == WNr]()
    constrained[size(layoutC.stride[0].tuple()[2]) == N * Kc]()

    constrained[size(layoutC.stride[1].tuple()[0]) == 1]()
    constrained[size(layoutC.stride[1].tuple()[1]) == WNr * Kc]()
    constrained[size(layoutC.stride[1].tuple()[2]) == Nc * Kc]()

    var pa: UnsafePointer[Scalar[elt]] = A.ptr
    var pd: UnsafePointer[Scalar[elt]] = D.ptr
    # Should we support heap-allocating and passing it in?
    var AB: UnsafePointer[Scalar[elt], alignment=64] = stack_allocation[
        Mc * Nc, elt, alignment=64
    ]()
    # TODO: prefetches, as descried in nest
    # NOTE: Read comments within the loop from the inside out.
    #       I.e., read following a post-order depth first traversal of the
    #       loop tree.
    for _ in range(M // Mc):  # mc
        var pb: UnsafePointer[Scalar[elt], alignment=64] = B.ptr
        var pc: UnsafePointer[Scalar[elt], alignment=64] = C.ptr
        var pak: UnsafePointer[Scalar[elt], alignment=64] = pa
        var pdk: UnsafePointer[Scalar[elt], alignment=64] = pd
        for lc in range(L // Nc):  # lc, reduction for (AB)*C
            pak = pa
            for kc in range(
                K // Kc
            ):  # kc, reduction for A*B - hold in l3 cache
                # Comment #2
                # Size of slices accessed per iteration:
                # AB[Mc, Nc]  - held
                # A[Mc, Kc]   - replaced
                # B[Kc, Nc]   - replaced
                #
                # The use of `prefetchnta` on `A` helps more at this level, as
                # `Mc x Kc` could be a very large chunk. Because `A[Mc, Kc]` is
                # replaced, it is not actually held/re-used at the L3 cache level.
                # Instead, we must stream through it.
                # Because it is also held in the L1 cache, this is a prime candidate
                # for `prefetchnta`, to load slices to the L1 where they may be
                # held and reused, without polluting any of the other caches, where
                # the memory is not re-used.
                var pabk: UnsafePointer[Scalar[elt], alignment=64] = AB
                var pbk: UnsafePointer[Scalar[elt], alignment=64] = pb
                for _ in range(Mc // Mr):  # mr               - hold in l2 cache
                    # Comment #1
                    # Size of slices accessed per iteration:
                    # AB[Mr, Nc]  - replaced
                    # A[Mr, Kc]   - replaced
                    # B[Kc, Nc]   - held
                    #
                    # If we use nontemporal prefetches (i.e. `prefetchnta`
                    # on x86) on `A`, then it won't necessarily be stored in the
                    # L2 cache, or it might be stored but not in a recently used
                    # position, so that it would be quickly evicted, and unlikely
                    # to use more than 1-way from each set it occupies.
                    pbk = pb
                    for _ in range(Nc // WNr):  # lr          - hold in L1 cache
                        # Comment #0
                        # Size of slices accessed per iteration:
                        # AB[Mr, WNr] - replaced on each iter
                        # A[Mr, Kc]   - held
                        # B[Kc, WNr]  - replaced
                        #
                        # These sizes roughly indicate how much data is needed at
                        # a cache level. Here, bbecause `A` is the only array that
                        # can be held, `matmul_ukern` strides across it, touching
                        # only one element per cacheline at a time, while streaming
                        # across `B`.
                        # This means that each cacheline of `B` is touched in
                        # sequence, and thus never retouched during the `ukern`
                        # call, while `A` is retouched a total of
                        # `cacheline_size / sizeof[elt]()`
                        # times. Each cacheline of `A` has thus been touched much more
                        # recently than most cachelines of `B`, allowing us to
                        # keep `A` in the L1 cache while using much larger
                        # values of `Kc` than if we did not do this.
                        # Because `B` must always be loaded from the `L2` cache,
                        # it would be worth trying `prefetch0` to mitigate latency.
                        # However, we access `B` in memory order, so hardware
                        # prefetchers should have little trouble.
                        matmul_ukern[elt, W, Mr, Nr, Kr, Kc // (Stride * Kr)](
                            pabk, pak, pbk, kc != 0
                        )
                        pbk = pbk.offset(WNr * Kc)
                        pabk = pabk.offset(Mr * WNr)
                    pak = pak.offset(Mr * Kc)
                pb = pbk
            pdk = pd
            for _ in range(
                N // Nc
            ):  # nc                     - hold in l3 cache
                # Comment #5
                # Size of slices accessed per iteration:
                # D[Mc, Nc]   - replaced
                # AB[Mc, Kc]  - held
                # C[Kc, Nc]   - replaced
                #
                # Above, we calculated `AB = A * B`.
                # Here, we calculate `D = AB * C`.
                # Hence, `AB` takes the role that `A` took above.
                # However, because of the different loop order for this
                # nest, we hold `AB` in the L3 cache, while we streamed `A`.
                # `AB` was also held in the `L3` cache in th previous subloop,
                # allowing for re-use of the block across these subloops.
                #
                # Instead, we stream through `D` and `C`.
                # `C` is held in the l2 cache, thus we may want to prefetch it
                # with `prefetch0` within the microkernel (although, as we access
                # it in memory order, hardware prefetchers should have little trouble).
                # Hypothetically, we could try something like `clflushopt` to evict
                # old pieces of `C` from the L3 and prevent it from using extra space
                # (this also applies to `B` in the preceding subloop, for `A*B`), but
                # I've never tried something like that before, and it would need size
                # checks of the arrays vs the actual cache size, since you wouldn't
                # want to forcefully flush it for smaller arrays, when everything
                # would have actually fit. Having not tried it, I don't know if it's
                # likely to help performance.
                # We might be able to load `D` with `prefetchnta` when updating it,
                # and using a streaming store to write? Although, this would
                # necessitate fences.
                var pabk: UnsafePointer[Scalar[elt], alignment=64] = AB
                var pck: UnsafePointer[Scalar[elt], alignment=64] = pc
                for _ in range(
                    Mc // Mr
                ):  # mr                - hold in l2 cache
                    # Comment #4
                    # D[Mr, Nc]   - replaced
                    # AB[Mr, Kc]  - replaced (Kc == Nc)
                    # C[Kc, Nc]   - held
                    pck = pc
                    for _ in range(
                        Nc // WNr
                    ):  # nr           - hold in l1 cache
                        # Comment #3
                        # D[Mr, Nr]   - replaced
                        # AB[Mr, Kc]  - held
                        # C[Kc, WNr]  - replaced
                        matmul_ukern[elt, W, Mr, Nr, Kr, Kc // (Stride * Kr)](
                            pdk, pabk, pck, lc != 0
                        )
                        pck = pck.offset(WNr * Kc)
                        pdk = pdk.offset(Mr * WNr)
                    pabk = pabk.offset(Mr * Kc)
                pc = pck
        pa = pak
        pd = pdk


@always_inline
fn bench_b2b[
    elt: DType,
    M: Int,
    N: Int,
    K: Int,
    L: Int,
    W: Int,
    Mc: Int,
    Nc: Int,
    Mr: Int,
    Nr: Int,
    Kr: Int,
](do_benchmark: Bool) raises:
    alias WNr: Int = W * Nr
    alias Stride: Int = stride[elt](WNr)
    alias Kc = Nc
    constrained[Nc % Stride == 0]()
    constrained[Kc % (Kr * Stride) == 0]()

    constrained[Mc % Mr == 0]()
    constrained[Nc % WNr == 0]()

    constrained[M % Mc == 0]()
    constrained[K % Kc == 0]()
    constrained[L % Nc == 0]()
    constrained[N % Nc == 0]()

    alias layout_D: Layout = Layout(
        IntTuple(
            IntTuple(Mr, Mc // Mr, M // Mc),
            IntTuple(Stride, WNr // Stride, Nc // WNr, N // Nc),
        ),
        IntTuple(
            IntTuple(Stride, Mr * Nc, Mc * N),
            IntTuple(1, Mr * Stride, Mr * WNr, Mc * Nc),
        ),
    )
    alias layout_AB: Layout = Layout(
        IntTuple(
            IntTuple(Mr, Mc // Mr, M // Mc),
            IntTuple(Stride, WNr // Stride, Kc // WNr, L // Kc),
        ),
        IntTuple(
            IntTuple(Stride, Mr * Nc, Mc * L),
            IntTuple(1, Mr * Stride, Mr * WNr, Mc * Nc),
        ),
    )
    alias layout_A: Layout = Layout(
        IntTuple(
            IntTuple(Mr, Mc // Mr, M // Mc),
            IntTuple(Stride, WNr // Stride, Kc // WNr, K // Kc),
        ),
        IntTuple(
            IntTuple(Stride, Mr * Kc, Mc * K),
            IntTuple(
                1,
                Mr * Stride,
                Mr * WNr,
                Mc * Kc,
            ),
        ),
    )
    alias layout_B: Layout = Layout(
        IntTuple(
            IntTuple(Stride, Kc // Stride, K // Kc),
            IntTuple(WNr, Nc // WNr, L // Kc),
        ),
        IntTuple(
            IntTuple((WNr * Kc) // Stride, WNr, Nc * Kc),
            IntTuple(
                1,
                WNr * Kc,
                Nc * K,
            ),
        ),
    )
    alias layout_CL_b2b: Layout = Layout(
        IntTuple(
            IntTuple(Stride, Kc // Stride, L // Kc),
            IntTuple(WNr, Nc // WNr, N // Nc),
        ),
        IntTuple(
            IntTuple((WNr * Kc) // Stride, WNr, N * Kc),
            IntTuple(
                1,
                WNr * Kc,
                Nc * Kc,
            ),
        ),
    )
    alias layout_C: Layout = Layout(
        IntTuple(
            IntTuple(Stride, Kc // Stride, L // Kc),
            IntTuple(WNr, Nc // WNr, N // Nc),
        ),
        IntTuple(
            IntTuple((WNr * Kc) // Stride, WNr, Nc * Kc),
            IntTuple(1, WNr * Kc, Nc * L),
        ),
    )

    var Dtile = alloc_tensor[elt, layout_D]()
    var Atile = alloc_tensor[elt, layout_A]()
    var Btile = alloc_tensor[elt, layout_B]()
    var Ctileb2b = alloc_tensor[elt, layout_CL_b2b]()
    var Ctile = alloc_tensor[elt, layout_C]()
    var ABtile = alloc_tensor[elt, layout_AB]()

    var Drm64 = alloc_tensor[DType.float64, Layout.row_major(M, N)]()
    var Arm64 = alloc_tensor[DType.float64, Layout.row_major(M, K)]()
    var Brm64 = alloc_tensor[DType.float64, Layout.row_major(K, L)]()
    var Crm64 = alloc_tensor[DType.float64, Layout.row_major(L, N)]()
    var ABrm64 = alloc_tensor[DType.float64, Layout.row_major(M, L)]()
    rand(Atile.ptr, Atile.layout.size())
    rand(Btile.ptr, Btile.layout.size())
    rand(Ctile.ptr, Ctile.layout.size())
    copy_to(Ctileb2b, Ctile)
    copy_to(Arm64, Atile)
    copy_to(Brm64, Btile)
    copy_to(Crm64, Ctile)
    matmul_naive(ABrm64, Arm64, Brm64)
    matmul_naive(Drm64, ABrm64, Crm64)

    @always_inline
    @parameter
    fn test_tile_fn():
        matmul[elt, M, L, K, W, Mc, Nc, Kc, Mr, Nr, Kr](ABtile, Atile, Btile)
        matmul[elt, M, N, L, W, Mc, Nc, Kc, Mr, Nr, Kr](Dtile, ABtile, Ctile)

    var flops = 2e-9 * (M * K * L + M * L * N)
    if do_benchmark:
        var secs_tile = benchmark.run[test_tile_fn](max_runtime_secs=1.0).mean()
        print("GFLOPS Tile: ", flops / secs_tile)
    else:
        test_tile_fn()

    check_approx_equal[DType.float32](Dtile, Drm64)

    @always_inline
    @parameter
    fn test_tile_b2b_fn():
        matmulb2b[elt, M, N, K, L, W, Mc, Nc, Mr, Nr, Kr](
            Dtile, Atile, Btile, Ctileb2b
        )

    if do_benchmark:
        var secs_tile_b2b = benchmark.run[test_tile_b2b_fn](
            max_runtime_secs=1.0
        ).mean()
        print("GFLOPS B2B:  ", flops / secs_tile_b2b)
    else:
        test_tile_b2b_fn()

    check_approx_equal[DType.float32](Dtile, Drm64)

    Atile.ptr.free()
    Btile.ptr.free()
    Ctile.ptr.free()
    Ctileb2b.ptr.free()
    Dtile.ptr.free()
    ABtile.ptr.free()
    Arm64.ptr.free()
    Brm64.ptr.free()
    Crm64.ptr.free()
    Drm64.ptr.free()
    ABrm64.ptr.free()


fn getMr() -> Int:
    if CompilationTarget.is_x86():
        if CompilationTarget.has_avx512f():
            return 9
    return 6


fn getNr() -> Int:
    if CompilationTarget.is_x86():
        if CompilationTarget.has_avx512f():
            return 3
        else:
            return 2
    return 4


fn main() raises -> None:
    alias elt = DType.float32
    alias W = simdwidthof[elt]()
    alias Mr = getMr()
    alias Nr = getNr()
    alias Kr = 2
    alias Mc = 50 * Mr

    alias Nc = 20 * Nr * W
    alias Stride = stride[DType.float32](W * Nr)
    alias Kc = Nc
    constrained[Kc % Stride == 0]()
    alias M = 4 * Mc
    alias N = 6 * Nc
    alias K = 2 * Kc
    alias L = 5 * Kc
    print("Multiplying M =", M, "; N =", N, "; K =", K, "; L =", L, "\n")
    constrained[Kc == Nc, "b2b requires Kc == Nc"]()
    var do_benchmark: Bool = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--benchmark" or args[i] == "--benchmark=yes":
            do_benchmark = True
    bench_b2b[elt, M, N, K, L, W, Mc, Nc, Mr, Nr, Kr](do_benchmark)
