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

from std.math import fma, isclose
from std.os import abort
from std.random import rand
from std.sys import CompilationTarget, argv, simd_width_of, size_of

import std.benchmark
from std.algorithm.functional import vectorize
from layout import (
    Coord,
    CoordLike,
    ComptimeInt,
    TileTensor,
    TensorLayout,
    coord,
    row_major,
)
from layout.tile_layout import Layout as NativeLayout
from layout.tensor_engine import DefaultEngine
from std.memory import unsafe_stack_allocation
from std.memory.alloc import unsafe_alloc
from std.testing import assert_false

from std.utils import StaticTuple


def _typed_metadata[
    L: TensorLayout
]() -> NativeLayout[shape_types=L._shape_types, stride_types=L._stride_types]:
    """Rebuilds the typed layout value from a layout type.

    Every dimension is known at compile time, so the value follows from the
    type alone.
    """
    comptime assert L.all_dims_known
    return NativeLayout[
        shape_types=L._shape_types, stride_types=L._stride_types
    ]()


def matmul_naive[
    layoutC_type: TensorLayout,
    layoutA_type: TensorLayout,
    layoutB_type: TensorLayout,
    elt: DType,
](
    C: TileTensor[
        mut=True,
        elt,
        layoutC_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    A: TileTensor[
        mut=True,
        elt,
        layoutA_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    B: TileTensor[
        mut=True,
        elt,
        layoutB_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
):
    comptime assert (
        C.rank
        == C.flat_rank
        == A.rank
        == A.flat_rank
        == B.rank
        == B.flat_rank
        == 2
    )
    comptime M: Int = C.static_shape[0]
    comptime N: Int = C.static_shape[1]
    comptime K: Int = A.static_shape[1]
    comptime assert M == A.static_shape[0]
    comptime assert N == B.static_shape[1]
    comptime assert K == B.static_shape[0]
    for m in range(M):
        for n in range(N):
            C[m, n] = Scalar[elt]()
    for m in range(M):
        for k in range(K):
            for n in range(N):
                C[m, n] += A[m, k] * B[k, n]


comptime cacheline_size: Int = 64


# We should be able to support 1-access per cacheline
# even when nr * width < cacheline_size // size_of[elt]()
# For Apple Silicon, the values for `float32` would be
# 4 * 4 < 128 // 4
# Firestorm core's l1 caches are so large, that we wouldn't
# really need this, though.
# Also: cacheline_size of 64 is currently hard coded.
@inline(.always)
def stride[elt: DType](nrw: Int) -> Int:
    if nrw * size_of[elt]() >= cacheline_size:
        return cacheline_size // size_of[elt]()
    else:
        return nrw


# Assumes that we have packed `A` and `B`, `C` also uses a packed layout.
def matmul_ukern[
    elt: DType, width: Int, mr: Int, nr: Int, kr: Int, kf: Int
](
    C: MutPointer[Scalar[elt], _],
    A: ImmPointer[Scalar[elt], _],
    B: ImmPointer[Scalar[elt], _],
    inc: Bool,
):
    comptime Align: Int = size_of[elt]() * width
    comptime Astride: Int = stride[elt](nr * width)
    comptime CstoreReps: Int = nr * width // Astride
    comptime assert CstoreReps * Astride == nr * width
    comptime CstoresPer: Int = Astride // width
    comptime assert CstoresPer * width == Astride
    comptime assert CstoresPer * CstoreReps == nr

    var acc: StaticTuple[SIMD[elt, width], mr * nr] = StaticTuple[
        SIMD[elt, width], mr * nr
    ]()

    comptime for m in range(mr):
        comptime for n in range(nr):
            acc[n + m * nr] = SIMD[elt, width]()
    var Bloads: StaticTuple[SIMD[elt, width], nr] = StaticTuple[
        SIMD[elt, width], nr
    ]()
    # We're assuming 64-byte cachelines
    # This is a trick to repeatedly re-touch A's memory in the microkernel,
    # so that A can stay in the L1-cache, while we stream B through it.

    var Ao = A
    var Bo = B
    # TODO: static assert that kf%Astride == 0
    for _ in range(Astride):
        # Because we repeatedly call `matmul_ukern` with the same
        # slice of `A`, but different slice of `B`, we wish for `A`
        # to remain in the l1 cache, but freely evict `B`.
        # Repeatedly re-touching the cachelines of `A` helps us achieve this.
        var Atmp = Ao
        Ao = Ao.unsafe_offset(1)
        for _ in range(kf):
            comptime for _ in range(kr):
                comptime for n in range(nr):
                    Bloads[n] = Bo.unsafe_load[width=width, alignment=Align](
                        n * width
                    )

                comptime for m in range(mr):
                    var Abroadcast: SIMD[elt, width] = SIMD[elt, width](
                        Atmp.unsafe_load[width=1, alignment=size_of[elt]()](
                            m * Astride
                        )
                    )

                    comptime for n in range(nr):
                        acc[n + m * nr] = fma(
                            Abroadcast, Bloads[n], acc[n + m * nr]
                        )
                Atmp = Atmp.unsafe_offset(mr * Astride)
                Bo = Bo.unsafe_offset(nr * width)
    if inc:
        # Note, `C` would have spilled from the L1 cache by the time
        # we load it again had we loaded before the reduction loop.
        comptime for m in range(mr):
            comptime for n0 in range(CstoreReps):
                comptime for n1 in range(CstoresPer):
                    acc[n1 + n0 * CstoresPer + m * nr] = acc[
                        n1 + n0 * CstoresPer + m * nr
                    ] + C.unsafe_load[width=width, alignment=Align](
                        (n1 + m * CstoresPer + n0 * (mr * CstoresPer)) * width
                    )

    comptime for m in range(mr):
        comptime for n0 in range(CstoreReps):
            comptime for n1 in range(CstoresPer):
                C.unsafe_store[alignment=Align](
                    (n1 + m * CstoresPer + n0 * (mr * CstoresPer)) * width,
                    acc[n1 + n0 * CstoresPer + m * nr],
                )


# C = A * B
# C and A have the same layout, to facilitate the next step of
# D = (A * B) * C
# i.e., make it easier to chain matrix multiplies.
# A's shape is (W*Mr, Mc/(W*Mr), M/Mc), (Kc, K/Kc)
# A's strides are (1, W*Mr*Kc, Mc*K), (W*Mr, Mc*Kc)
# B's shape is (Kc*size_of(elt)/64, 64/size_of(elt), K/Kc), (nr,Nc/nr,N/Nc)
# B's strides are ((nr*64/size_of(elt), 1, Nc*Kc), (64/size_of(elt), nr*Kc, Nc*K)
#
def matmul[
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
    layoutC_type: TensorLayout,
    layoutA_type: TensorLayout,
    layoutB_type: TensorLayout,
](
    C: TileTensor[
        mut=True,
        elt,
        layoutC_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    A: TileTensor[
        mut=True,
        elt,
        layoutA_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    B: TileTensor[
        mut=True,
        elt,
        layoutB_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
):
    comptime assert C.rank == A.rank == B.rank == 2
    comptime WNr = W * Nr
    comptime Stride = stride[elt](WNr)

    var pc = C.ptr
    var pa = A.ptr
    # TODO: nontemporal prefetches on the microkernel slices of `B`
    #       as the slice does not get reused at the L2 or L3 level.
    # TODO: prefetches on `A`, to hide latency, as we stream it through
    #       the L1, suffering L2->register latency for each load.
    # NOTE: Read comments within the loop from the inside out.
    for _ in range(M // Mc):
        var pb = B.ptr
        var pak = pa
        for _ in range(N // Nc):
            var pck = pc
            pak = pa
            for kc in range(K // Kc):
                var pbk = pb
                pck = pc
                for _ in range(Mc // Mr):  # mr
                    pbk = pb
                    for _ in range(Nc // WNr):  # nr
                        matmul_ukern[elt, W, Mr, Nr, Kr, Kc // (Stride * Kr)](
                            pck, pak, pbk, kc != 0
                        )
                        pbk = pbk.unsafe_offset(WNr * Kc)
                        pck = pck.unsafe_offset(Mr * WNr)
                    pak = pak.unsafe_offset(Mr * Kc)
                pb = pbk
            pc = pck
        pa = pak


def alloc_tensor[
    elt: DType, layout: TensorLayout
](rtlayout: layout) -> TileTensor[mut=True, elt, layout, MutUntrackedOrigin]:
    return TileTensor[mut=True, elt, layout, MutUntrackedOrigin](
        unsafe_alloc[Scalar[elt]](Int(rtlayout.product()), alignment=64),
        rtlayout,
    )


def max_min_idx_positive(x: List[Int], y: List[Int]) -> Int:
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


def delete_idx(arg: List[Int], idx: Int) -> List[Int]:
    var res = List[Int]()
    res.reserve(len(arg) - 1)
    for i in range(len(arg)):
        if i != idx:
            res.append(arg[i])
    return res^


@inline(.always)
def strided_load[
    elt: DType, //, W: Int, X: Int
](p: ImmPointer[Scalar[elt], _], i: Int) -> SIMD[elt, W]:
    comptime if X == 1:
        return p.unsafe_load[width=W](i)
    else:
        return p.unsafe_offset(i * X).unsafe_strided_load[width=W](X)


@inline(.always)
def strided_store[
    elt: DType, W: Int, //, X: Int
](p: MutPointer[Scalar[elt], _], i: Int, x: SIMD[elt, W]):
    comptime if X == 1:
        p.unsafe_store(i, x)
    else:
        p.unsafe_offset(i * X).unsafe_strided_store(x, X)


@inline(.always)
def vectorize_flat[
    elt_a: DType,
    elt_b: DType,
    //,
    simd_width: Int,
    unroll_factor: Int,
    shape: List[Int],
    stride_a: List[Int],
    stride_b: List[Int],
](
    a: MutPointer[Scalar[elt_a], _],
    b: ImmPointer[Scalar[elt_b], _],
    f: Some[
        def[
            width: Int, stride_a: Int, stride_b: Int
        ](
            MutPointer[Scalar[elt_a], _],
            ImmPointer[Scalar[elt_b], _],
            Int,
        ) -> None
    ],
):
    comptime assert len(shape) == len(stride_a)
    comptime assert len(shape) == len(stride_b)

    comptime if len(shape) == 1:
        # perform the copy
        comptime int_stride_a: Int = stride_a[0]
        comptime int_stride_b: Int = stride_b[0]
        comptime size = shape[0]

        @inline(.always)
        def vf[width: Int](i: Int) {var a, var b, imm f}:
            f[width, int_stride_a, int_stride_b](a, b, i)

        vectorize[
            simd_width,
            unroll_factor=min(size // simd_width, unroll_factor),
        ](size, vf)
    else:
        # we find the maximum min stride, subset, and loop over it.
        comptime max_idx = max_min_idx_positive(stride_b, stride_a)
        comptime subset_shape = delete_idx(shape, max_idx)
        comptime subset_stride_b = delete_idx(stride_b, max_idx)
        comptime subset_stride_a = delete_idx(stride_a, max_idx)
        comptime loop_size: Int = shape[max_idx]
        comptime a_stride: Int = stride_a[max_idx]
        comptime b_stride: Int = stride_b[max_idx]
        for i in range(loop_size):
            vectorize_flat[
                simd_width,
                unroll_factor,
                subset_shape,
                subset_stride_a,
                subset_stride_b,
            ](a.unsafe_offset(i * a_stride), b.unsafe_offset(i * b_stride), f)


struct _ExpandedModes(Copyable, Movable):
    """Flat shape and per-operand strides of two mode-aligned layouts."""

    var shape: List[Int]
    """Common shape leaves, in depth-first order."""

    var stride_a: List[Int]
    """Stride leaves of the first layout, aligned with `shape`."""

    var stride_b: List[Int]
    """Stride leaves of the second layout, aligned with `shape`."""

    def __init__(out self):
        self.shape = List[Int]()
        self.stride_a = List[Int]()
        self.stride_b = List[Int]()


@always_inline
def _emit_flat[
    s: CoordLike, t: CoordLike
](shape: s, stride: t, mut out_shape: List[Int], mut out_stride: List[Int],):
    """Appends the leaves of one shape/stride mode in depth-first order."""
    comptime if s.is_tuple:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()
        comptime for i in range(s.__len__()):
            _emit_flat(shape_t[i], stride_t[i], out_shape, out_stride)
    else:
        out_shape.append(Int(shape))
        out_stride.append(Int(stride))


@always_inline
def _emit_expanded_strides[
    s: CoordLike
](shape: s, mut stride_value: Int, mut out: List[Int]):
    """Expands one scalar stride over the leaves of `shape`.

    The first leaf keeps `stride_value` and every later leaf steps by the
    product of the leaves preceding it, so a scalar mode of extent `n`
    contributes `n` consecutive strides. This is the typed form of
    `expand_strides`.
    """
    comptime if s.is_tuple:
        var shape_t = shape.tuple()
        comptime for i in range(s.__len__()):
            _emit_expanded_strides(shape_t[i], stride_value, out)
    else:
        out.append(stride_value)
        stride_value *= Int(shape)


@always_inline
def _expand_modes_alike[
    sa: CoordLike, ta: CoordLike, sb: CoordLike, tb: CoordLike
](
    shape_a: sa,
    stride_a: ta,
    shape_b: sb,
    stride_b: tb,
    mut expanded: _ExpandedModes,
):
    """Aligns two layouts' mode nesting and appends their flat leaves.

    Mirrors `layout.layout.expand_modes_alike`: a scalar mode facing a nested
    one is expanded leaf by leaf with `_emit_expanded_strides`, so both
    operands end up with one stride per common shape leaf.
    """
    comptime assert sa.__len__() == sb.__len__(), (
        "expand_modes_alike requires the same mode count, got "
        + String(sa.__len__())
        + " and "
        + String(sb.__len__())
    )
    var shape_a_t = shape_a.tuple()
    var stride_a_t = stride_a.tuple()
    var shape_b_t = shape_b.tuple()
    var stride_b_t = stride_b.tuple()
    comptime for i in range(sa.__len__()):
        comptime a_is_tuple = sa.ParamListType[i].is_tuple
        comptime b_is_tuple = sb.ParamListType[i].is_tuple
        comptime if a_is_tuple and b_is_tuple:
            _expand_modes_alike(
                shape_a_t[i],
                stride_a_t[i],
                shape_b_t[i],
                stride_b_t[i],
                expanded,
            )
        elif a_is_tuple:
            _emit_flat(
                shape_a_t[i], stride_a_t[i], expanded.shape, expanded.stride_a
            )
            var b_stride = Int(stride_b_t[i])
            _emit_expanded_strides(shape_a_t[i], b_stride, expanded.stride_b)
        elif b_is_tuple:
            var a_stride = Int(stride_a_t[i])
            _emit_expanded_strides(shape_b_t[i], a_stride, expanded.stride_a)
            _emit_flat(
                shape_b_t[i], stride_b_t[i], expanded.shape, expanded.stride_b
            )
        else:
            # Both modes are scalars; `expand_modes_alike` keeps the second
            # shape here.
            expanded.shape.append(Int(shape_b_t[i]))
            expanded.stride_a.append(Int(stride_a_t[i]))
            expanded.stride_b.append(Int(stride_b_t[i]))


def _flat_modes[
    layout_a_type: TensorLayout, layout_b_type: TensorLayout
]() -> _ExpandedModes:
    """Returns the flat, mode-aligned shape/strides of two layouts."""
    comptime layout_a = _typed_metadata[layout_a_type]()
    comptime layout_b = _typed_metadata[layout_b_type]()
    var expanded = _ExpandedModes()
    _expand_modes_alike(
        layout_a.shape_coord(),
        layout_a.stride_coord(),
        layout_b.shape_coord(),
        layout_b.stride_coord(),
        expanded,
    )
    return expanded^


def vectorize_tile_tensor[
    elt_a: DType,
    layout_a_type: TensorLayout,
    elt_b: DType,
    layout_b_type: TensorLayout,
    //,
    simd_width: Int = max(simd_width_of[elt_a](), simd_width_of[elt_b]()),
    unroll_factor: Int = 4,
](
    a: TileTensor[
        mut=True,
        elt_a,
        layout_a_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    b: TileTensor[
        mut=True,
        elt_b,
        layout_b_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    f: Some[
        def[
            width: Int, stride_a: Int, stride_b: Int
        ](
            MutPointer[Scalar[elt_a], _],
            ImmPointer[Scalar[elt_b], _],
            Int,
        ) -> None
    ],
):
    comptime expanded = _flat_modes[layout_a_type, layout_b_type]()
    vectorize_flat[
        simd_width,
        unroll_factor,
        expanded.shape,
        expanded.stride_a,
        expanded.stride_b,
    ](a.ptr, b.ptr, f)


def copy_to[
    elt_dst: DType,
    layout_dst_type: TensorLayout,
    elt_src: DType,
    layout_src_type: TensorLayout,
    //,
    simd_width: Int = max(simd_width_of[elt_dst](), simd_width_of[elt_src]()),
    unroll_factor: Int = 4,
](
    dst: TileTensor[
        mut=True,
        elt_dst,
        layout_dst_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    src: TileTensor[
        mut=True,
        elt_src,
        layout_src_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
):
    comptime layout_dst = _typed_metadata[layout_dst_type]()
    comptime layout_src = _typed_metadata[layout_src_type]()

    @inline(.always)
    def copy[
        width: Int, stride_a: Int, stride_b: Int
    ](
        dstp: MutPointer[Scalar[elt_dst], _],
        srcp: ImmPointer[Scalar[elt_src], _],
        i: Int,
    ) {var}:
        var vsrc = strided_load[width, stride_b](srcp, i)
        strided_store[stride_a](dstp, i, vsrc.cast[elt_dst]())

    vectorize_tile_tensor[simd_width, unroll_factor](dst, src, copy)


def check_approx_equal[
    elt_dst: DType,
    layout_dst_type: TensorLayout,
    elt_src: DType,
    layout_src_type: TensorLayout,
    //,
    cmp_elt: DType,
    simd_width: Int = max(simd_width_of[elt_dst](), simd_width_of[elt_src]()),
    *,
    unroll_factor: Int = 4,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
](
    dst: TileTensor[
        mut=True,
        elt_dst,
        layout_dst_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    src: TileTensor[
        mut=True,
        elt_src,
        layout_src_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
) raises:
    comptime layout_dst = _typed_metadata[layout_dst_type]()
    comptime layout_src = _typed_metadata[layout_src_type]()
    var fail: Bool = False

    @inline(.always)
    def check[
        width: Int, stride_a: Int, stride_b: Int
    ](
        pa: MutPointer[Scalar[elt_dst], _],
        pb: ImmPointer[Scalar[elt_src], _],
        i: Int,
    ) {mut}:
        var va = strided_load[width, stride_a](pa, i).cast[cmp_elt]()
        var vb = strided_load[width, stride_b](pb, i).cast[cmp_elt]()
        if not all(isclose(va, vb, atol=atol, rtol=rtol, equal_nan=equal_nan)):
            fail = True

    vectorize_tile_tensor[simd_width, unroll_factor](dst, src, check)
    assert_false(fail)


# Kc == Nc, so don't need to specify both
def matmulb2b[
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
    layoutD_type: TensorLayout,
    layoutA_type: TensorLayout,
    layoutB_type: TensorLayout,
    layoutC_type: TensorLayout,
](
    D: TileTensor[
        mut=True,
        elt,
        layoutD_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    A: TileTensor[
        mut=True,
        elt,
        layoutA_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    B: TileTensor[
        mut=True,
        elt,
        layoutB_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    C: TileTensor[
        mut=True,
        elt,
        layoutC_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
):
    comptime assert D.rank == A.rank == B.rank == C.rank == 2
    comptime WNr = W * Nr
    comptime Stride = stride[elt](WNr)
    comptime Kc = Nc

    var pa = A.ptr
    var pd = D.ptr
    # Should we support heap-allocating and passing it in?
    var AB = unsafe_stack_allocation[Mc * Nc, elt, alignment=64]()
    # TODO: prefetches, as described in nest
    # NOTE: Read comments within the loop from the inside out.
    #       I.e., read following a post-order depth first traversal of the
    #       loop tree.
    for _ in range(M // Mc):  # mc
        var pb = B.ptr
        var pc = C.ptr
        var pak = pa
        var pdk = pd
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
                # replaced, it is not actually held/reused at the L3 cache level.
                # Instead, we must stream through it.
                # Because it is also held in the L1 cache, this is a prime candidate
                # for `prefetchnta`, to load slices to the L1 where they may be
                # held and reused, without polluting any of the other caches, where
                # the memory is not reused.
                var pabk = AB
                var pbk = pb
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
                        # a cache level. Here, because `A` is the only array that
                        # can be held, `matmul_ukern` strides across it, touching
                        # only one element per cacheline at a time, while streaming
                        # across `B`.
                        # This means that each cacheline of `B` is touched in
                        # sequence, and thus never retouched during the `ukern`
                        # call, while `A` is retouched a total of
                        # `cacheline_size / size_of[elt]()`
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
                        pbk = pbk.unsafe_offset(WNr * Kc)
                        pabk = pabk.unsafe_offset(Mr * WNr)
                    pak = pak.unsafe_offset(Mr * Kc)
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
                # `AB` was also held in the `L3` cache in the previous subloop,
                # allowing for reuse of the block across these subloops.
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
                var pabk = AB
                var pck = pc
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
                        pck = pck.unsafe_offset(WNr * Kc)
                        pdk = pdk.unsafe_offset(Mr * WNr)
                    pabk = pabk.unsafe_offset(Mr * Kc)
                pc = pck
        pa = pak
        pd = pdk


@inline(.always)
def bench_b2b[
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
    comptime WNr: Int = W * Nr
    comptime Stride: Int = stride[elt](WNr)
    comptime Kc = Nc
    comptime assert Nc % Stride == 0
    comptime assert Kc % (Kr * Stride) == 0

    comptime assert Mc % Mr == 0
    comptime assert Nc % WNr == 0

    comptime assert M % Mc == 0
    comptime assert K % Kc == 0
    comptime assert L % Nc == 0
    comptime assert N % Nc == 0

    comptime layout_D = NativeLayout(
        Coord(
            coord[Mr, Mc // Mr, M // Mc],
            coord[Stride, WNr // Stride, Nc // WNr, N // Nc],
        ),
        Coord(
            coord[Stride, Mr * Nc, Mc * N],
            coord[1, Mr * Stride, Mr * WNr, Mc * Nc],
        ),
    )
    comptime layout_AB = NativeLayout(
        Coord(
            coord[Mr, Mc // Mr, M // Mc],
            coord[Stride, WNr // Stride, Kc // WNr, L // Kc],
        ),
        Coord(
            coord[Stride, Mr * Nc, Mc * L],
            coord[1, Mr * Stride, Mr * WNr, Mc * Nc],
        ),
    )
    comptime layout_A = NativeLayout(
        Coord(
            coord[Mr, Mc // Mr, M // Mc],
            coord[Stride, WNr // Stride, Kc // WNr, K // Kc],
        ),
        Coord(
            coord[Stride, Mr * Kc, Mc * K],
            coord[1, Mr * Stride, Mr * WNr, Mc * Kc],
        ),
    )
    comptime layout_B = NativeLayout(
        Coord(
            coord[Stride, Kc // Stride, K // Kc],
            coord[WNr, Nc // WNr, L // Kc],
        ),
        Coord(
            coord[(WNr * Kc) // Stride, WNr, Nc * Kc],
            coord[1, WNr * Kc, Nc * K],
        ),
    )
    comptime layout_CL_b2b = NativeLayout(
        Coord(
            coord[Stride, Kc // Stride, L // Kc],
            coord[WNr, Nc // WNr, N // Nc],
        ),
        Coord(
            coord[(WNr * Kc) // Stride, WNr, N * Kc],
            coord[1, WNr * Kc, Nc * Kc],
        ),
    )
    comptime layout_C = NativeLayout(
        Coord(
            coord[Stride, Kc // Stride, L // Kc],
            coord[WNr, Nc // WNr, N // Nc],
        ),
        Coord(
            coord[(WNr * Kc) // Stride, WNr, Nc * Kc],
            coord[1, WNr * Kc, Nc * L],
        ),
    )

    # The plan functions below are generic over the tensor layouts, so the
    # fixtures are validated here, where they are concrete values.
    #
    # `matmul`'s view: D/AB share the C layout, A and B their own.
    comptime assert layout_AB.rank == 2
    comptime assert layout_A.rank == 2
    comptime assert layout_B.rank == 2
    # I am assuming that the `shape` and `stride` are congruent (i.e., equal length)
    # so that I don't need to check both here.
    comptime assert layout_AB.shape_coord()[0].__len__() == 3
    comptime assert layout_AB.shape_coord()[1].__len__() == 4
    comptime assert layout_A.shape_coord()[0].__len__() == 3
    comptime assert layout_A.shape_coord()[1].__len__() == 4
    comptime assert layout_B.shape_coord()[0].__len__() == 3
    comptime assert layout_B.shape_coord()[1].__len__() == 3

    # Matrix C
    comptime assert Int(layout_AB.shape_coord()[0][0]) == Mr
    comptime assert Int(layout_AB.shape_coord()[0][1]) * Mr == Mc
    comptime assert Int(layout_AB.shape_coord()[0][2]) * Mc == M

    comptime assert Int(layout_AB.shape_coord()[1][0]) == Stride
    comptime assert Int(layout_AB.shape_coord()[1][1]) * Stride == WNr
    comptime assert Int(layout_AB.shape_coord()[1][2]) * WNr == Nc
    comptime assert Int(layout_AB.shape_coord()[1][3]) * Nc == L

    comptime assert Int(layout_AB.stride_coord()[0][0]) == Stride
    comptime assert Int(layout_AB.stride_coord()[0][1]) == Mr * Nc
    comptime assert Int(layout_AB.stride_coord()[0][2]) == Mc * L

    comptime assert Int(layout_AB.stride_coord()[1][0]) == 1
    comptime assert Int(layout_AB.stride_coord()[1][1]) == Mr * Stride
    comptime assert Int(layout_AB.stride_coord()[1][2]) == Mr * WNr
    comptime assert Int(layout_AB.stride_coord()[1][3]) == Mc * Nc

    # Matrix A
    comptime assert Int(layout_A.shape_coord()[0][0]) == Mr
    comptime assert Int(layout_A.shape_coord()[0][1]) * Mr == Mc
    comptime assert Int(layout_A.shape_coord()[0][2]) * Mc == M

    comptime assert Int(layout_A.shape_coord()[1][0]) == Stride
    comptime assert Int(layout_A.shape_coord()[1][1]) * Stride == WNr
    comptime assert Int(layout_A.shape_coord()[1][2]) * WNr == Kc
    comptime assert Int(layout_A.shape_coord()[1][3]) * Kc == K

    comptime assert Int(layout_A.stride_coord()[0][0]) == Stride
    comptime assert Int(layout_A.stride_coord()[0][1]) == Mr * Kc
    comptime assert Int(layout_A.stride_coord()[0][2]) == Mc * K

    comptime assert Int(layout_A.stride_coord()[1][0]) == 1
    comptime assert Int(layout_A.stride_coord()[1][1]) == Mr * Stride
    comptime assert Int(layout_A.stride_coord()[1][2]) == Mr * WNr
    comptime assert Int(layout_A.stride_coord()[1][3]) == Mc * Kc

    # Matrix B
    comptime assert Int(layout_B.shape_coord()[0][0]) == Stride
    comptime assert Int(layout_B.shape_coord()[0][1]) * Stride == Kc
    comptime assert Int(layout_B.shape_coord()[0][2]) * Kc == K

    comptime assert Int(layout_B.shape_coord()[1][0]) == WNr
    comptime assert Int(layout_B.shape_coord()[1][1]) * WNr == Nc
    comptime assert Int(layout_B.shape_coord()[1][2]) * Nc == L

    comptime assert Int(layout_B.stride_coord()[0][0]) * Stride == WNr * Kc
    comptime assert Int(layout_B.stride_coord()[0][1]) == WNr
    comptime assert Int(layout_B.stride_coord()[0][2]) == Nc * Kc

    comptime assert Int(layout_B.stride_coord()[1][0]) == 1
    comptime assert Int(layout_B.stride_coord()[1][1]) == WNr * Kc
    comptime assert Int(layout_B.stride_coord()[1][2]) == Nc * K

    # `matmulb2b`'s view, which also covers the fused-back-to-back C tile.
    comptime assert layout_D.rank == 2
    comptime assert layout_A.rank == 2
    comptime assert layout_B.rank == 2
    comptime assert layout_CL_b2b.rank == 2

    comptime assert layout_D.shape_coord()[0].__len__() == 3
    comptime assert layout_D.shape_coord()[1].__len__() == 4
    comptime assert layout_A.shape_coord()[0].__len__() == 3
    comptime assert layout_A.shape_coord()[1].__len__() == 4
    comptime assert layout_B.shape_coord()[0].__len__() == 3
    comptime assert layout_B.shape_coord()[1].__len__() == 3
    comptime assert layout_CL_b2b.shape_coord()[0].__len__() == 3
    comptime assert layout_CL_b2b.shape_coord()[1].__len__() == 3

    # Matrix D
    comptime assert Int(layout_D.shape_coord()[0][0]) == Mr
    comptime assert Int(layout_D.shape_coord()[0][1]) * Mr == Mc
    comptime assert Int(layout_D.shape_coord()[0][2]) * Mc == M

    comptime assert Int(layout_D.shape_coord()[1][0]) == Stride
    comptime assert Int(layout_D.shape_coord()[1][1]) * Stride == WNr
    comptime assert Int(layout_D.shape_coord()[1][2]) * WNr == Nc
    comptime assert Int(layout_D.shape_coord()[1][3]) * Nc == N

    comptime assert Int(layout_D.stride_coord()[0][0]) == Stride
    comptime assert Int(layout_D.stride_coord()[0][1]) == Mr * Nc
    comptime assert Int(layout_D.stride_coord()[0][2]) == Mc * N

    comptime assert Int(layout_D.stride_coord()[1][0]) == 1
    comptime assert Int(layout_D.stride_coord()[1][1]) == Mr * Stride
    comptime assert Int(layout_D.stride_coord()[1][2]) == Mr * WNr
    comptime assert Int(layout_D.stride_coord()[1][3]) == Mc * Nc

    # Matrix A
    comptime assert Int(layout_A.shape_coord()[0][0]) == Mr
    comptime assert Int(layout_A.shape_coord()[0][1]) * Mr == Mc
    comptime assert Int(layout_A.shape_coord()[0][2]) * Mc == M

    comptime assert Int(layout_A.shape_coord()[1][0]) == Stride
    comptime assert Int(layout_A.shape_coord()[1][1]) * Stride == WNr
    comptime assert Int(layout_A.shape_coord()[1][2]) * WNr == Kc
    comptime assert Int(layout_A.shape_coord()[1][3]) * Kc == K

    comptime assert Int(layout_A.stride_coord()[0][0]) == Stride
    comptime assert Int(layout_A.stride_coord()[0][1]) == Mr * Kc
    comptime assert Int(layout_A.stride_coord()[0][2]) == Mc * K

    comptime assert Int(layout_A.stride_coord()[1][0]) == 1
    comptime assert Int(layout_A.stride_coord()[1][1]) == Mr * Stride
    comptime assert Int(layout_A.stride_coord()[1][2]) == Mr * WNr
    comptime assert Int(layout_A.stride_coord()[1][3]) == Mc * Kc

    # Matrix B
    comptime assert Int(layout_B.shape_coord()[0][0]) == Stride
    comptime assert Int(layout_B.shape_coord()[0][1]) * Stride == Kc
    comptime assert Int(layout_B.shape_coord()[0][2]) * Kc == K

    comptime assert Int(layout_B.shape_coord()[1][0]) == WNr
    comptime assert Int(layout_B.shape_coord()[1][1]) * WNr == Nc
    comptime assert Int(layout_B.shape_coord()[1][2]) * Nc == L

    comptime assert Int(layout_B.stride_coord()[0][0]) * Stride == WNr * Nc
    comptime assert Int(layout_B.stride_coord()[0][1]) == WNr
    comptime assert Int(layout_B.stride_coord()[0][2]) == Nc * Kc

    comptime assert Int(layout_B.stride_coord()[1][0]) == 1
    comptime assert Int(layout_B.stride_coord()[1][1]) == WNr * Kc
    comptime assert Int(layout_B.stride_coord()[1][2]) == Nc * K

    # Matrix C
    comptime assert Int(layout_CL_b2b.shape_coord()[0][0]) == Stride
    comptime assert Int(layout_CL_b2b.shape_coord()[0][1]) * Stride == Kc
    comptime assert Int(layout_CL_b2b.shape_coord()[0][2]) * Kc == L

    comptime assert Int(layout_CL_b2b.shape_coord()[1][0]) == WNr
    comptime assert Int(layout_CL_b2b.shape_coord()[1][1]) * WNr == Nc
    comptime assert Int(layout_CL_b2b.shape_coord()[1][2]) * Nc == N

    comptime assert Int(layout_CL_b2b.stride_coord()[0][0]) * Stride == WNr * Kc
    comptime assert Int(layout_CL_b2b.stride_coord()[0][1]) == WNr
    comptime assert Int(layout_CL_b2b.stride_coord()[0][2]) == N * Kc

    comptime assert Int(layout_CL_b2b.stride_coord()[1][0]) == 1
    comptime assert Int(layout_CL_b2b.stride_coord()[1][1]) == WNr * Kc
    comptime assert Int(layout_CL_b2b.stride_coord()[1][2]) == Nc * Kc

    var Dtile = alloc_tensor[elt](layout_D)
    var Atile = alloc_tensor[elt](layout_A)
    var Btile = alloc_tensor[elt](layout_B)
    var Ctileb2b = alloc_tensor[elt](layout_CL_b2b)
    var Ctile = alloc_tensor[elt](layout_C)
    var ABtile = alloc_tensor[elt](layout_AB)

    var Drm64 = alloc_tensor[.float64](row_major[M, N]())
    var Arm64 = alloc_tensor[.float64](row_major[M, K]())
    var Brm64 = alloc_tensor[.float64](row_major[K, L]())
    var Crm64 = alloc_tensor[.float64](row_major[L, N]())
    var ABrm64 = alloc_tensor[.float64](row_major[M, L]())
    comptime layout_A_size: Int = layout_A.size()
    comptime layout_B_size: Int = layout_B.size()
    comptime layout_C_size: Int = layout_C.size()
    rand(Atile.ptr, layout_A_size)
    rand(Btile.ptr, layout_B_size)
    rand(Ctile.ptr, layout_C_size)
    copy_to(Ctileb2b, Ctile)
    copy_to(Arm64, Atile)
    copy_to(Brm64, Btile)
    copy_to(Crm64, Ctile)
    matmul_naive(ABrm64, Arm64, Brm64)
    matmul_naive(Drm64, ABrm64, Crm64)

    @inline(.always)
    def test_tile_fn() {var}:
        matmul[elt, M, L, K, W, Mc, Nc, Kc, Mr, Nr, Kr](ABtile, Atile, Btile)
        matmul[elt, M, N, L, W, Mc, Nc, Kc, Mr, Nr, Kr](Dtile, ABtile, Ctile)

    var flops = 2e-9 * Float64(M * K * L + M * L * N)
    if do_benchmark:
        var secs_tile = std.benchmark.run(
            test_tile_fn, max_runtime_secs=1.0
        ).mean()
        print("GFLOPS Tile: ", flops / secs_tile)
    else:
        test_tile_fn()

    check_approx_equal[.float32](Dtile, Drm64)

    @inline(.always)
    def test_tile_b2b_fn() {var}:
        matmulb2b[elt, M, N, K, L, W, Mc, Nc, Mr, Nr, Kr](
            Dtile, Atile, Btile, Ctileb2b
        )

    if do_benchmark:
        var secs_tile_b2b = std.benchmark.run(
            test_tile_b2b_fn, max_runtime_secs=1.0
        ).mean()
        print("GFLOPS B2B:  ", flops / secs_tile_b2b)
    else:
        test_tile_b2b_fn()

    check_approx_equal[.float32](Dtile, Drm64)

    Atile.ptr.unsafe_free()
    Btile.ptr.unsafe_free()
    Ctile.ptr.unsafe_free()
    Ctileb2b.ptr.unsafe_free()
    Dtile.ptr.unsafe_free()
    ABtile.ptr.unsafe_free()
    Arm64.ptr.unsafe_free()
    Brm64.ptr.unsafe_free()
    Crm64.ptr.unsafe_free()
    Drm64.ptr.unsafe_free()
    ABrm64.ptr.unsafe_free()


def getMr() -> Int:
    if CompilationTarget.is_x86():
        if CompilationTarget.has_avx512f():
            return 9
    return 6


def getNr() -> Int:
    if CompilationTarget.is_x86():
        if CompilationTarget.has_avx512f():
            return 3
        else:
            return 2
    return 4


def main() raises -> None:
    comptime elt = DType.float32
    comptime W = simd_width_of[elt]()
    comptime Mr = getMr()
    comptime Nr = getNr()
    comptime Kr = 2
    comptime Mc = 50 * Mr

    comptime Nc = 20 * Nr * W
    comptime Stride = stride[.float32](W * Nr)
    comptime Kc = Nc
    comptime assert Kc % Stride == 0
    comptime M = 4 * Mc
    comptime N = 6 * Nc
    comptime K = 2 * Kc
    comptime L = 5 * Kc
    print(t"Multiplying M = {M} ; N = {N} ; K = {K} ; L = {L} \n")
    comptime assert Kc == Nc, "b2b requires Kc == Nc"
    var do_benchmark: Bool = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--benchmark" or args[i] == "--benchmark=yes":
            do_benchmark = True
    bench_b2b[elt, M, N, K, L, W, Mc, Nc, Mr, Nr, Kr](do_benchmark)
