# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# ===-----------------------------------------------------------------------===#
# Motivation of thread swizzling                                               #
# ===-----------------------------------------------------------------------===#


# |--------| : A vector of 4 FP32 elements.
#
# Assumptions:
# * BM x BK = 64 x 16
# * Row-major matrix
#
# A BM x BN tile in global memory, showing 8 rows:
#
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ...
#
# This tile is be copied to shared memory. Q: How to map threads to vectors?
#
# Naive method:
# Global memory view:
#
#     ... |-- T0 --|-- T1 --|-- T2 --|-- T3 --| ...
#     ... |-- T4 --|-- T5 --|---T6 --|-- T7 --| ...
#     ... |-- T8 --|-- T9 --|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|-- T31--| ...
#
# Shared memory view:
# The layout is row_major(BM, BK) but I draw 8 vectors per row. It's easier to
# to show bank conflicts since each row covers all 32 banks.
#
#     |-- T0 --|-- T1 --|-- T2 --|-- T3 --|-- T4 --|-- T5 --|---T6 --|-- T7 --|
#     |-- T8 --|-- T9 --|--------|--------|--------|--------|--------|-- T15--|
#     |--------|--------|--------|--------|--------|--------|--------|--------|
#     |--------|--------|--------|--------|--------|--------|--------|-- T31--|
#
# Phases, wavefronts:
# * Access 128B from 32 Banks is considered a wavefront.
# * Different wavefronts are processed in separate phases, cycles.
#   Assuming no bank conflict:
#   * Scalar load/store, a warp's requests are in a single phase/wavefront
#   * V2 load/store, 16 threads fit in one phase/wavefront
#   * V4 load/store, 8  threads fit in one phase/wavefront
#   * Only bank conflict within one phase  hurts performance.
#
# The loads are completed in four phases: T0-T7, T8-T15, T16-T23, T24-T31.
# T0 and T8 are mapped to same banks but their loads are in two separate phases.
# There is NO bank conflict.
#
# ==============================================================================
#
# TF32 MMA
#
# M x N x K = 16 x 8 x 8
#
# Thread map for the 16 x 8 matrix:
# | Ti |: a scalar mapped to thread Ti
#
#     | T0 | T1 | T2 | T3 || T0 | T1 | T2 | T3 |
#     | T4 | T5 | T6 | T7 || T4 | T5 | T6 | T7 |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | T31|| -- | -- | -- | T31|
#     +----------------------------------------+
#     | T0 | T1 | T2 | T3 || T0 | T1 | T2 | T3 |
#     | T4 | T5 | T6 | T7 || T4 | T5 | T6 | T7 |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | T31|| -- | -- | -- | T31|
#
#     mma(
#         d0, d1, d2, d3,
#         a0, a1, a2, a3,       # The above four scalars mapped to T0
#         b0, b1,
#         c0, c1, c2, c3,
#     )
#
# Q: How can we let T0 load its four scalars?
# Four scalar loads are inefficient
#
#
# ldmatrix: conceptually vector load + warp shuffle
#
#
#     lane_addr     8 x 4 matrix, each elment is 32 bits.
#
#       T0    ->    | T0 | T1 | T2 | T3 |
#       T1    ->    | T4 | T5 | T6 | T7 |
#       T2    ->    | -- | -- | -- | -- |
#       T3    ->    | -- | -- | -- | -- |
#       T4    ->    | -- | -- | -- | -- |
#       T5    ->    | -- | -- | -- | -- |
#       T6    ->    | -- | -- | -- | -- |
#       T7    ->    | -- | -- | -- | T31|
#
#     intruction:  `ld_matrix...x1... reg, Ti_address`
#
#     * T0-T7's lane address are used in *vector* loads. Rest addresses are ignored.
#     * T0-T7 does vector load.
#     * `reg` has the scalar value for Ti.
#
# What about the 16 x 8 matrix in tf32 mma?
#
#     intruction:  `ld_matrix...x4... reg0, reg1, reg2, reg3, Ti_address`
#
#     * All threads' lane address are used in *vector* loads.
#     * Output four registers.
#
#
# Where problem arises: how it works with our naive thread map?
#
# Recap. Naive Thread map for shared memory store:
# Each row covers 32 banks.
#
#     |-- T0 --|-- T1 --|-- T2 --|-- T3 --|-- T4 --|-- T5 --|---T6 --|-- T7 --|
#     |-- T8 --|-- T9 --|--------|--------|--------|--------|--------|-- T15--|
#     |--------|--------|--------|--------|--------|--------|--------|--------|
#     |--------|--------|--------|--------|--------|--------|--------|-- T31--|
#
# Assign lane address for ld_matrix. Consider the first 8x4 matrix in 16x8.
# T1's lane_addr points to the 2nd row of the BMxBK = 64x16 tile.
#
#     |-- T0 --|--------|--------|--------|-- T1 --|--------|--------|--------|
#     |-- T2 --|--------|--------|--------|-- T3 --|--------|--------|--------|
#     |-- T4 --|--------|--------|--------|-- T5 --|--------|--------|--------|
#     |-- T6 --|--------|--------|--------|-- T7 --|--------|--------|--------|
#
#                !!!!!!!! 4-way bank conflicts per phase !!!!!!!!
#
#
# ==============================================================================
#
# Thread Swizzling:
#
# https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
#
# https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/python/pycute/swizzle.py#L48-L59
#
# Swizzle[bits, base, shift](index):
#
#     A generic Swizzle functor
#     0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
#                                   ^--^  Base is the number of least-sig bits to keep constant
#                      ^-^       ^-^      Bits is the number of bits in the mask
#                        ^---------^      Shift is the distance to shift the YYY mask
#                                           (pos shifts YYY to the right, neg shifts YYY to the left)
#
#     e.g. Given Swizzle[2, 0, shift]
#     0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZ
#     the result is
#     0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAA where AA = ZZ xor YY
#
#
# xor refresh
#   [0|1] ^ 0 = [0|1], no change
#   [0|1] ^ 1 = [1|0], flip
#
#   A   ^ 1    : swap odd and even indices.
#   Ax  ^ 10   : swap two elments each time.
#   Axx ^ 100  : swap 2**2 elments each time.
#   ...
#
#
# Swizzle[2, 0, 3] for shared memory store:
#
#     lane_id: xxxxx
#              ^^    shift 3 bits and extract 2 bits mask
#
#  00xxx ^ 00  |-- T0 --|-- T1 --|-- T2 --|-- T3 --|-- T4 --|-- T5 --|-- T6 --|-- T7 --|
#  01xxx ^ 01  |-- T9 --|-- T8 --|-- T11--|-- T10--|-- T13--|-- T12--|-- T15--|-- T14--|
#  10xxx ^ 10  |-- T18--|-- T19--|-- T16--|-- T17--|-- T22--|-- T23--|-- T20--|-- T21--|
#  11xxx ^ 11  |-- T27--|-- T26--|-- T25--|-- T24--|-- T31--|-- T30--|-- T29--|-- T28--|
#


from .layout import LayoutTrait

# ===-----------------------------------------------------------------------===#
# Helpers                                                                      #
# ===-----------------------------------------------------------------------===#


fn shiftr(a: Int, s: Int) -> Int:
    return a >> s if s > 0 else shiftl(a, -s)


fn shiftl(a: Int, s: Int) -> Int:
    return a << s if s > 0 else shiftr(a, -s)


fn shiftr(a: Scalar, s: Scalar[a.type]) -> Scalar[a.type]:
    return a >> s if s > 0 else shiftl(a, -s)


fn shiftl(a: Scalar, s: Scalar[a.type]) -> Scalar[a.type]:
    return a << s if s > 0 else shiftr(a, -s)


# ===-----------------------------------------------------------------------===#
# Swizzle                                                                      #
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct Swizzle[bits: Int, base: Int, shift: Int](
    LayoutTrait, Stringable, Formattable
):
    alias yyy_mask = ((1 << bits) - 1) << (base + max(0, shift))
    alias zzz_mask = ((1 << bits) - 1) << (base - min(0, shift))

    @always_inline
    fn __init__(inout self):
        constrained[
            bits >= 0 and base >= 0, "Require non-negative mask bits and base"
        ]()
        constrained[
            abs(shift) >= bits, "Require shift greater than mask bits"
        ]()

    fn __call__(self, index: IntTuple) -> Int:
        # Disable this function for now because we can't prevent user from
        # passing in non-trivial tuples.
        constrained[
            False, "Swizzle don't support IntTuple as argument for now"
        ]()

        return self.__call__(index.value())

    @always_inline
    fn __call__(self, offset: Int) -> Int:
        return offset ^ shiftr(offset & self.yyy_mask, shift)

    @always_inline
    fn __call__(self, offset: Scalar) -> Scalar[offset.type]:
        return offset ^ shiftr(offset & self.yyy_mask, shift)

    @always_inline
    fn size(self) -> Int:
        return 1 << (bits + base + abs(shift))

    @always_inline
    fn cosize(self) -> Int:
        return self.size()

    @staticmethod
    @always_inline
    fn has_shape() -> Bool:
        return False

    fn format_to(self, inout writer: Formatter):
        write_to(writer, "(")
        write_to(writer, str(bits))
        write_to(writer, ",")
        write_to(writer, str(base))
        write_to(writer, ",")
        write_to(writer, str(shift))
        write_to(writer, ")")

    fn __str__(self) -> String:
        return String.format_sequence(self)


struct SwizzleEx(LayoutTrait, Stringable, Formattable):
    var bits: Int
    var base: Int
    var shift: Int
    var yyy_mask: Int
    var zzz_mask: Int

    @always_inline
    fn __init__(inout self, bits: Int, base: Int, shift: Int) raises:
        if bits < 0 or base < 0:
            raise Error("Require non-negative mask bits and base")

        if abs(shift) < bits:
            raise Error("Require shift greater than mask bits")

        self.bits = bits
        self.base = base
        self.shift = shift
        self.yyy_mask = ((1 << self.bits) - 1) << (
            self.base + max(0, self.shift)
        )
        self.zzz_mask = ((1 << self.bits) - 1) << (
            self.base - min(0, self.shift)
        )

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.bits = other.bits
        self.base = other.base
        self.shift = other.shift
        self.yyy_mask = other.yyy_mask
        self.zzz_mask = other.zzz_mask

    @always_inline
    fn __call__(self, index: IntTuple) -> Int:
        return self.__call__(index.value())

    @always_inline
    fn __call__(self, offset: Int) -> Int:
        return offset ^ shiftr(offset & self.yyy_mask, self.shift)

    @always_inline
    fn __call__(self, offset: Scalar) -> Scalar[offset.type]:
        return offset ^ shiftr(offset & self.yyy_mask, self.shift)

    @always_inline
    fn size(self) -> Int:
        return 1 << (self.bits + self.base + abs(self.shift))

    @always_inline
    fn cosize(self) -> Int:
        return self.size()

    @staticmethod
    @always_inline
    fn has_shape() -> Bool:
        return False

    fn format_to(self, inout writer: Formatter):
        write_to(writer, "(")
        write_to(writer, str(self.bits))
        write_to(writer, ",")
        write_to(writer, str(self.base))
        write_to(writer, ",")
        write_to(writer, str(self.shift))
        write_to(writer, ")")

    fn __str__(self) -> String:
        return String.format_sequence(self)


# ===-----------------------------------------------------------------------===#
# Composed Layout                                                              #
# ===-----------------------------------------------------------------------===#


struct ComposedLayout[
    LayoutA: LayoutTrait, LayoutB: LayoutTrait, offset_val: Int
](LayoutTrait):
    var layout_a: LayoutA
    var layout_b: LayoutB
    var offset: Int

    @always_inline
    fn __init__(inout self, layout_a: LayoutA, layout_b: LayoutB):
        constrained[offset_val >= 0, "Requires non-negative offset value"]()
        self.layout_a = layout_a
        self.layout_b = layout_b
        self.offset = offset_val

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.layout_a = other.layout_a
        self.layout_b = other.layout_b
        self.offset = other.offset

    @always_inline
    fn __call__(self, idx: IntTuple) -> Int:
        return self.layout_b(self.offset + self.layout_a(idx))

    @always_inline
    fn size(self) -> Int:
        return self.layout_a.size()

    @always_inline
    fn cosize(self) -> Int:
        return self.layout_b.cosize()

    @staticmethod
    @always_inline
    fn has_shape() -> Bool:
        return LayoutA.has_shape() or LayoutB.has_shape()
