# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max, min, abs


fn shiftr(a: Int, s: Int) -> Int:
    return a >> s if s > 0 else shiftl(a, -s)


fn shiftl(a: Int, s: Int) -> Int:
    return a << s if s > 0 else shiftr(a, -s)


## A generic Swizzle functor
# 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
#                               ^--^  Base is the number of least-sig bits to keep constant
#                  ^-^       ^-^      Bits is the number of bits in the mask
#                    ^---------^      Shift is the distance to shift the YYY mask
#                                       (pos shifts YYY to the right, neg shifts YYY to the left)
#
# e.g. Given
# 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
# the result is
# 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
#


@register_passable("trivial")
struct Swizzle[bits: Int, base: Int, shift: Int]:
    var yyy_mask: Int
    var zzz_mask: Int

    fn __init__(inout self):
        constrained[
            bits >= 0 and base >= 0, "Require non-negative mask bits and base"
        ]()
        constrained[
            abs(shift) >= bits, "Require shift greater than mask bits"
        ]()

        var bit_msk = (1 << bits) - 1
        self.yyy_mask = bit_msk << (base + max(0, shift))
        self.zzz_mask = bit_msk << (base - min(0, shift))

    fn __call__(self, offset: Int) -> Int:
        return offset ^ shiftr(offset & self.yyy_mask, shift)

    fn size(self) -> Int:
        return 1 << (bits + base + abs(shift))

    fn cosize(self) -> Int:
        return self.size()

    fn __str__(self) -> String:
        return "(" + str(bits) + " " + str(base) + " " + str(shift) + ")"
