# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that struct members typed as SIMD[T, 1] (i.e. Scalar[T]) display
# correctly in the debugger.  This exercises the !kgen.simd<1, ...> fallback
# in unwrapToScalarOrPointer.


@fieldwise_init
struct ScalarMembers(TrivialRegisterPassable):
    var int_scalar: SIMD[DType.int, 1]
    var bool_scalar: SIMD[DType.bool, 1]
    var uint8_scalar: SIMD[DType.uint8, 1]

    def __init__(out self):
        self.int_scalar = SIMD[DType.int, 1](42)
        self.bool_scalar = SIMD[DType.bool, 1](True)
        self.uint8_scalar = SIMD[DType.uint8, 1](255)


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


def main():
    var s = ScalarMembers()

    print("breakpoint")  # breakpoint

    keep_alive(s)
