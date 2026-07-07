# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


trait Store:
    comptime StorageType[dtype: DType]: TrivialRegisterPassable

    @staticmethod
    def add[
        LT: ImplicitlyCopyable, //, dtype: DType
    ](storage: Tuple[Self.StorageType[dtype], LT]):
        ...


struct PS(Store):
    comptime StorageType[dtype: DType]: TrivialRegisterPassable = SIMD[dtype, 1]

    @staticmethod
    def add[
        LT: ImplicitlyCopyable, //, dtype: DType
    ](storage: Tuple[Self.StorageType[dtype], LT]):
        # CHECK: worked
        print("worked")


# Dispatch `add` through the generic `S: Store` bound.
def go[
    S: Store, dt: DType, LT: ImplicitlyCopyable
](p: S.StorageType[dt], layout: LT):
    # Here, in order to match the type between trait and witness table, we need
    # to be able to fold `conforms_to(SIMD[*(0, 1)], xxx)` to true. (Note that
    # SIMD[*(0, 1)] is not yet concretized).
    S.add((p, layout))


def main():
    go[PS](Float32(1.0), 0)
