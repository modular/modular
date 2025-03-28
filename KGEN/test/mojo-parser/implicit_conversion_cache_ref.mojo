# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s


struct FromType:
    var n: Int

    fn __init__(out self, n: Int):
        self.n = n


struct ToTypeImm:
    var n: Int

    @implicit
    fn __init__[
        O: ImmutableOrigin,
    ](out self, ref [O]f: FromType):
        self.n = f.n


struct ToTypeMut:
    var n: Int

    @implicit
    fn __init__[
        O: MutableOrigin,
    ](out self, ref [O]f: FromType):
        self.n = f.n


fn useToType(s: ToTypeImm):
    pass


fn useToType(s: ToTypeMut):
    pass


# Tests that implicit conversions that are ref-dependent are cached correctly.
# If the implicit conversion cache does not take into account the RefType, the
# second call to useToType will emit an error.
fn test[
    O: Origin[True], O2: Origin[False]
](ref [O]fImm: FromType, ref [O2]fMut: FromType):
    # CHECK: lit.call {{.*}}@ToTypeMut::@"__init__
    useToType(fImm)
    # CHECK: lit.call {{.*}}@ToTypeImm::@"__init__
    useToType(fMut)


fn main():
    f1 = FromType(1)
    f2 = FromType(2)
    test(f1, f2)
