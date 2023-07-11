# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file is imported by 'import.mojo' as part of testing import functionality,
# and does not include any useful testing by itself.

from SIMD import Float64


fn imported_fn():
    return


fn _ignored_wildcard_fn():
    return


@value
@register_passable("trivial")
struct MyStruct:
    var value: __mlir_type.index

    @always_inline("nodebug")
    fn __init__(value: Int) -> MyStruct:
        return Self {value: value.__as_mlir_index()}
