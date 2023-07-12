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
struct VeryUniqueStruct:
    var very_unique_field: __mlir_type.index

    @staticmethod
    fn very_unique_func(very_unique_arg: Int) -> VeryUniqueStruct:
        return Self {very_unique_field: very_unique_arg.__as_mlir_index()}
