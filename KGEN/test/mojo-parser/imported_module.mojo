# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file is imported by 'import.mojo' as part of testing import functionality,
# and does not include any useful testing by itself.


fn imported_fn():
    return


fn _ignored_wildcard_fn():
    return


@value
@register_passable("trivial")
struct VeryUniqueStruct:
    var very_unique_field: __mlir_type.index

    # C-3PO is a short and very unique argument name. We use it to make
    # FileCheck matching easier.
    @staticmethod
    fn very_unique_func(`C-3PO`: Int) -> VeryUniqueStruct:
        return Self {very_unique_field: `C-3PO`.__mlir_index__()}
