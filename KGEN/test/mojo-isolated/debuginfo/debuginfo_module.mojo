# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This file is imported by 'import-debuginfo.mojo' and does not include any
# tests itself.

# Don't move things around in this file, or else location info will break.
fn imported_fn():
    return

@fieldwise_init
@register_passable("trivial")
struct VeryUniqueStruct:
    var very_unique_field: __mlir_type.index

    # C-3PO is a short and very unique argument name. We use it to make
    # FileCheck matching easier.
    @staticmethod
    fn very_unique_func(`C-3PO`: __mlir_type.index) -> VeryUniqueStruct:
        return Self(`C-3PO`)
