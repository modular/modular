# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @__allow_legacy_any_origin_fields is recognized on struct fields
# and sets the allowLegacyAnyOrigin attribute in the IR.


# CHECK-LABEL: lit.struct.decl @WithLegacyField
struct WithLegacyField:
    # CHECK: lit.struct.field legacy_field {allowLegacyAnyOrigin}
    @__allow_legacy_any_origin_fields
    var legacy_field: Int

    # CHECK: lit.struct.field normal_field
    # CHECK-NOT: allowLegacyAnyOrigin
    var normal_field: Int
