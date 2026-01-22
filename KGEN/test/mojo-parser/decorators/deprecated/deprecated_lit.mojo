# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests for @deprecated decorator IR generation (LIT tests).
# For warning emission tests, see deprecated.mojo.

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


# ===----------------------------------------------------------------------=== #
# Test: Basic @deprecated IR generation
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @DeprecatedStruct
# CHECK-SAME: deprecationWarning = "struct"
@deprecated("struct")
struct DeprecatedStruct:
    pass


# CHECK-LABEL: lit.fn @"deprecated_func
# CHECK-SAME: deprecationWarning = "func"
@deprecated("func")
fn deprecated_func():
    pass


# CHECK-LABEL: lit.trait.decl @DeprecatedTrait
# CHECK-SAME: deprecationWarning = "trait"
@deprecated("trait")
trait DeprecatedTrait:
    pass


# CHECK-LABEL: lit.alias.decl *"deprecated_alias
# CHECK-SAME: deprecationWarning = "alias"
@deprecated("alias")
comptime deprecated_alias = 1


# ===----------------------------------------------------------------------=== #
# Test: @deprecated(use=...) syntax
# ===----------------------------------------------------------------------=== #


struct DeprecatedStructTarget:
    pass


# CHECK-LABEL: lit.struct.decl @DeprecatedStructUse
# CHECK-SAME: deprecationWarning = "'DeprecatedStructUse' is deprecated, use 'DeprecatedStructTarget' instead"
@deprecated(use=DeprecatedStructTarget)
struct DeprecatedStructUse:
    pass


fn deprecated_func_target():
    pass


# CHECK-LABEL: lit.fn @"deprecated_func_use
# CHECK-SAME: deprecationWarning = "'deprecated_func_use' is deprecated, use 'deprecated_func_target' instead"
@deprecated(use=deprecated_func_target)
fn deprecated_func_use():
    pass


trait DeprecatedTraitTarget:
    pass


# CHECK-LABEL: lit.trait.decl @DeprecatedTraitUse
# CHECK-SAME: deprecationWarning = "'DeprecatedTraitUse' is deprecated, use 'DeprecatedTraitTarget' instead"
@deprecated(use=DeprecatedTraitTarget)
trait DeprecatedTraitUse:
    pass


comptime deprecated_alias_target = 1


# CHECK-LABEL: lit.alias.decl *"deprecated_alias_use
# CHECK-SAME: deprecationWarning = "'deprecated_alias_use' is deprecated, use 'deprecated_alias_target' instead"
@deprecated(use=deprecated_alias_target)
comptime deprecated_alias_use = 1
