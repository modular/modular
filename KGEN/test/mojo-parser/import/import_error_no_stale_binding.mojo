# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

# A structural import failure records its diagnostic keyed by the full dotted
# path, but must not bind that name in the importing scope. An
# escaped-identifier reference with the same spelling is an unrelated name: it
# must not silently resolve to the stale failure record.

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'bar'}}
import import_through_module.nested_package.module1.bar


def use_stale() -> Int:
    # expected-error @+1 {{use of unknown declaration 'import_through_module.nested_package.module1.bar'}}
    return `import_through_module.nested_package.module1.bar`()


# // -----

# Likewise, a real binding of the same spelling must not collide with the
# failure record.

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'bar'}}
import import_through_module.nested_package.module1.bar


def `import_through_module.nested_package.module1.bar`() -> Int:
    return 7


def use_real() -> Int:
    return `import_through_module.nested_package.module1.bar`()
