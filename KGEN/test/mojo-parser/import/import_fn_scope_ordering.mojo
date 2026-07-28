# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Function-scoped imports are order-sensitive, like other statements in a
# function body: the name is bound from the import statement onward. This is
# deliberately different from module scope, where a def above the import line
# can use the imported name (see import_fn_scope_binding.mojo).

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# Using the name before the import statement in the same function is an error.


def use_before_import():
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()
    from wildcard_shadow_a import shadowed_fn


# // -----

# An import in unreachable code after `return` does not bind earlier uses.


def unreachable_import() -> Int:
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    return shadowed_fn()
    from wildcard_shadow_a import shadowed_fn


# // -----

# A nested def cannot reference an import that appears later in the enclosing
# function (no Python-style late binding).


def outer():
    def inner():
        # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
        _ = shadowed_fn()

    from wildcard_shadow_a import shadowed_fn

    inner()
