# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Int = __mlir_type.index

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


fn takes_pos_or_kw(i: Int, j: Int):
    pass


fn test_kw_operand_parsing(x: Int):
    takes_pos_or_kw(
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword argument 'j'}}
    )
    takes_pos_or_kw(
        j=x,
        x,  # expected-error {{positional argument follows keyword argument}}
    )
