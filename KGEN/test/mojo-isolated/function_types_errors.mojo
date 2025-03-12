# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct MemType:
    pass


fn mut_ship_function(mut x: MemType):
    ...


# We can convert from fn(read MemType)->None to fn(mut MemType)->None but not
# vice versa (see TTSMFS).
# expected-error @below {{cannot implicitly convert 'fn(mut x: MemType) -> None' value to 'fn(MemType) -> None' in alias initializer}}
alias read_ship_fn_alias: fn (read MemType) -> None = mut_ship_function
