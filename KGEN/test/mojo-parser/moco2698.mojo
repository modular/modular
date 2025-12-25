# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s

comptime float = __mlir_type.`!pop.scalar<f64>`

# expected-error @below {{'fn' statement must be on its own line}}
# expected-warning @below {{transfer from an owned value has no effect and can be removed}}
# expected-warning @below {{'float' value is unused}}
# expected-note @below {{'float' is aka '__mlir_type.`!pop.scalar<f64>`'}}
struct a: fn b(c, d : float) : d^
