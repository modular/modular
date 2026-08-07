# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

# Lifting a closure demotes an origin parameter to immutable when nothing the
# closure captures uses that origin mutably.


@fieldwise_init
struct Inner(Copyable, Movable):
    var value: Int


struct Outer:
    var inner: Inner

    def apply(mut self):
        var p = Pointer(to=self.inner)

        # `self`'s origin must stay mutable in the closure's storage type: the
        # captured pointer writes through it. Only the closure's own capture of
        # the local `p` is immutable, per `{imm}`.
        # CHECK: lit.struct.decl @"Outer::apply{{.*}}::closure::__storage"<["self{{[^"]*}}"]*"self{{[^"]*}}": origin<true>
        @always_inline
        def closure() {imm}:
            p[].value = 5

        closure()
