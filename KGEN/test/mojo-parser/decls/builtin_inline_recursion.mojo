# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


struct a(TrivialRegisterPassable):
    @always_inline("builtin")
    def b(c, d: a):
        c & d

    @always_inline("builtin")
    def __rand__(c, e: a):
        # expected-error @below {{'@always_inline("builtin")' does not support recursion}}
        e & c


# // -----


struct S(TrivialRegisterPassable):
    @always_inline("builtin")
    def f(self, x: S):
        # expected-error @below {{'@always_inline("builtin")' does not support recursion}}
        self.f(x)


# // -----


struct S(TrivialRegisterPassable):
    @always_inline("builtin")
    def f(self, x: S):
        self.g(x)

    @always_inline("builtin")
    def g(self, x: S):
        # expected-error @below {{'@always_inline("builtin")' does not support recursion}}
        self.f(x)
