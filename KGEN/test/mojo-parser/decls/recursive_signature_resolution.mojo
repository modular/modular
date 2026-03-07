# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Ensure that recursive signature resolution reached through implicit
# conversion during constraint checking produces a normal cycle diagnostic
# instead of crashing.

# CHECK: error: attempt to resolve a recursive reference to declaration 'MyInt.__init__'
# CHECK: note: referenced from here
# CHECK: note: referenced through this use
# CHECK: note: by declaration 'MyInt.__init__'

@fieldwise_init
struct Wrapper(ImplicitlyCopyable, RegisterPassable):
    var value: Int


struct MyInt(TrivialRegisterPassable):
    var value: Int

    fn __init__(out self):
        self.value = 0

    @implicit
    fn __init__(out self, arg: Wrapper)
        where orig():
        self.value = arg.value


fn take_myint(arg: MyInt):
    pass


fn orig() -> Bool
    where take_myint(Wrapper(0)):
    return True
