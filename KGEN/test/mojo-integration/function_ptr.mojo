# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Various integration tests for function types.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


def target(foo: Foo) -> Int:
    foo.closure_taking_a_foo(foo)
    return foo.thing


def bar(var s: Foo):
    print(s.thing)


struct Foo(ImplicitlyCopyable, ImplicitlyDeletable):
    var thing: Int
    var closure_taking_a_foo: def(var s: Foo) thin

    def __init__(out self, x: Int, f: def(var s: Foo) thin):
        self.thing = x
        self.closure_taking_a_foo = f


def main():
    # COM: runtime
    # CHECK: 4
    var foo_run = Foo(4, bar)
    _ = target(foo_run)

    # COM: comptime
    # CHECK: 2
    comptime foo = Foo(2, bar)
    comptime z = target(foo)
    var y = materialize[foo]()
    _ = target(y)
