# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Test that '@parameter if' and '@parameter for' issue deprecation warnings.


@fieldwise_init
struct IterRange(ImplicitlyCopyable, Iterator):
    comptime Element = Int

    var value: Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.value <= 0:
            raise StopIteration()
        return self.value


def test_parameter_if[a: __mlir_type.`!kgen.scalar<bool>`]():
    # CHECK: warning: '@parameter if' is deprecated; use 'comptime if'
    @parameter
    if a:
        var inside: Int


def test_parameter_for[a: Int]():
    # CHECK: warning: '@parameter for' is deprecated; use 'comptime for'
    @parameter
    for i in IterRange(a):
        pass


# Test that 'comptime if' and 'comptime for' do NOT issue deprecation warnings.
# CHECK-NOT: '@parameter
def test_comptime_if[a: __mlir_type.`!kgen.scalar<bool>`]():
    comptime if a:
        var inside: Int


def test_comptime_for[a: Int]():
    comptime for i in IterRange(a):
        pass
