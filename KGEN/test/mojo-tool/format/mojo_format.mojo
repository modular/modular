# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# `mojo format` only works on `.mojo` files, and modifies them in place.
# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cp %s %t.mojo
# RUN: mojo format %t.mojo
# RUN: cat %t.mojo | grep -v "# CHECK:" | FileCheck %t.mojo

# CHECK: fn function() -> Int:
# CHECK: return 10
fn function()    -> Int:
    return     10


# CHECK: struct Foo(Copyable, Movable, Writable):
struct Foo(Movable, Writable, Copyable):
  pass


# CHECK: trait Bar(Copyable, Movable, Writable):
trait Bar(Writable, Movable, Copyable):
  pass

# CHECK: struct Bar[x: Int]:
struct Bar[x: Int]:
  pass

# CHECK: struct ComplexStruct[
# CHECK:     bar: Bar[5],
# CHECK:     index_type: DType = _get_index_type(layout, address_space),
# CHECK: ](Copyable, Movable, Writable):
# CHECK:     pass
struct ComplexStruct[
    bar: Bar[5], index_type: DType = _get_index_type(layout, address_space),
](Writable, Movable, Copyable):
    pass
