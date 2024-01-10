# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -mojo-experimental-lifetimes %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias Lifetime = __mlir_type.`!lit.lifetime`

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample:
  fn __init__(inout self): pass
  fn __del__(owned self): pass
  fn noop(self): pass

# CHECK-LABEL: lit.func @"borrow{{.*}}"<
# CHECK-SAME: [[LT:.*_lt]][lt]: lifetime>(%a: !lit.ref<!MemExample, [[LT]]> borrow)
fn borrow[lt: Lifetime](a: ref[lt] MemExample):
  pass

# CHECK-LABEL: lit.func @"mutate{{.*}}"<
# CHECK-SAME: [[LT:.*_lt]][lt]: lifetime>(%a: !lit.ref<mut !MemExample, [[LT]]> borrow)
fn mutate[lt: Lifetime](a: mutref[lt] MemExample):
  pass

# CHECK-LABEL: lit.func @"implicit_borrow
fn implicit_borrow(a: MemExample):
  pass

# CHECK-LABEL: lit.func @"implicit_inout
fn implicit_inout(inout a: MemExample):
  pass

# CHECK-LABEL: lit.func @"implicit_owned
fn implicit_owned(owned a: MemExample):
  pass

# CHECK-LABEL: lit.func @"addrSpace
# CHECK: <{{.*}}lt[lt]: lifetime, {{.*}}_addr[addr]>(%a: !lit.ref<mut !MemExample, {{.*}}_lt, {{.*}}_addr> borrow)
fn addrSpace[lt: Lifetime, addr: __mlir_type.index](a: mutref[lt, addr] MemExample):
  pass

##===----------------------------------------------------------------------===##
# Conditional lifetimes
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"testConditional
fn testConditional(cond: __mlir_type.i1):
  # CHECK-NOT: __del__

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%a)
  let a = MemExample()
  # CHECK: lit.call @{{.*}}__del__{{.*}}(%a)

  # CHECK: lit.call @{{.*}}__init__{{.*}}(%b)
  let b = MemExample()
  # CHECK: lit.call @{{.*}}__del__{{.*}}(%b)

  let aref = __get_ref_from_value(a)
  let bref = __get_ref_from_value(b)

  # CHECK: %cref = lit.letreg.decl "cref"
  let cref = aref if cond else bref

  # TODO: __get_value_from_ref(cref).noop()
  # TODO: HECK: lit.call @{{.*}}noop
