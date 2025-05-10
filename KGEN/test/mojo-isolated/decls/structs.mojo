# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Support types
# ===----------------------------------------------------------------------=== #

@register_passable("trivial")
trait RPTTrait:
    pass

# ===----------------------------------------------------------------------=== #
# Destructor tests
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.struct.decl @DtorExample1
# Shouldn't have a registered destructor because it's trivial and not explicit.
# It does have a destructor though because of AnyType conformance.
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
@register_passable("trivial")
struct DtorExample1(AnyType):
  var a: Int

# CHECK-LABEL: lit.struct.decl @DtorExample2
# Shouldn't have a registered destructor because it's trivial and not explicit
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
@register_passable("trivial")
struct DtorExample2(AnyType):
  var a: Int

# CHECK-LABEL: lit.struct.decl @DtorExample3
# Should have a registered destructor because it's explicit.
# CHECK-NEXT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
@register_passable
struct DtorExample3(AnyType):
  var a: Int
 
  fn __del__(owned self):
    pass

# CHECK-LABEL: lit.struct.decl @DtorExample4
# Shouldn't have a registered destructor because it's trivial and not explicit
# CHECK-NOT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample4[T: RPTTrait]:
    var thing: T

# CHECK-LABEL: lit.struct.decl @DtorExample5
# Should have a registered destructor because T has a destructor.
# CHECK-NEXT: destructor :!lit.generator
# CHECK: lit.fn @"__del__
struct DtorExample5[T: AnyType]:
    var thing: T

