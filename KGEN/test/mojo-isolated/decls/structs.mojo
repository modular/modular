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

# ===----------------------------------------------------------------------=== #
# Copy/Move synthesis tests
# ===----------------------------------------------------------------------=== #

struct IntPair(Copyable, Movable, ExplicitlyCopyable):
  var x: Int
  var y: Int

struct IntPairWrapper(Copyable, Movable, ExplicitlyCopyable):
  var value: IntPair


# CHECK-LABEL: lit.struct.decl @IntPairWrapper
# CHECK-LABEL: lit.fn @"copy
# CHECK-SAME: (%existing: !lit.ref<!IntPairWrapper{{.*}}> read_mem,
# CHECK-SAME: %__result__: !lit.ref<!IntPairWrapper{{.*}}> byref_result)
# CHECK-NEXT: lit.call {{.*}}IntPairWrapper::@"__copyinit__{{.*}}(%existing, %__result__)

# CHECK-LABEL: lit.fn @"testCopyMoveSynth
fn testCopyMoveSynth(owned a: IntPair, owned b: IntPairWrapper):
  # CHECK: lit.call {{.*}}IntPair::@"__copyinit__{{.*}}({{.*}}, %aCopy)
  var aCopy = a

  # CHECK: lit.call {{.*}}IntPair::@"__moveinit__{{.*}}({{.*}}, %aMove)
  var aMove = a^

  # CHECK: lit.call {{.*}}IntPair::@"copy{{.*}}({{.*}}, %aExCopy)
  var aExCopy = a.copy()

  # CHECK: lit.call {{.*}}IntPairWrapper::@"__copyinit__{{.*}}({{.*}}, %bCopy)
  var bCopy = b

  # CHECK: lit.call {{.*}}IntPairWrapper::@"__moveinit__{{.*}}({{.*}}, %bMove)
  var bMove = b^

  # CHECK: lit.call {{.*}}IntPairWrapper::@"copy{{.*}}({{.*}}, %bExCopy)
  var bExCopy = b.copy()

# ===----------------------------------------------------------------------=== #
# Fieldwise init tests
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct FieldwiseInitExample1[T: Movable]:
  var x: Int
  var y: T

# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample1
# CHECK: lit.fn @"__init__
# CHECK-SAME: (%x: !Int, %y: !lit.ref<:!Movable T, mut *"y`"> owned_in_mem,
# CHECK-SAME: %self: !lit.ref<{{.*}}> byref_result)
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[x]
# CHECK-NEXT: lit.ref.store %x, [[TMP]]
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.struct.ger %self[y]
# CHECK-NEXT: lit.call{{.*}}"__moveinit__"{{.*}}(%y, [[TMP]]) 
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none> 


# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample2
@fieldwise_init("implicit")
struct FieldwiseInitExample2:
  var x: Int
  
# CHECK-LABEL: lit.fn @"testFieldwiseInitExample2
# CHECK: FieldwiseInitExample2::@"__init__{{.*}}(%a, %b)
fn testFieldwiseInitExample2(a: Int):
  var b : FieldwiseInitExample2 = a

# Register passable example.
# CHECK-LABEL: lit.struct.decl @FieldwiseInitExample3
@fieldwise_init("implicit")
@register_passable
struct FieldwiseInitExample3:
  var x: Int

