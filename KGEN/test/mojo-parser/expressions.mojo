# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -verify-diagnostics -import-mojo %s | FileCheck %s

# CHECK: module {

fn noop(): pass

# CHECK-LABEL: lit.struct.decl @MemoryOnlyInt
struct MemoryOnlyInt:
  var x: Int

  # CHECK-LABEL: lit.func @"__init__
  fn __init__(inout self, a: Int = 42):
    # CHECK: %0 = lit.ref.struct.ger %self[x]
    # CHECK: %1 = {{.*}}constant: !Int = <{1}>
    # CHECK: lit.ref.store %1, %0
    self.x = 1
  fn __del__(owned self): pass

  # CHECK-LABEL: lit.func @"__copyinit__
  fn __copyinit__(inout self, other: Self):
    self.x = other.x

  @staticmethod
  fn variadic(*value: MemoryOnlyInt):
    pass

fn consume(owned a: MemoryOnlyInt): pass

# This type is used to test implicit conversion from MemoryOnlyInt
struct MemoryOnlyFloat64:
  var x: Float64
  fn __init__(inout self, value: MemoryOnlyInt):
    self.x = 1.0

# CHECK-LABEL: lit.struct.decl @MemoryOnlyPair
struct MemoryOnlyPair:
  var x: MemoryOnlyInt
  var y: Int

  # CHECK: lit.func @"__copyinit__{{.*}}(%self: !lit.ref<!MemoryOnlyPair, mut {{.*}}> init_self,
  # CHECK-SAME: %other: !lit.ref<!MemoryOnlyPair, imm {{.*}}> borrow_in_mem)
  fn __copyinit__(inout self, other: MemoryOnlyPair):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %other[x]
    # CHECK-NEXT: %1 = lit.ref.struct.ger %self[x]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%1, %0)
    # CHECK-NEXT: %3 = lit.ref.struct.ger %other[y]
    # CHECK-NEXT: %4 = lit.ref.struct.ger %self[y]
    # CHECK-NEXT: %5 = lit.ref.load %3
    # CHECK-NEXT: lit.ref.store %5, %4
    self.x = other.x
    self.y = other.y

  # CHECK: lit.func @"method{{.*}}(
  # CHECK-SAME: %self: !lit.ref<!MemoryOnlyPair, mut {{.*}}> owned_in_mem,
  # CHECK-SAME: %arg: !lit.ref<!MemoryOnlyInt, mut {{.*}}> owned_in_mem)
  fn method(owned self, owned arg: MemoryOnlyInt):
    # CHECK: %0 = lit.ref.struct.ger %self[y]
    # CHECK: %1 = lit.ref.struct.ger %arg[x]
    # CHECK: %2 = lit.ref.load %0
    # CHECK: %3 = lit.ref.load %1
    # CHECK: %4 = lit.call @{{.*}}__add__{{.*}}"(%2, %3)
    _ = self.y+arg.x

fn inferred_function_with_memory_result[
  width: Int](x: SIMD[DType.float32, width]) -> MemoryOnlyInt: pass

# CHECK-LABEL: lit.func @"memoryOnlyOps
fn memoryOnlyOps(inout a: MemoryOnlyPair) -> MemoryOnlyPair:
  # CHECK-NEXT: %v1 = lit.var.decl {{.*}} var : !lit.ref<!MemoryOnlyPair,
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%v1, [[IMMREF]])
  var v1 = a

  # CHECK-NEXT: %v2 = lit.var.decl "v2"
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%v2, [[IMMREF]])
  var v2 : MemoryOnlyPair = a

  # CHECK-NEXT: %anonymous2A = lit.var.decl {{.*}} synth
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, [[IMMREF]])
  _ = a

  a  # expected-warning {{'MemoryOnlyPair' value is unused}}

  # CHECK-NEXT: %regX = lit.var.decl {{.*}}
  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[AX]]
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%regX, [[IMMREF]])
  var regX = a.x

  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[AX]], [[IMMREF]])
  a.x = regX

  # Pass memory only things by value as arguments.

  # CHECK-NEXT: [[TMPPAIR:%.*]] = lit.var.decl {{.*}}!MemoryOnlyPair
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[TMPPAIR]], [[IMMREF]])
  # CHECK-NEXT: [[TMPINT:%.*]] = lit.var.decl {{.*}}!MemoryOnlyInt
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[TMPINT]], [[IMMREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"method{{.*}}([[TMPPAIR]], [[TMPINT]])
  a.method(regX)

  # Drill into rvalue without cloning intermediate values.
  # CHECK-NEXT: %v2xx = lit.var.decl "v2xx"
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[V2XX:%.*]] = lit.ref.struct.ger [[V2X]][x]
  # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load [[V2XX]]
  # CHECK-NEXT: lit.ref.store [[VAL]], %v2xx
  var v2xx = v2.x.x

  # Implicit conversion between memory-only types.
  # CHECK-NEXT: %mpFloat = lit.var.decl
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[V2X]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mpFloat, [[IMMREF]])
  var mpFloat : MemoryOnlyFloat64 = v2.x

  # CHECK: [[SIMDTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.call {{.*}}SIMD::@"__init__{{.*}}([[SIMDTMP]])
  # CHECK-NEXT: [[SIMDVAL:%.*]] = lit.ref.load [[SIMDTMP]]

  # CHECK: [[TMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.call @{{.*}}inferred_function_with_memory_result{{.*}}([[SIMDVAL]], [[TMP]])
  _ = inferred_function_with_memory_result(SIMD[DType.float32, 4]())

  # Memory-only default argument with memory-only result.
  # CHECK-NEXT: %[[C42:.*]] = {{.*}}constant: !Int = <{42}>
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}([[TMP]], %[[C42]])
  _ = MemoryOnlyInt()

  # CHECK-NEXT: [[IMMREF1:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[IMMREF2:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.create [[[IMMREF1]], [[IMMREF2]]]
  # CHECK-NEXT: lit.call @{{.*}}variadic{{.*}}([[VARIADIC]])
  MemoryOnlyInt.variadic(regX, regX)

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %v2
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%__result__, [[IMMREF]])
  # CHECK-NEXT: [[NONEVAL:%.*]] = kgen.param.constant: none = <#kgen.none>
  # CHECK-NEXT: lit.return [[NONEVAL]]
  return v2

struct DirectInit:
  fn __init__(inout self):
    pass

fn direct_call_init():
  var value: DirectInit
  # COM: Make sure this doesn't warn about an unused result.
  value.__init__()

struct DummyFunc:
    fn __init__(inout self, f: def(Int)):
        pass

fn func_arg_conversion(f: DummyFunc): pass

# CHECK-LABEL: lit.func @"implicit_func_conversion()"
fn implicit_func_conversion():
    def take_int(x: Int):
        pass

    # CHECK: %f = lit.var.decl "f"
    # CHECK: [[CLOSURE:%.*]] = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}(%f, [[CLOSURE]])
    var f: DummyFunc = take_int
    # CHECK: [[CLOSURE:%.*]] = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}(%anonymous2A, [[CLOSURE]])
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A
    # CHECK: call {{.*}}func_arg_conversion{{.*}}([[IMMREF]])
    func_arg_conversion(take_int)

# CHECK-LABEL: lit.struct.decl @RegPassable
@register_passable
struct RegPassable:
  var value: Int
  # CHECK-LABEL: lit.func @"__init__
  fn __init__(inout self, value: Int):
    self.value = value

  fn __copyinit__(inout self, existing: Self): pass
  fn __del__(owned self): pass
  fn __neg__(self) -> Self: pass
  fn __add__(self, rhs: Self) -> Self: pass
  fn __matmul__(self, rhs: Self) -> Self: pass
  fn __rmatmul__(lhs, self: Self) -> Self: pass

# CHECK-LABEL: lit.struct.decl @StructWithFuncParam<comparator: !lit.signature
# CHECK-SAME: <"T": type>(!kgen.paramref<*(0,0)>, |)
struct StructWithFuncParam[comparator: fn[T: AnyTrivialRegType] (T) -> None]:
    # CHECK-LABEL: lit.func @"f
    # CHECK-SAME: %self: !lit.ref<{{.*}}<:!lit.signature<<"T": type>(!kgen.paramref<*(0,0)>
    fn f(self):
        pass

    # CHECK-LABEL: lit.func @"g
    fn g(self):
        # CHECK: call {{.*}}[imm *"self`2x"]<:!lit.signature<<"T": type>(!kgen.paramref<*(0,0)>, |)
        # CHECK-SAME: !lit.ref<{{.*}}<"T": type>(!kgen.paramref<*(0,0)>, |)
        self.f()

# CHECK-LABEL: lit.func @"simpleMath
fn simpleMath(a: Int, b: Int) -> Int:
  # CHECK: %0 = lit.call {{.*}}Int::@"__mul__{{.*}}(%b, %a)
  # CHECK: %1 = lit.call {{.*}}Int::@"__sub__{{.*}}(%a, %0)
  # CHECK: lit.return %1 : !Int
  return a-b*a

# CHECK-LABEL: lit.func @"precedence_associativity
fn precedence_associativity(a: Int):
  # CHECK: %z = lit.var.decl "z" var
  var z: Int = 0

  # CHECK: [[SEVENTEENINT:%.*]] = kgen{{.*}}{17}
  # CHECK-NEXT: lit.ref.store [[SEVENTEENINT]], %z
  z = 17  # Implicit conversion

  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT: %[[POW0:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}{2}
  # CHECK-NEXT: %[[POW1:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0]])
  # CHECK-NEXT: lit.ref.store %[[POW1]], %z
  z = 2**(a**z)
  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT: %[[POW0:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}{2}
  # CHECK-NEXT: %[[POW1:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0]])
  # CHECK-NEXT: lit.ref.store %[[POW1]], %z
  z = 2**a**z
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[MUL:.*]] = kgen.param.constant: !Int = <{-6}
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[Z]], %[[MUL]])
  # CHECK-NEXT:  lit.ref.store %[[ADD]], %z
  z = z + 3 * -2
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[FLOOR_DIV:.*]] = kgen.param.constant: !Int = <{-2}>
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[Z]], %[[FLOOR_DIV]])
  # CHECK-NEXT:  lit.ref.store %[[ADD]], %z
  z = z + 3 // -2
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[INT_THREE:.*]] = kgen{{.*}}{3}
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[Z]], %[[INT_THREE]])
  # CHECK-NEXT:  %[[NEG:.*]] = kgen{{.*}}{-2}
  # CHECK-NEXT:  %[[MUL:.*]] =  lit.call {{.*}}Int::@"__mul__{{.*}}(%[[ADD]], %[[NEG]])
  # CHECK-NEXT:  lit.ref.store %[[MUL]], %z
  z = (z + 3) * -+2
  # CHECK-NEXT:  %[[INT_TWO:.*]] = kgen{{.*}}{2}
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[POW:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[Z]])
  # CHECK-NEXT:  %[[NEG:.*]] = lit.call {{.*}}Int::@"__neg__{{.*}}(%[[POW]])
  # CHECK-NEXT:  lit.ref.store %[[NEG]], %z
  z = -2**z
  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT: %[[ONE:.*]] = kgen{{.*}}{1}
  # CHECK-NEXT: %[[RES:.*]] = lit.call {{.*}}Int::@"__radd__(::Int,::Int)"(%[[Z]], %[[ONE]])
  # CHECK-NEXT: lit.ref.store %[[RES]], %z
  z = Int(1).value + z

  # div tests
  # CHECK: lit.call {{.*}}__truediv__
  var r0 = Float32(33.0) / Float32(42.0)

  # CHECK: lit.call {{.*}}__truediv__
  var r1 = Float32(33.0) / 42.0

  # COM: test if-else operator associativity
  # CHECK: %[[C:.*]] = lit.ref.load %c
  # CHECK-NEXT: %[[TEN:.*]] = kgen.param.constant: !Int = <{10}>
  # CHECK-NEXT: %[[EQ:.*]] = lit.call {{.*}}__eq__{{.*}}(%[[C]], %[[TEN]])
  # CHECK-NEXT: %[[EQI1:.*]] = lit.call {{.*}}__mlir_i1__{{.*}}%[[EQ]]
  # CHECK-NEXT: %[[RESULT:.*]] = hlcf.if %[[EQI1]] -> !Int {
  # CHECK-NEXT:   %[[ZERO:.*]] = kgen.param.constant: !Int = <{0}>
  # CHECK-NEXT:   hlcf.yield %[[ZERO]] : !Int
  # CHECK-NEXT: } else {
  # CHECK-NEXT:  %[[C:.*]] = lit.ref.load %c
  # CHECK-NEXT:  %[[ELEVEN:.*]] = kgen.param.constant: !Int = <{11}>
  # CHECK-NEXT:  %[[EQ:.*]] = lit.call {{.*}}__eq__{{.*}}(%[[C]], %[[ELEVEN]])
  # CHECK-NEXT:  %[[EQI1:.*]] = lit.call {{.*}}__mlir_i1__{{.*}}(%[[EQ]])
  # CHECK-NEXT:  %[[RIGHT_IF_RESULT:.*]] = hlcf.if %[[EQI1]] -> !Int {
  # CHECK-NEXT:    %[[ONE:.*]] = kgen.param.constant: !Int = <{1}>
  # CHECK-NEXT:    hlcf.yield %[[ONE]] : !Int
  # CHECK-NEXT:  } else {
  # CHECK-NEXT:    %[[TWO:.*]] = kgen.param.constant: !Int = <{2}>
  # CHECK-NEXT:    hlcf.yield %[[TWO]] : !Int
  # CHECK-NEXT:  }
  # CHECK-NEXT:  hlcf.yield %[[RIGHT_IF_RESULT]] : !Int
  # CHECK-NEXT:}
  var c = 10
  z = 0 if c == 10 else 1 if c == 11 else 2

# CHECK-LABEL: lit.func @"reverse_operators
fn reverse_operators(a: Int):
  # CHECK: lit.call {{.*}}Int::@"__radd__(::Int,::Int)"
  var z = Int(1).value + a

  # CHECK: lit.call {{.*}}Int::@"__rsub__(::Int,::Int)"
  z = Int(2).value - z

  # CHECK: lit.call {{.*}}Int::@"__rmul__(::Int,::Int)"
  z = Int(3).value * z

  # div tests
  # CHECK: lit.call {{.*}}__rtruediv__
  # CHECK: lit.call {{.*}}Int::@"__rfloordiv__(::Int,::Int)"
  var r1 = 33.0 / Float32(42.0)
  z = Int(33).value // z

  # CHECK: lit.call {{.*}}Int::@"__rmod__(::Int,::Int)"
  var i0 = Int(10).value % z

# CHECK: lit.call {{.*}}Int::@"__rpow__(::Int,::Int)"
  var i1 = Int(3).value ** z

  # CHECK: lit.call {{.*}}Int::@"__rlshift__(::Int,::Int)"
  var i2 = Int(1).value << z

  # CHECK: lit.call {{.*}}Int::@"__rrshift__(::Int,::Int)"
  var i3 = Int(1).value >> z

  # CHECK: lit.call {{.*}}Int::@"__rand__(::Int,::Int)"
  z = Int(1).value & z

  # CHECK: lit.call {{.*}}Int::@"__ror__(::Int,::Int)"
  z = Int(2).value | z

  # CHECK: lit.call {{.*}}Int::@"__rxor__(::Int,::Int)"
  z = Int(3).value ^ z

# CHECK-LABEL: lit.func @"precedence_matmul
fn precedence_matmul(z: RegPassable) -> RegPassable:
  # CHECK:  [[THREE:%.*]] = kgen.param.constant: !Int = <{3}>
  # CHECK-NEXT: [[THREETMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.call {{.*}}@RegPassable::@"__init__{{.*}}([[THREETMP]], [[THREE]])
  # CHECK-NEXT:  [[TWO:%.*]] = kgen.param.constant: !Int = <{2}>
  # CHECK-NEXT:  [[TWOTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.call {{.*}}@RegPassable::@"__init__{{.*}}([[TWOTMP]], [[TWO]])
  # CHECK-NEXT:  [[INT_TWO:%.*]] = lit.ref.load [[TWOTMP]]
  # CHECK-NEXT:  [[NEG:%.*]] = lit.call {{.*}}@RegPassable::@"__neg__{{.*}}([[INT_TWO]])
  # CHECK-NEXT:  [[INT_THREE:%.*]] = lit.ref.load [[THREETMP]]
  # CHECK-NEXT:  [[MATMUL:%.*]] = lit.call {{.*}}@RegPassable::@"__matmul__{{.*}}([[INT_THREE]], [[NEG]])
  # CHECK-NEXT:  [[ADD:%.*]] = lit.call {{.*}}@RegPassable::@"__add__{{.*}}(%z, [[MATMUL]])
  # CHECK-NEXT:  lit.return [[ADD]] : !RegPassable
  return z + RegPassable(3) @ -RegPassable(2)

# CHECK-LABEL: lit.func @"precedence_bitwise
fn precedence_bitwise(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}{2}
  # CHECK-NEXT: %[[MUL:.*]] = lit.call {{.*}}Int::@"__mul__{{.*}}(%a, %[[INT_TWO]])
  # CHECK-NEXT: %[[AND:.*]] = lit.call {{.*}}Int::@"__and__{{.*}}(%[[MUL]], %b)
  # CHECK-NEXT: %[[INT_FOUR:.*]] = kgen{{.*}}{4}
  # CHECK-NEXT: %[[XOR:.*]] = lit.call {{.*}}Int::@"__xor__{{.*}}(%[[INT_FOUR]], %c)
  # CHECK-NEXT: %[[OR:.*]] = lit.call {{.*}}Int::@"__or__{{.*}}(%[[AND]], %[[XOR]])
  # CHECK-NEXT: lit.return %[[OR]]
  return a * 2 & b | 4 ^ c

# CHECK-LABEL: @"comparisons
fn comparisons(a: Int, b: Int):
   var res: Bool
   # CHECK: lit.call {{.*}}Int::@"__lt__{{.*}}(%a, %b)
   res = a < b
   # CHECK: lit.call {{.*}}Int::@"__le__{{.*}}(%a, %b)
   res = a <= b
   # CHECK: lit.call {{.*}}Int::@"__gt__{{.*}}(%a, %b)
   res = a > b
   # CHECK: lit.call {{.*}}Int::@"__ge__{{.*}}(%a, %b)
   res = a >= b
   # CHECK: lit.call {{.*}}Int::@"__eq__{{.*}}(%a, %b)
   res = a == b
   # CHECK: lit.call {{.*}}Int::@"__ne__{{.*}}(%a, %b)
   res = a != b

@register_passable
struct Boolish:
  fn __copyinit__(inout self, existing: Self): pass
  fn __bool__(self) -> Bool: return True

struct MemBoolish:
  fn __init__(inout self, value: Boolish): pass
  fn __copyinit__(inout self, other: Self): pass
  fn __bool__(self) -> Bool: return True

# CHECK-LABEL: @"unary
fn unary(a: Bool, b: Int, c: Boolish, d: MemBoolish):
  # CHECK: %0 = lit.call {{.*}}Bool::@"__bool__({{.*}}Bool)"(%a)
  # CHECK: %1 = lit.call {{.*}}Bool::@"__invert__({{.*}}Bool)"(%0)
  _ = not a

  # CHECK: [[EQ:%.*]] = lit.call {{.*}}Int::@"__eq__(::Int,::Int)"
  # CHECK: [[EQBOOL:%.*]] = lit.call {{.*}}Bool::@"__bool__({{.*}}Bool)"([[EQ]])
  # CHECK:  = lit.call {{.*}}Bool::@"__invert__({{.*}}Bool)"([[EQBOOL]])
  _ = not b == 0

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%c)
  # CHECK:  = lit.call {{.*}}Bool::@"__invert__({{.*}}Bool)"([[BOOL]])
  _ = not c

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}@"__bool__{{.*}}(%d)
  # CHECK-NEXT: lit.call {{.*}}__invert__{{.*}}([[BOOL]])
  _ = not d

# CHECK-LABEL: lit.func @"andOr
fn andOr(a: Boolish, b: Boolish, c: Bool, d: MemBoolish):
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BOOL]])
  # CHECK: hlcf.if [[I1]] -> !Boolish {
  # CHECK:   = lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %b)
  # CHECK:   hlcf.yield
  # CHECK: } else {
  # CHECK:   [[ANON:%.*]] = lit.var.decl
  # CHECK:   [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[ANON]], %a)
  # CHECK:   [[A:%.*]] = lit.load.consume [[ANON]]
  # CHECK:   hlcf.yield [[A]]
  # CHECK: }
  _ = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.  Boolish is defined with copy ctor so it must be invoked.

  # CHECK-NEXT: [[ABOOL:%.*]] = lit.call {{.*}}Boolish::@"__bool__{{.*}}"(
  # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !Boolish {
  # CHECK:        = lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %a)
  # CHECK:        hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK:        lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %b)
  # CHECK:        hlcf.yield
  # CHECK-NEXT: }
  _ = a or b

  # Testing two different logic'y types returns the common bool type if present.

  # CHECK-NEXT: [[ABOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !Bool {
  # CHECK-NEXT:   hlcf.yield %c
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   %anonymous2A_0 = lit.var.decl "anonymous*"
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}(%anonymous2A_0, [[I1]])
  # CHECK-NEXT:   [[TMP:%.*]] = lit.load.consume %anonymous2A_0
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a and c

  # Check incompatible types that are nevertheless boolish.

  # CHECK-NEXT: [[BBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%b)
  # CHECK-NEXT: [[BI1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BBOOL]])
  # CHECK-NEXT: = hlcf.if [[BI1]] -> !Bool {
  # CHECK-NEXT:   %anonymous2A_0 = lit.var.decl "anonymous*"
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}(%anonymous2A_0, [[BI1]])
  # CHECK-NEXT:   [[TMP:%.*]] = lit.load.consume %anonymous2A_0
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK: } else {
  # CHECK-NEXT: hlcf.yield %c : !Bool
  # CHECK-NEXT: }
  _ = b or c

  # Check memory-only boolish types.
  # Boolish and MemBoolish has a common type of MemBoolish.

  # CHECK-NEXT: [[DBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%d)
  # CHECK-NEXT: [[DI1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[DBOOL]])
  # CHECK-NEXT: [[IFRESULT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!MemBoolish
  # CHECK-NEXT: hlcf.if [[DI1]] {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %d)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMPMEM:%.*]] = lit.var.decl
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}([[TMPMEM]], %b)
  # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut [[TMPMEM]]
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, [[IMMREF]])
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  _ = d or b

# CHECK-LABEL: lit.func @"paramAndOr{{.*}}"<a: !Boolish, b: !Boolish>
fn paramAndOr[a: Boolish, b: Boolish]():
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: lit.alias.decl *"c{{.*}}": !Boolish = <cond(apply({{.*}}@Bool::@"__mlir_i1__{{.*}}", apply({{.*}}Boolish::@"__bool__{{.*}}", a)), b, a)>
  alias c = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.

  # CHECK: lit.alias.decl *"d{{.*}}": !Boolish = <cond(apply({{.*}}@Bool::@"__mlir_i1__{{.*}}", apply({{.*}}Boolish::@"__bool__{{.*}}", a)), a, b)>
  alias d = a or b

# CHECK-LABEL: lit.func @"do_math
fn do_math(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %z = lit.var.decl "z" var
  var z : Int
  # CHECK-NEXT: %[[INT_5:.*]] = kgen{{.*}}{5}
  # CHECK-NEXT: %[[MUL:.*]] = lit.call {{.*}}Int::@"__mul__{{.*}}(%[[INT_5]], %a)
  # CHECK-NEXT: %[[INT_42:.*]] = kgen{{.*}}{42}
  # CHECK-NEXT: %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[INT_42]], %[[MUL]])
  # CHECK-NEXT: lit.ref.store %[[ADD]], %z
  z = 42 + 5*a

  # CHECK-NEXT: %x = lit.var.decl "x" var
  # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %z
  # CHECK-NEXT: lit.ref.store [[TMP]], %x
  # This is checking the lexer handles \ at end of line correctly.
  var x : Int
  x = \
z

  # CHECK-NEXT: lit.call @{{.*}}noop()"()
  noop()

  # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %x
  # CHECK-NEXT: lit.return [[TMP]]
  return x

# CHECK-LABEL: lit.func @"listValues()"
fn listValues():
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__{{.*}}(%a
  var a = [1, 2, 2+1]
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__{{.*}}(%a
  a = [1, 2, 2+1,]
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__{{.*}}(%a
  a = [1, 2, 2+1]
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__{{.*}}(%b
  var b = []

# CHECK-LABEL: lit.func @"initializers
fn initializers():
  # CHECK-NEXT: %a = lit.var.decl "a"
  # CHECK: %0 = kgen.param.constant: !Int = <{42}>
  # CHECK-NEXT: lit.ref.store %0, %a
  var a = Int{value: Int(42).value}

  # Issue #7343: Trailing comma ok too.
  _ = Int{value: Int(42).value,}

  # Issue #12067, suffix stuff ok.
  _ = Int{ value: Int(1).value }.value

# CHECK-LABEL: lit.func @"test_if_cond
fn test_if_cond(owned cond: Bool, memCond: MemBoolish):
    # CHECK: lit.ref.store %cond, %cond_0
    # CHECK: %i = lit.var.decl "i"
    # CHECK: %[[COND:.*]] = lit.ref.load %cond_0
    # CHECK: %[[LIT_BOOLI1:.*]] = lit.call {{.*}}__mlir_i1__{{.*}}(%[[COND]])
    # CHECK-NEXT: %[[IF_RES:.*]] = hlcf.if %[[LIT_BOOLI1]]
    # CHECK-NEXT:   %[[INT_TWO:.*]] = kgen{{.*}}{2}
    # CHECK-NEXT:   hlcf.yield %[[INT_TWO]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %[[INT_THREE:.*]] = kgen{{.*}}{3}
    # CHECK-NEXT:   hlcf.yield %[[INT_THREE]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF_RES]], %i
    var i: Int = 2 if cond else 3

    # CHECK: [[TRUEB:%.+]] = kgen{{.*}}{:i1 1}
    # CHECK-NEXT: lit.ref.store [[TRUEB]], %cond_0
    cond = True
    i += i
    if cond:     # 'if' stmt, not an 'if' expression.
        i += 1

# CHECK-LABEL: lit.func @"test_param_if_cond{{.*}}"<cond: !Bool>
fn test_param_if_cond[cond: Bool]() -> Int:
  # CHECK-NEXT: lit.alias.decl [[I_ALIAS:.*]]: !IntLiteral = <cond(apply({{.*}}Bool::@"__mlir_i1__{{.*}}", cond), {:!kgen.int_literal 2}, {:!kgen.int_literal 3})>
  alias i = 2 if cond else 3

  # CHECK-NEXT: lit.alias.decl *"j{{.*}}": !FloatLiteral = <cond(apply({{.*}}Bool::@"__mlir_i1__{{.*}}", cond), {:!kgen.float_literal #kgen.float_literal<2|1>}, {:!kgen.float_literal #kgen.float_literal<3|1>})>
  alias j = 2.0 if cond else 3

  # CHECK-NEXT: %[[I:.*]] = kgen.param.constant: !Int = {{.*}}IntLiteral{{.*}}[[I_ALIAS]]{{.*}}
  return i

# CHECK-LABEL: lit.func @"callable_mv[fn(::Int, /) -> ::Int](::Int)"
# CHECK-SAME: <callable: !lit.signature<(!Int, |) -> !Int>>(%a: !Int) -> !Int
fn callable_mv[callable: fn (Int) -> Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call[!lit.signature<(!Int, |) -> !Int>: callable](%a)
  return callable(a)

# CHECK-LABEL: lit.func @"callable_mv_inputs{{.*}})"<
# CHECK-SAME: callable: !lit.signature<<"x": !Int>(!Int, |) -> !Int>, b: !Int>(%a: !Int) -> !Int
fn callable_mv_inputs[callable: fn[x: Int](Int) -> Int, b: Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call[!lit.signature<(!Int, |) -> !Int>: bind_signature({{.*}}callable, b)](%a)
  return callable[b](a)

# CHECK-LABEL: lit.func @"takeIndexParam{{.*}}"<a: !Int>() -> !Int
fn takeIndexParam[a: Int]() -> Int:
  return a + 1

# CHECK-LABEL: lit.func @"returnIndex()"() -> !Int
fn returnIndex() -> Int:
  return 0

# CHECK-LABEL: lit.func @"returnIndex2()"() -> !Int
fn returnIndex2() -> Int:
  # CHECK-NEXT: %0 = lit.call @{{.*}}takeIndexParam{{.*}}"<:!Int apply({{.*}}@{{.*}}returnIndex()")>()
  # CHECK-NEXT: return %0
  return takeIndexParam[returnIndex()]()

# CHECK-LABEL: lit.func @"callInParam[fn[::Int](::Int, /) -> ::Int]()"
# CHECK-SAME: <callable: !lit.signature<<"x": !Int>(!Int, |) -> !Int>>() -> !Int
fn callInParam[callable: fn[x: Int](Int) -> Int]() -> Int:
  # CHECK-NEXT: %0 = lit.call @{{.*}}takeIndexParam{{.*}}()"<:!Int apply({{.*}}bind_signature({{.*}}callable, {1}), {1})>()
  # CHECK-NEXT: return %0
  return takeIndexParam[callable[1](1)]()

# CHECK-LABEL: lit.func @"parameterExprs{{.*}}()"
# CHECK-SAME: <a: !Int, a2: !Int>
fn parameterExprs[a: Int, a2: Int]():
  # CHECK: lit.alias.decl *"b{{.*}}": !Int = <apply({{.*}}__sub__{{.*}}, a, a)>
  alias b = a-a
  # CHECK: lit.alias.decl *"c{{.*}}": !Int = <apply({{.*}}__add__{{.*}}, a, {{.*}}42{{.*}})>
  alias c = a+42
  # CHECK: lit.alias.decl *"d{{.*}}": !Int = <apply({{.*}}__mul__{{.*}}, a, a2)>
  alias d = a*a2

##===----------------------------------------------------------------------===##
# Patterns, LValues and RValues
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"patterns()
fn patterns():
  # CHECK: %z2 = lit.var.decl "z2" var
  var z2: Int

  (((z2))) = 42  # Paren patterns
  # CHECK: [[TMP:%.*]] = {{.*}}constant: !Int = <{42}>
  # CHECK: lit.ref.store [[TMP]], %z2

  var someInt : Int
  (someInt) += someInt
  # CHECK: %someInt = lit.var.decl "someInt" var
  # CHECK:  %1 = lit.ref.load %someInt
  # CHECK:   = lit.call {{.*}}Int::@"__iadd__{{.*}}(%someInt, %1)

  # Discard pattern with different types.
  (_) = someInt
  # CHECK: [[TMP:%.*]] = lit.ref.load %someInt

  (_) = 1.0

  # CHECK: %someFloat32 = lit.var.decl "someFloat32" var
  # CHECK: [[Float32:%.*]] = lit.ref.load %someFloat32
  # CHECK: {{%.*}} = lit.call {{.*}}__iadd__{{.*}}(%someFloat32, [[Float32]])
  var someFloat32 : Float32
  (someFloat32) += someFloat32

  # CHECK: %someSIMD = lit.var.decl "someSIMD" var
  # CHECK: [[SIMD:%.*]] = lit.ref.load %someSIMD
  # CHECK: {{%.*}} = lit.call {{.*}}@builtin::@simd::@SIMD::@"__iadd__({{.*}}(%someSIMD, [[SIMD]])
  var someSIMD : SIMD[DType.float64, 4]
  (someSIMD) += someSIMD

# CHECK-LABEL: lit.func @"byval_byref_function(::Int,::Int&)"{{.*}}(%a: !Int, %b: !lit.ref<!Int, mut {{.*}}> inout) -> !kgen.none
fn byval_byref_function(a: Int, inout b: Int):
  # CHECK-NEXT: [[BI:%.*]] = kgen.rebind %b {{.*}}#lit.invalid.ref.lifetime
  # CHECK-NEXT: lit.ref.store %a, [[BI]]
  b = a

  # CHECK-NEXT: %x = lit.var.decl "x" var
  var x : Int
  # This needs to load 'b' to pass it by value for the first arg, but pass its
  # address in directly for the second.
  # CHECK: [[TMP:%.*]] = lit.ref.load [[BI]]
  # CHECK: = lit.call @{{.*}}::@"byval_byref_function{{.*}}([[TMP]], [[BI]])
  byval_byref_function(b, b)

# CHECK-LABEL: lit.func @"lvaluesAndRValues()
fn lvaluesAndRValues() -> __mlir_type.index:
  # CHECK: [[VALUE:%.*]] = kgen.param.constant = <4>
  # CHECK: lit.return [[VALUE]] : index
  return Int(4).value

# CHECK-LABEL: lit.func @"mvalueStructField()"
fn mvalueStructField():
  # CHECK: lit.alias.decl [[INT:.*]]: !Int = <{4}>
  alias int = Int(4)
  # CHECK: lit.alias.decl *"value{{.*}}" = <#lit.struct.extract<:!Int [[INT]], "value">>
  alias value = int.value
  alias foldToValue = Int(5).value

##===----------------------------------------------------------------------===##
# Augmented Assignments
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"basic_assignments
fn basic_assignments(a0: Int, b: Int, c: RegPassable, d: RegPassable):
  var a = a0
  # CHECK-NEXT:      %a = lit.var.decl "a" var
  # CHECK-NEXT: lit.ref.store %a0, %a
  # CHECK-NEXT: lit.call {{.*}}Int::@"__iadd__{{.*}}(%a, %b)
  a += b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__isub__{{.*}}(%a, %b)
  a -= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__imul__{{.*}}(%a, %b)
  a *= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__ifloordiv__{{.*}}(%a, %b)
  a //= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__imod__{{.*}}(%a, %b)
  a %= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__ipow__{{.*}}(%a, %b)
  a **= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__irshift__{{.*}}(%a, %b)
  a >>= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__ilshift__{{.*}}(%a, %b)
  a <<= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__iand__{{.*}}(%a, %b)
  a &= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__ixor__{{.*}}(%a, %b)
  a ^= b
  # CHECK-NEXT: lit.call {{.*}}Int::@"__ior__{{.*}}(%a, %b)
  a |= b

  var x: Int
  # CHECK-NEXT: %x = lit.var.decl
  # CHECK-NEXT: %[[FOUR:.*]] = kgen.param.constant: !Int = <{4}>
  # CHECK-NEXT: lit.ref.store %[[FOUR]], %x
  # CHECK-NEXT: lit.ref.store %[[FOUR]], %a
  a = x = 4

  # Walrus
  # CHECK-NEXT: %[[SEVEN:.*]] = kgen.param.constant: !Int = <{7}>
  # CHECK-NEXT: lit.ref.store %[[SEVEN]], %x
  # CHECK-NEXT: %[[A:.*]] = lit.ref.load %a
  # CHECK-NEXT: lit.call {{.*}}simpleMath{{.*}}(%[[A]], %[[SEVEN]])
  _ = simpleMath(a, x := 7)

# Issue #20145: Walrus operator should implicitly declare variable in def functions.
# CHECK-LABEL: lit.func @"walrus_implicit_decl
def walrus_implicit_decl():
  # CHECK:      %d = lit.var.decl "d" imp
  # CHECK:      %c = lit.var.decl "c" imp
  # CHECK:      %b = lit.var.decl "b" imp
  # CHECK:      %a = lit.var.decl "a" imp

  # CHECK-NEXT: [[THREE:%.*]] = kgen.param.constant: !Int = <{3}>
  # CHECK-NEXT: lit.ref.store [[THREE]], %a
  # CHECK-NEXT: [[VAR_A:%.*]] = lit.ref.load %a
  # CHECK-NEXT: lit.call {{.*}}([[THREE]], [[VAR_A]])
  _ = simpleMath(a := 3, a)

  # CHECK-NEXT: hlcf.elif {
  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: !Int = <{4}>
  # CHECK-NEXT: lit.ref.store [[FOUR]], %b
  if b := 4:
    print(b)

  # CHECK: [[FIVE:%.*]] = kgen.param.constant: !Int = <{5}>
  # CHECK-NEXT: lit.ref.store [[FIVE]], %c
  # CHECK-NEXT: lit.ref.store [[FIVE]], %d
  d = c := 5

##===----------------------------------------------------------------------===##
# Literals
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"literals
def literals():
    a = 5             # CHECK: 5
    a = 55            # CHECK: 55
    a = 10500         # CHECK: 10500
    a = 12_500        # CHECK: 12500
    a = 0             # CHECK: 0
    a = 00            # CHECK: 0
    a = 0____0__0_0   # CHECK: 0
    a = 0__           # CHECK: 0
    a = 00__0_0       # CHECK: 0
    a = 1__9_         # CHECK: 19
    a = 0x123         # CHECK: 291
    a = 0X123         # CHECK: 291
    a = 0b10101       # CHECK: 21
    a = 0B10101       # CHECK: 21
    a = 0o711         # CHECK: 457
    a = 0O711         # CHECK: 457
    # Test parsing for this value with lots of underscores here because mblack
    # can't handle it.
    alias b = 1_2.3__1e+1_1 # CHECK: #kgen.float_literal<1231000000000|1>
    c = False         # CHECK: !Bool = <{:i1 0}>
    c = True          # CHECK: !Bool = <{:i1 1}>

# CHECK-LABEL: lit.func @"_strings
fn _strings():
   """
      Various tests on strings
   """

    var a = ""                 # CHECK: ""
    # CHECK: "hello world"
    a = "hello \
world"

    # COM: match newline hex values via regex since they vary between OSs
    # CHECK: "hello \\{{[\\0-9A-Z]+}}world"
    a = r"hello \
world"

    # CHECK:  "1'{{(\\0D)?}}\0A2"
    a = """1'
2"""

    # CHECK:  "1\222"
    a = '''1"\
2'''

    # CHECK:   "1\22{{(\\0D)?}}\0A2"
    a = '''1"
2'''

    # CHECK:   "1\22\0A2"
    a = '1"\n2'

    # CHECK: "hello concat world"
    a = "hello " "concat " "world"

    a = "Hello"            # CHECK: "Hello"
    a = "Hello 'world'"    # CHECK: "Hello 'world'"
    a = "A\x42"            # CHECK: "AB"
    a = "A\x423"           # CHECK: "AB3"
    a = "A\102"            # CHECK: "AB"
    a = "A\1023"           # CHECK: "AB3"

    # COM: the MLIR textual representation escapes strings, so below \ is \\ and " is \"
    a = 'Hello "world"'    # CHECK: "Hello \22world\22"
    a = r"A\x42"           # CHECK: "A\\x42"
    a = R"A\x42"           # CHECK: "A\\x42"
    a = r"AB\\"            # CHECK: "AB\\\\"
    a = r"A\x"             # CHECK: "A\\x"
    a = "AB\\"             # CHECK: "AB\\"
    a = r"A\"B"            # CHECK: "A\\\22B"
    a = r'A\'B'            # CHECK: "A\\'B"
    a = "A\"B"             # CHECK: "A\22B"
    a = 'A\'B'             # CHECK: "A'B"
    a = r"A\zB"            # CHECK: "A\\zB"

    # Issue #201: https://github.com/modularml/mojo/issues/201
    # CHECK: lit.func *"hello{{.*}} {
    fn hello() -> StringLiteral:
        # CHECK: kgen.param.constant: !StringLiteral = <{:string "123"}>
        return "123"
        # lit.end_func
    """other comment"""


##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

# This is an array that has elements of MemoryOnlyInt.
struct MemoryOnlyIntArray:
  fn __getitem__(inout self, x: Int) -> MemoryOnlyInt: pass
  fn __setitem__(inout self, x: Int, owned value: MemoryOnlyInt): pass

# CHECK-LABEL: lit.func @"testMemoryOnlyIntArray
fn testMemoryOnlyIntArray(inout arr: MemoryOnlyIntArray, x: Int, owned moi: MemoryOnlyInt):
  # CHECK: %moi28transfer29 = lit.transfer_mem_ownership %moi
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %moi28transfer29)
  arr[x] = moi^
  # CHECK: [[ANON:%.*]] = lit.var.decl "anonymous*"
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, %anonymous2A)
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %anonymous2A)
  arr[x] = arr[x]

  # CHECK: [[ANON:%.*]] = lit.var.decl "__store_tmp__"
  # CHECK-SAME: : !lit.ref<!MemoryOnlyInt, mut *"__store_tmp__`
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[ANON]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[ANON]][x]
  # CHECK: %[[C1:.*]] = {{.*}}constant: !Int = <{1}>
  # CHECK: lit.ref.store %[[C1:.*]], [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x].x = 1

  # Initialize in memory through a temp + setitem.
  # CHECK: [[ANON:%.*]] = lit.var.decl "anonymous*"
  # CHECK: lit.call @{{.*}}__init__{{.*}}([[ANON]],
  # CHECK: lit.call {{.*}}"__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x] = MemoryOnlyInt(42)

  # CHECK: [[STORETMP:%.*]] = lit.var.decl "__store_tmp__"
  # CHECK-SAME: : !lit.ref<!MemoryOnlyInt, mut *"__store_tmp__`
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[STORETMP]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[STORETMP]][x]
  # CHECK: lit.ref.store {{.*}}, [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[STORETMP]])
  arr[x].x += 1

struct MyInlineIntInit:
    var value: MemoryOnlyInt
    # CHECK-LABEL: lit.func @"__init__(expressions::MyInlineIntInit=&,expressions::MemoryOnlyInt)"
    # CHECK-SAME: (%self: !lit.ref<!MyInlineIntInit, mut {{.*}}> init_self, |, %value: !lit.ref<!MemoryOnlyInt, imm {{.*}}> borrow_in_mem) -> !kgen.none
    fn __init__(inout self, value: MemoryOnlyInt):
        # CHECK: %0 = lit.ref.struct.ger %self[value]
        # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%0, %value)
        self.value = value

@register_passable
struct ConstDynamicObject:
    fn __init__(inout self):
        return

    fn __getattr__(self, name: StringLiteral) -> Int:
        return 0

struct DynamicObject:
    fn __init__(inout self):
        pass

    fn __getattr__(self, name: StringLiteral) -> Int:
        return 0

    fn __setattr__(self, name: StringLiteral, value: Int):
        pass


# CHECK-LABEL: lit.func @"dynamic_attribute()"
fn dynamic_attribute():
    # CHECK: %const_obj = lit.var.decl "const_obj"
    var const_obj = ConstDynamicObject()
    # CHECK: %[[KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "dynamic_attribute"}>
    # CHECK: call {{.*}}@ConstDynamicObject::@"__getattr__{{.*}}"(
    _ = const_obj.dynamic_attribute

    var obj = DynamicObject()
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: %[[KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "some_attr"}>
    # CHECK: call {{.*}}@DynamicObject::@"__getattr__{{.*}}([[IMMREF]],
    _ = obj.some_attr
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: %[[KEY:.*]] = kgen.param.constant: !StringLiteral = <{:string "some_attr"}>
    # CHECK: %[[VALUE:.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: call {{.*}}@DynamicObject::@"__setattr__{{.*}}([[IMMREF]], {{.*}}, %[[VALUE]])
    obj.some_attr = 42


# CHECK-LABEL: lit.func @"chained_cmp
fn chained_cmp(a: Int, b: Int, c: Int, d: Int, e: Int):
    # CHECK-NEXT: %res = lit.var.decl "res"
    # CHECK:      [[CMP_A_B:%.*]] = lit.call @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK-NEXT: %[[CMP_A_B_I1:.*]] = lit.call @{{.*}}__mlir_i1__{{.*}}([[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.if %[[CMP_A_B_I1]]
    # CHECK-NEXT:   %[[CMP_B_C:.*]] = lit.call @{{.*}}__lt__{{.*}}(%b, %c)
    # CHECK:        %[[IF_B_C:.*]] = hlcf.if
    # CHECK-NEXT:     %[[CMP_C_D:.*]] = lit.call @{{.*}}__lt__{{.*}}(%c, %d)
    # CHECK-NEXT:     hlcf.yield %[[CMP_C_D]]
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT:   }
    # CHECK-NEXT:   hlcf.yield %[[IF_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield [[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF_A_B]], %res
    var res = a < b < c < d

    # COM: This checks the parsing precedence between `<` and `and`.
    # CHECK:      %[[CMP_A_B:.*]] = lit.call @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK:       %[[CMP_A_B_I1:.*]] = lit.call @{{.*}}__mlir_i1__{{.*}}(%[[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.if %[[CMP_A_B_I1]]
    # CHECK:   %[[CMP_B_C:.*]] = lit.call @{{.*}}__lt__{{.*}}(
    # CHECK-NEXT:   hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: %[[CMP_I1:.*]] = lit.call @{{.*}}__mlir_i1__{{.*}}(%[[IF_A_B]])
    # CHECK-NEXT: %[[IF:.*]] = hlcf.if %[[CMP_I1]]
    # CHECK-NEXT:   %[[CMP_D_E:.*]] = lit.call @{{.*}}__lt__{{.*}}(%d, %e)
    # CHECK-NEXT:   hlcf.yield %[[CMP_D_E]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[IF_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF]], %res
    res = a < b < c and d < e

# Test chained comparison op in parameter domain for issue
# https://github.com/modularml/modular/issues/22050
# CHECK: lit.alias.decl *"chainedCmpAlias1{{.*}}": !Bool ={{.*}}{:i1 0}
alias chainedCmpAlias1 = 1 == 2 == 3 == 4 == 5
# CHECK: lit.alias.decl *"chainedCmpAlias2{{.*}}": !Bool ={{.*}}{:i1 1}
alias chainedCmpAlias2 = 1 <= 2 <= 3 <= 4 <= 5
# CHECK: lit.alias.decl *"chainedCmpAlias3{{.*}}": !Bool ={{.*}}{:i1 0}
alias chainedCmpAlias3 = 1 <= 2 <= 9 <= 4 <= 5
fn chainedCmpSemiDyn(x: Int, a: Int, b: Int, c: Int):
  # CHECK: [[XCMP:%.*]] = lit.var.decl "xCmp"
  # CHECK-NEXT: [[IFCOND:%.*]] = kgen.param.constant: i1 = <1>
  # CHECK-NEXT: [[FINALRESULT:%.*]] = hlcf.if [[IFCOND]] -> !Bool {
  # CHECK-NEXT:   [[PV:%.*]] = {{.*}}constant{{.*}}77
  # CHECK-NEXT:   [[CMPRESULT1:%.*]] = {{.*}}__lt__{{.*}}([[PV]], %x)
  # CHECK-NEXT:   [[IFCOND:%.*]] = {{.*}}__mlir_i1__{{.*}}([[CMPRESULT1]])
  # CHECK-NEXT:   [[INNERRESULT:%.*]] = hlcf.if [[IFCOND]] -> !Bool {
  # CHECK-NEXT:     [[PV:%.*]] = {{.*}}constant{{.*}}105
  # CHECK-NEXT:     [[CMPRESULT2:%.*]] = {{.*}}__lt__{{.*}}(%x, [[PV]])
  # CHECK-NEXT:     [[IFCOND:%.*]] = {{.*}}__mlir_i1__{{.*}}([[CMPRESULT2]])
  # CHECK-NEXT:     [[MOSTINNERRESULT:%.*]] = hlcf.if [[IFCOND]] -> !Bool {
  # CHECK-NEXT:       [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}{:i1 1}
  # CHECK-NEXT:       hlcf.yield [[TRUEPARAM]]
  # CHECK-NEXT:     } else {
  # CHECK-NEXT:       hlcf.yield [[CMPRESULT2]]
  # CHECK-NEXT:     }
  # CHECK-NEXT:     hlcf.yield [[MOSTINNERRESULT]]
  # CHECK-NEXT:   } else {
  # CHECK-NEXT:     hlcf.yield [[CMPRESULT1]]
  # CHECK-NEXT:   }
  # CHECK-NEXT:   hlcf.yield [[INNERRESULT]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}{:i1 1}
  # CHECK-NEXT:   hlcf.yield [[TRUEPARAM]]
  # CHECK-NEXT: }
  # CHECK-NEXT: lit.ref.store [[FINALRESULT]], [[XCMP]]
  var xCmp = 5 < 77 < x < 105 < 177
  # A fully deep check of this would be a lot of work, but this at least
  # shows that its not choking during parsing on a mix of dynamic and
  # parameter comparisons.  It required some care with the interaction
  # between recursive calls of emitNextCmp calls to get this to work.
  var mixedChain = 0 < 1 < a < 10 < 11 < b < 20 < 21 < c < 30 < 31

# CHECK-LABEL: lit.func @"ref_utilities
fn ref_utilities(a: MemoryOnlyInt, inout b: MemoryOnlyInt,
                 inout c: MemoryOnlyInt,
                 cond: __mlir_type.i1):
  # Get the address of the specified physical bvalue or lvalue as a lit.ref.

  # CHECK: %ref1 = lit.var.decl "ref1"
  var ref1 = __get_mvalue_as_litref(a)
  # CHECK: %ref2 = lit.var.decl "ref2"
  var ref2 = __get_mvalue_as_litref(b)

  # CHECK: %ptr1 = lit.var.decl "ptr1"
  # CHECK: [[REF1V:%.*]] = lit.ref.load %ref1
  # CHECK-NEXT: [[MV:%.*]] = lit.ref.to_pointer [[REF1V]]
  # CHECK-NEXT: lit.ref.store [[MV]], %ptr1
  var ptr1 = __mlir_op.`lit.ref.to_pointer`(ref1)

  # CHECK-NEXT: %localLet = lit.var.decl "localLet"
  var localLet = MemoryOnlyInt()
  # CHECK: %ref3 = lit.var.decl "ref3"
  var ref3 = __get_mvalue_as_litref(localLet)

  # CHECK: %localVar = lit.var.decl "localVar"
  var localVar = MemoryOnlyInt()
  # CHECK: %ref4 = lit.var.decl "ref4"
  var ref4 = __get_mvalue_as_litref(localVar)

  # CHECK: %ref5 = lit.var.decl "ref5"
  # CHECK: [[COMMON:%.*]] = hlcf.if %cond -> !lit.ref<!MemoryOnlyInt, imm {*"a`", (mutcast mut *"b`1"), (mutcast mut *"c`2")}> {
  # CHECK-NEXT:   [[COMMONINNER:%.*]] = hlcf.if %cond -> !lit.ref<!MemoryOnlyInt, imm {*"a`", (mutcast mut *"b`1")}> {
  # CHECK-NEXT:     [[TMP:%.*]] = kgen.rebind %ref1
  # CHECK-NEXT:     [[REF1V:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT:     hlcf.yield [[REF1V]]
  # CHECK-NEXT:   } else {
  # CHECK-NEXT:     [[TMP:%.*]] = kgen.rebind %ref2
  # CHECK-NEXT:     [[REF2V:%.*]] = lit.ref.load [[TMP]]
  # CHECK-NEXT:     hlcf.yield [[REF2V]]{{.*}}>
  # CHECK-NEXT:   }
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind [[COMMONINNER]]
  # CHECK-SAME:           !lit.ref<!MemoryOnlyInt, imm {*"a`", (mutcast mut *"b`1")}> to !lit.ref<!MemoryOnlyInt, imm {*"a`", (mutcast mut *"b`1"), (mutcast mut *"c`2")}>
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind %c : !lit.ref<!MemoryOnlyInt, mut *"c`2"> to !lit.ref<!MemoryOnlyInt, imm {*"a`", (mutcast mut *"b`1"), (mutcast mut *"c`2")}>
  # CHECK-NEXT:   hlcf.yield [[TMP]] : !lit.ref<{{.*}}>
  # CHECK-NEXT: }
  # CHECK-NEXT: lit.ref.store [[COMMON]], %ref5
  var ref5 = (ref1 if cond else ref2) if cond else __get_mvalue_as_litref(c)

  # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant: !Int = <{42}>
  # CHECK-NEXT: [[TMP2:%.*]] = lit.ref.load %ref2
  # CHECK-NEXT: lit.call {{.*}}MemoryOnlyInt::@"__init__{{.*}}([[TMP2]], [[TMP]])
  __get_litref_as_mvalue(ref2) = MemoryOnlyInt()

struct CallableStruct:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __call__(self, rhs: Int) -> Int:
        return self.value + rhs

# CHECK-LABEL: lit.func @"test_call_method()"
fn test_call_method():
    # CHECK: %[[C2:.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: lit.call {{.*}}@"__call__{{.*}}(%{{.*}}, %[[C2]])
    var value = CallableStruct(5)
    _ = value(2)

struct MemoryType:
  fn __copyinit__(inout self, other: Self):
    pass

@register_passable
struct RegType: pass

# CHECK-LABEL: lit.struct.decl @ParamType
# CHECK-SAME: <a: !Int>
@register_passable
struct ParamType[a: Int]: pass

# CHECK-LABEL: lit.func @"function_types
fn function_types[
  # CHECK-SAME: p0: {{.*}}<<"a": !Int>(!lit.struct<#ParamType <:!Int *(0,0)>{{.*}}>, |) -> !kgen.none
  p0: fn[a: Int](ParamType[a]) -> None,

  # CHECK-SAME: p1: {{.*}}<[2]<"a": !Int, "b": {{.*}}@ParamType<:!Int *(0,0)>>(?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> i1
  p1: def[a: Int, b: ParamType[a]]() -> None,

  # CHECK-SAME: p2: {{.*}}"Ts": variadic<!AnyType> var>(!lit.struct<#VariadicPack <:i1 0, :lifetime<0> *[0,0], :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> *(0,0)>> borrow_in_mem|pack, ?, "__result__": !lit.ref<none, mut *[0,1]> byref_result) async
  p2: async fn[*Ts: AnyType](* *Ts) -> None,
](
  # CHECK-SAME: %{{.*}}: {{.*}}(!Int, |) -> !Int
  float0: fn(Int) -> Int,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!MemoryType, imm {{.*}}> borrow_in_mem, |, ?, "__result__": !lit.ref<!MemoryType, mut {{.*}}> byref_result) -> !kgen.none
  float1: fn(MemoryType) -> MemoryType,

  # CHECK-SAME: %{{.*}}: {{.*}}(!RegType owned, |) -> !RegType
  float2: fn(owned RegType) -> RegType,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!MemoryType, mut *[0,0]> owned_in_mem, |) -> !kgen.none
  float3: fn(owned MemoryType) -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!Int, mut *[0,0]> inout, |) -> !kgen.none
  float4: fn(inout Int) -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!Int, |, ?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> i1
  float5: fn(Int) raises -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!Int, |, ?, "__result__": !lit.ref<none, mut *[0,0]> byref_result) async|capturing -> !kgen.none
  float6: async fn(Int) capturing -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!kgen.variadic<!Int> var, ?, {{.*}}) throws -> i1
  float7: def(*Int) -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}<(!Int = {10}, !StringLiteral = {:string "foo"}, |) -> !kgen.none>
  float12: fn(Int = 10, StringLiteral = "foo") -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}<[1]("x": !lit.ref<!MemoryType, imm {{.*}}> borrow_in_mem) -> !Int>
  named: fn(x: MemoryType) -> Int
): pass

# CHECK-LABEL: lit.struct.decl @Mem
# CHECK:         lit.alias.decl *"x{{.*}}": type = <i8>
# CHECK-NEXT:    lit.alias.decl *"B{{.*}}": type = <!lit.signature<("foo": i8) -> !kgen.none>>
struct Mem:
   alias x = __mlir_type.i8
   alias B = fn (foo: Self.x) -> None

alias fn_type_alias = fn() -> None

@always_inline
fn func_with_decorator(): pass


struct TwoParamsStruct[a: Int, b: Int]:
    fn __copyinit__(inout self, other: Self):
        pass

# CHECK-LABEL: lit.func @"variadic_subscript{{.*}}"<idx: !Int, a: variadic<!Int> var>
fn variadic_subscript[idx: Int, *a: Int](*b: Int):
    # CHECK-NEXT: %b_0 = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}VariadicList{{.*}}__init__{{.*}}(%b_0, %b)
    # CHECK: lit.alias.decl *"v0{{.*}}": {{.*}}Int = <variadic_get(:variadic<!Int> a, 2)>
    alias v0 = a[2]

    # CHECK: %v1 = lit.var.decl "v1"
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <variadic_get(:variadic<!Int> a, 3)>
    # CHECK: lit.ref.store [[TMP]], %v1
    var v1 = a[3]
    # CHECK: %[[LIST:.*]] = lit.ref.load %b_0
    # CHECK: lit.call {{.*}}__getitem__{{.*}}(%[[LIST]],
    var v2 = b[idx]


# CHECK-LABEL: lit.func @"variadic_memory_subscript
# CHECK-SAME: variadic<!lit.ref<{{.*}}TwoParamsStruct<
# CHECK-SAME:   :!Int variadic_get({{.*}}a, 0)
# CHECK-SAME:   :!Int variadic_get({{.*}}a, 1)
fn variadic_memory_subscript[*a: Int](*b: TwoParamsStruct[a[0], a[1]]):
    # CHECK: %b_0 = lit.var.decl
    # CHECK: %v0 = lit.var.decl
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %b_0 :
    # CHECK: [[B1REF:%.*]] = {{.*}}__getitem__{{.*}}([[IMMREF]],
    # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%v0, [[B1REF]])
    var v0 = b[1]
    # CHECK: %v1 = lit.var.decl
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %b_0 :
    # CHECK: [[B2REF:%.*]] = {{.*}}__getitem__{{.*}}([[IMMREF]],
    # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%v1, [[B2REF]])
    var v1 = b[2]

fn takeMemory(a: MemoryType): pass

# CHECK-LABEL: lit.func @"testConds
fn testConds(cond: __mlir_type.i1, a: MemoryType, b: MemoryType, m: RegPassable, i: Int) -> MemoryType:
  # Implicit conversions.
  # Mojo Issue #49: https://github.com/modularml/mojo/issues/49

  # CHECK-NEXT: hlcf.if %cond -> !RegPassable {
  # CHECK:        lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %m)
  # CHECK:        hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.var.decl "anonymous
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}({{.*}}, %i)
  # CHECK-NEXT:   [[V:%.*]] = lit.load.consume
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: }
  _ = m if cond else i

  # CHECK-NEXT: hlcf.if %cond -> !RegPassable {
  # CHECK:        lit.call {{.*}}__init__{{.*}}({{.*}}, %i)
  # CHECK:        hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK:        lit.call {{.*}}__copyinit__{{.*}}({{.*}}, %m)
  # CHECK:        hlcf.yield
  # CHECK-NEXT: }
  _ = i if cond else m

  # Memory only conds.
  # Issue (#13379)

  # CHECK-NEXT: %anonymous2A = lit.var.decl
  # CHECK-NEXT: hlcf.if %cond {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %a)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %b)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A
  # CHECK-NEXT: lit.call {{.*}}takeMemory{{.*}}([[IMMREF]])
  takeMemory(a if cond else b)

  # CHECK-NEXT: hlcf.if %cond {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%__result__, %a)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%__result__, %b)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  # CHECK-NEXT: kgen.param.constant: none = <#kgen.none>
  return a if cond else b

fn testTransferWarning():
  var a = MemoryOnlyInt()

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(a^^)

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(MemoryOnlyInt()^)


##===----------------------------------------------------------------------===##
# Test nonmaterializable IntLiteral beyond Int bounds.
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"bigggNumber{{.*}}": !IntLiteral = <{:!kgen.int_literal 115792089237316195423570985008687907853269984665640564039457584007913129639936}>
alias bigggNumber = 2 << 255
fn useBigNumber() -> Int:
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <{512}>
  var notSoBig = bigggNumber // (2 << 246)
  # Easy min-int
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <{-9223372036854775808}>
  var minInt = -(2<<62)
  return notSoBig

##===----------------------------------------------------------------------===##
# Test return slot optimization
##===----------------------------------------------------------------------===##

struct Unmovable:
  fn __init__(inout self): pass

# NOTE: Don't remove this argument, this was defeating return slot opzn.
fn getUnmovable(a: Unmovable) -> Unmovable:
  return Unmovable()

# This can only be codegen'd directly into x.
# CHECK-LABEL: lit.func @"testUnmovable
fn testUnmovable(a: Unmovable):
   # CHECK-NEXT: %x = lit.var.decl "x"
   # CHECK-NEXT: lit.call {{.*}}(%a, %x)
   var x : Unmovable = getUnmovable(a)

# Issue 23233 https://github.com/modularml/modular/issues/23233
fn setitemParamToDLValue():
  alias x = 3
  var coords = StaticIntTuple[3](0)
  # The main check is just that it's not erroring.
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <apply{{.*}}__neg__
  # CHECK: lit.call {{.*}}StaticIntTuple{{.*}}__setitem__{{.*}}[[VAR]]
  coords[1] = -x

# https://github.com/modularml/mojo/issues/734
fn reg_passable_trivial():
  var x : Int = 100
  x = 42
  _ = x^  # expected-warning {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}

  var y : Int = 100
  # expected-warning @+1 {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}
  _ = y^  # Consume RValue / BValue is not, this isn't tracked.




fn del_warnings():
  # These copy the value before destroying it, which is pointless.
  var m = MemoryOnlyInt()
  m.__del__()  # expected-warning {{explicit call to '__del__' destroys a copy of the value; consider removing this call}}
  var r = RegPassable(1)
  r.__del__()  # expected-warning {{explicit call to '__del__' destroys a copy of the value; consider removing this call}}

  # These is weird/unneeded, but at least it does what it says.
  MemoryOnlyInt().__del__()
  RegPassable(1).__del__()

##===----------------------------------------------------------------------===##
# __type_of
##===----------------------------------------------------------------------===##

alias index = __mlir_type.index

# CHECK-LABEL: lit.func @"foo(
# CHECK: __mlir_type.index)"(%x: index) -> index
fn foo(x: index) -> __type_of(x):
    return x


# CHECK-LABEL: lit.func @"bar(
# CHECK: __mlir_type.index,__mlir_type.index)"(%x: index, %y: index) -> index
fn bar(x: index, y: __type_of(x)) -> index:
    var z : __type_of(x) = y
    return z

##===----------------------------------------------------------------------===##
# __lifetime_of
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"lifetime_of
fn lifetime_of(x: Unmovable, y: Unmovable, inout z: Unmovable):
    # CHECK-NEXT: lifetime<1> = <#lit.lifetime>
    alias lt0 = __lifetime_of()
    # CHECK-NEXT: lifetime<0> = <*"x`">
    alias lt1 = __lifetime_of(x)
    # CHECK-NEXT: lifetime<0> = <{*"x`", *"y`1"}>
    alias lt2 = __lifetime_of(x, y)
    # CHECK-NEXT: lifetime<1> = <*"z`2">
    alias lt3 = __lifetime_of(z)
    # CHECK-NEXT: lifetime<0> = <{*"x`", (mutcast mut *"z`2")}>
    alias lt4 = __lifetime_of(x, z)

##===----------------------------------------------------------------------===##
# in / not in
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"test_in
fn test_in(a: String, b: String):
    # CHECK-NEXT: lit.call {{.*}}__contains__{{.*}}(%b, %a)
    _ = a in b
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}__contains__{{.*}}(%b, %a)
    # CHECK-NEXT: [[RESB:%.*]] = lit.call {{.*}}__bool__{{.*}}([[RES]])
    # CHECK-NEXT: = lit.call {{.*}}__invert__{{.*}}([[RESB]])
    _ = a not in b

##===----------------------------------------------------------------------===##
# Parameter inference
##===----------------------------------------------------------------------===##

# Test that parameter inference can handle this.
fn dependent_call_it[dtype: DType](ptr: DTypePointer[dtype]):
   dependent_callee(ptr, 0.0)
# This requires substitution to realize that storage.type == DType
fn dependent_callee[dtype: DType](storage: DTypePointer[dtype],
                   pad_value: Scalar[storage.type]):
   pass

# This requires handling of VariadicAttr in parameter inference.
fn variadic_attr_caller(*inputs: Tuple[Int]):
   variadic_attr_callee[Int](inputs)
fn variadic_attr_callee[key_type: CollectionElement](
       inputs: VariadicListMem[Tuple[key_type], _, _]
    ):
  pass

# Test that parameter inference works with implicit conversions - in this case
# that we can infer the parameters of 'thing_taking_reference' even though x
# needs to be built as a Reference.
fn thing_taking_ref[
  type: AnyType,
  //,
  # TODO: Add _.
  is_mutable: Bool,
  lifetime: AnyLifetime[is_mutable].type,
](ref [lifetime] arg: type): pass

fn thing_taking_ref2[type: AnyType](ref [_] arg: type): pass

fn thing_taking_reference2[type: AnyType](arg: Reference[type, _]): pass

# CHECK-LABEL: lit.func @"test_thing_taking_reference
fn test_thing_taking_reference(inout x: String):
  # CHECK-NEXT: lit.call {{.*}}thing_taking_ref{{.*}}(%x)
  thing_taking_ref(x)
  # CHECK-NEXT: lit.call {{.*}}thing_taking_ref2{{.*}}(%x)
  thing_taking_ref2(x)
# CHECK-NEXT: %anonymous2A = lit.var.decl
# CHECK-NEXT: lit.call {{.*}}@Reference::@"__init__
# CHECK-SAME: <:!Bool {:i1 1}, :!AnyType #String1, :lifetime<1> *"x`", :!AddressSpace {_value: !Int = {0}}>
  thing_taking_reference2(x)

struct StructWithStaticMethods:
   @staticmethod
   fn _init_op_state(state: Reference[Int, _], foo: Int): pass
   fn thing(self):
     var x = 42
     Self._init_op_state(x, x)

fn infer_through_alias():
  alias MyType = MemoryOnlyInt
  _ = MyType(4)


# CHECK-LABEL: lit.func @"infer_address_space
fn infer_address_space[
    is_mutable: __mlir_type.i1,
    lifetime: AnyLifetime[is_mutable].type
](a: Reference[Int, lifetime, AddressSpace(4)]._mlir_type):
  # Show that we can infer the address space parameter of Reference from a
  # !lit.ref.

  # CHECK: lit.call {{.*}}@Reference::@"__init__{{.*}}:!AddressSpace {_value: !Int = {4}}>
  var x = Reference(__get_litref_as_mvalue(a))


# https://linear.app/modularml/issue/MOCO-584/[references]-we-cannot-bind-litref-in-parameter-context
# [References] We cannot bind !lit.ref in parameter context
struct ThingWithMethodReferenceSelf:
    fn method(ref [_] a: Self):
      pass

# CHECK-LABEL: lit.func @"testThingWithMethodReferenceSelf
fn testThingWithMethodReferenceSelf[a: ThingWithMethodReferenceSelf]():
    # CHECK-NEXT: lit.alias.decl *"sizzle`": none =
    # CHECK-SAME: <apply(:!lit.signature<("a": !lit.ref<!ThingWithMethodReferenceSelf,
    # CHECK-SAME:     <:i1 1, :lifetime<1> #lit.lifetime>, store_to_mem(a))>
    alias sizzle = a.method()
