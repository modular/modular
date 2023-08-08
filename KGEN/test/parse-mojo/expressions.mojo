# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -verify-diagnostics -import-mojo %s | FileCheck %s

# CHECK: module {

from DType import DType
from SIMD import Float32, SIMD
from Object import object

fn noop(): pass

# CHECK-LABEL: lit.struct.decl @MemoryOnlyInt
struct MemoryOnlyInt:
  var x: Int

  # CHECK-LABEL: lit.func @"__init__
  fn __init__(inout self, a: Int = 42):
    # CHECK: %0 = lit.struct.gep %self[x]
    # CHECK: %1 = {{.*}}constant: {{.*}}Int = {{.*}} 1
    # CHECK: pop.store %1, %0
    self.x = 1

  # CHECK-LABEL: lit.func @"__copyinit__
  fn __copyinit__(inout self, existing: Self):
    self.x = existing.x

  @staticmethod
  fn variadic(*value: MemoryOnlyInt):
    pass

fn consume(owned a: MemoryOnlyInt): pass

# This type is used to test implicit conversion from MemoryOnlyInt
struct MemoryOnlyFloat64:
  var x: FloatLiteral
  fn __init__(inout self, value: MemoryOnlyInt):
    self.x = 1.0

# CHECK-LABEL: lit.struct.decl @MemoryOnlyPair
struct MemoryOnlyPair:
  var x: MemoryOnlyInt
  var y: Int

  # CHECK: lit.func @"__copyinit__{{.*}}"(
  # CHECK-SAME: %self: !pop.pointer<@"$expressions"::@MemoryOnlyPair> init_self,
  # CHECK-SAME: %existing: !pop.pointer<@"$expressions"::@MemoryOnlyPair> borrow_in_mem) -> !lit.none
  fn __copyinit__(inout self, existing: MemoryOnlyPair):
    # CHECK-NEXT: %0 = lit.struct.gep %existing[x]
    # CHECK-NEXT: %1 = lit.struct.gep %self[x]
    # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}(%1, %0)
    # CHECK-NEXT: %3 = lit.struct.gep %existing[y]
    # CHECK-NEXT: %4 = lit.struct.gep %self[y]
    # CHECK-NEXT: %5 = pop.load %3
    # CHECK-NEXT: pop.store %5, %4
    self.x = existing.x
    self.y = existing.y

  # CHECK: lit.func @"method{{.*}}"(
  # CHECK-SAME: %self: !pop.pointer<@"$expressions"::@MemoryOnlyPair> owned_in_mem,
  # CHECK-SAME: %arg: !pop.pointer<@"$expressions"::@MemoryOnlyInt> owned_in_mem)
  fn method(owned self, owned arg: MemoryOnlyInt):
    # CHECK: %0 = lit.struct.gep %self[y]
    # CHECK: %1 = lit.struct.gep %arg[x]
    # CHECK: %2 = pop.load %0
    # CHECK: %3 = pop.load %1
    # CHECK: %4 = kgen.call @"{{.*}}__add__{{.*}}"(%2, %3)
    _ = self.y+arg.x

fn inferred_function_with_memory_result[
  width: Int](x: SIMD[DType.float32, width]) -> MemoryOnlyInt: pass

# CHECK-LABEL: lit.func @"memoryOnlyOps
fn memoryOnlyOps(inout a: MemoryOnlyPair) -> MemoryOnlyPair:
  # CHECK-NEXT: %v1 = lit.varlet.decl {{.*}} : <@"$expressions"::@MemoryOnlyPair>
  # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}(%v1, %a)
  var v1 = a

  # CHECK-NEXT: %v2 = lit.varlet.decl "v2", var = false
  # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}(%v2, %a)
  let v2 : MemoryOnlyPair = a

  # CHECK-NEXT: %anonymous2A = lit.varlet.decl
  # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %a)
  _ = a

  a  # expected-warning {{'MemoryOnlyPair' value is unused}}

  # CHECK-NEXT: %regX = lit.varlet.decl
  # CHECK-NEXT: [[AX:%.*]] = lit.struct.gep %a[x]
  # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}(%regX, [[AX]])
  let regX = a.x

  # CHECK-NEXT: [[AX:%.*]] = lit.struct.gep %a[x]
  # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}([[AX]], %regX)
  a.x = regX

  # Pass memory primary things by value as arguments.

  # CHECK-NEXT: [[TMPPAIR:%.*]] = lit.varlet.decl {{.*}}@MemoryOnlyPair
  # CHECK-NEXT: kgen.call @{{.*}}@"__copyinit__{{.*}}"([[TMPPAIR]], %a)
  # CHECK-NEXT: [[TMPINT:%.*]] = lit.varlet.decl {{.*}}@MemoryOnlyInt
  # CHECK-NEXT: kgen.call @{{.*}}@"__copyinit__{{.*}}"([[TMPINT]], %regX)
  # CHECK-NEXT: kgen.call @{{.*}}@"method{{.*}}"([[TMPPAIR]], [[TMPINT]])
  a.method(regX)

  # Drill into rvalue without cloning intermediate values.
  # CHECK-NEXT: [[V2X:%.*]] = lit.struct.gep %v2[x]
  # CHECK-NEXT: [[V2XX:%.*]] = lit.struct.gep [[V2X]][x]
  # CHECK-NEXT: [[VAL:%.*]] = pop.load [[V2XX]] : !pop.pointer<{{.*}}@"$Int"::@Int>
  # CHECK-NEXT: lit.letreg.decl "v2xx" = [[VAL]]
  let v2xx = v2.x.x

  # Implicit conversion between memory-only types.
  # CHECK-NEXT: %mpFloat = lit.varlet.decl
  # CHECK-NEXT: [[V2X:%.*]] = lit.struct.gep %v2[x]
  # CHECK-NEXT: kgen.call {{.*}}__init__{{.*}}(%mpFloat, [[V2X]])
  let mpFloat : MemoryOnlyFloat64 = v2.x

  # CHECK: [[TMP:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK-NEXT: kgen.call @{{.*}}inferred_function_with_memory_result{{.*}}([[TMP]]
  _ = inferred_function_with_memory_result(SIMD[DType.float32, 4]())

  # Memory-only default argument with memory-only result.
  # CHECK-NEXT: [[TMP:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK-NEXT: %[[C42:.*]] = {{.*}}constant: {{.*}}Int = {{.*}} 42
  # CHECK-NEXT: kgen.call @{{.*}}__init__{{.*}}([[TMP]], %[[C42]])
  _ = MemoryOnlyInt()

  # CHECK-NEXT: [[VARIADIC:%.*]]  = pop.variadic.create [%regX, %regX]
  # CHECK-NEXT: kgen.call @{{.*}}variadic{{.*}}([[VARIADIC]])
  MemoryOnlyInt.variadic(regX, regX)
  # CHECK-NEXT: lit.ownership.use %regX : !pop.pointer<@"$expressions"::@MemoryOnlyInt>
  # CHECK-NEXT: lit.ownership.use %regX

  # CHECK-NEXT: kgen.call {{.*}}__copyinit__{{.*}}(%__result__, %v2)
  # CHECK-NEXT: [[NONEVAL:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
  # CHECK-NEXT: lit.return [[NONEVAL]]
  return v2

struct DummyFunc:
    fn __init__(inout self, f: def(Int)):
        pass

fn func_arg_conversion(f: DummyFunc): pass

# CHECK-LABEL: lit.func @"implicit_func_conversion()"
fn implicit_func_conversion():
    @noncapturing
    def take_int(x: Int):
        pass

    # CHECK: %0 = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}(%f, %0)
    var f: DummyFunc = take_int
    # CHECK: %2 = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}(%anonymous2A, %2)
    # CHECK: call {{.*}}func_arg_conversion{{.*}}(%anonymous2A)
    func_arg_conversion(take_int)

# CHECK-LABEL: lit.struct.decl @M
@register_passable
struct M:
  var value: Int
  # CHECK-LABEL: lit.func @"__init__
  fn __init__(value: Int) -> M:
  # CHECK-NEXT: %0 = lit.struct.create(value=%value) : (!kgen.declref<{{.*}}@"$Int"::@Int>) -> !kgen.declref<@"$expressions"::@M>
    return M{value: value}

  fn __copyinit__(self) -> M:
    # FIXME: Should generate an error.
    pass

  fn __neg__(self) -> M:
    return M(0)
  fn __add__(self, rhs: M) -> M:
    return M(0)
  fn __matmul__(self, rhs: M) -> M:
    return M(0)
  fn __rmatmul__(lhs, self: M) -> M:
    return M(0)

# CHECK-LABEL: lit.func @"simpleMath
fn simpleMath(a: Int, b: Int) -> Int:
  # CHECK: %0 = kgen.call {{.*}}@"$Int"::@Int::@"__mul__{{.*}}(%b, %a)
  # CHECK: %1 = kgen.call {{.*}}@"$Int"::@Int::@"__sub__{{.*}}(%a, %0)
  # CHECK: lit.return %1 : !kgen.declref<{{.*}}@"$Int"::@Int>
  return a-b*a

# CHECK-LABEL: lit.func @"precedence_associativity
fn precedence_associativity(a: Int):
  # CHECK: %z = lit.varlet.decl "z", var = true
  var z: Int = 0

  # CHECK: [[SEVENTEENINT:%.*]] = kgen{{.*}}#lit.struct<{value = 17}>
  # CHECK-NEXT: pop.store [[SEVENTEENINT]], %z
  z = 17  # Implicit conversion

  # CHECK-NEXT: %[[Z:.*]] = pop.load %z
  # CHECK-NEXT: %[[POW0:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT: %[[POW1:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0]])
  # CHECK-NEXT: pop.store %[[POW1]], %z
  z = 2**(a**z)
  # CHECK-NEXT: %[[Z:.*]] = pop.load %z
  # CHECK-NEXT: %[[POW0:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT: %[[POW1:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0]])
  # CHECK-NEXT: pop.store %[[POW1]], %z
  z = 2**a**z
  # CHECK-NEXT:  %[[Z:.*]] = pop.load %z
  # CHECK-NEXT:  %[[MUL:.*]] = kgen.param.constant: {{.*}}@"$Int"::@Int = <{{.*}} = -6}
  # CHECK-NEXT:  %[[ADD:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__add__{{.*}}(%[[Z]], %[[MUL]])
  # CHECK-NEXT:  pop.store %[[ADD]], %z
  z = z + 3 * -2
  # CHECK-NEXT:  %[[Z:.*]] = pop.load %z
  # CHECK-NEXT:  %[[FLOOR_DIV:.*]] = kgen.param.constant: {{.*}}@"$Int"::@Int = <{{.*}} = -2}
  # CHECK-NEXT:  %[[ADD:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__add__{{.*}}(%[[Z]], %[[FLOOR_DIV]])
  # CHECK-NEXT:  pop.store %[[ADD]], %z
  z = z + 3 // -2
  # CHECK-NEXT:  %[[Z:.*]] = pop.load %z
  # CHECK-NEXT:  %[[INT_THREE:.*]] = kgen{{.*}}#lit.struct<{value = 3}>
  # CHECK-NEXT:  %[[ADD:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__add__{{.*}}(%[[Z]], %[[INT_THREE]])
  # CHECK-NEXT:  %[[NEG:.*]] = kgen{{.*}}#lit.struct<{value = -2}>
  # CHECK-NEXT:  %[[MUL:.*]] =  kgen.call {{.*}}@"$Int"::@Int::@"__mul__{{.*}}(%[[ADD]], %[[NEG]])
  # CHECK-NEXT:  pop.store %[[MUL]], %z
  z = (z + 3) * -+2
  # CHECK-NEXT:  %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT:  %[[Z:.*]] = pop.load %z
  # CHECK-NEXT:  %[[POW:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[Z]])
  # CHECK-NEXT:  %[[NEG:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__neg__{{.*}}(%[[POW]])
  # CHECK-NEXT:  pop.store %[[NEG]], %z
  z = -2**z
  # CHECK-NEXT: [[Z:%.*]] = pop.load %z
  # CHECK-NEXT: [[ONE:%.*]] = kgen{{.*}}#lit.struct<{value = 1}>
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__radd__({{.*}}$Int::Int,{{.*}}$Int::Int)"([[Z]], [[ONE]])
  # CHECK-NEXT: pop.store [[RES]], %z
  z = (1).value + z

  # div tests
  # CHECK: kgen.call {{.*}}__truediv__
  var r0 = Float32(33.0) / Float32(42.0)

  # CHECK: kgen.call {{.*}}__truediv__
  var r1 = Float32(33.0) / 42.0

# CHECK-LABEL: lit.func @"reverse_operators
fn reverse_operators(a: Int):
  # CHECK: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__radd__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  var z = (1).value + a

  # CHECK: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__rsub__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  z = (2).value - z

  # CHECK: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__rmul__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  z = (3).value * z

  # div tests
  # CHECK: kgen.call {{.*}}__rtruediv__
  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rfloordiv__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  var r1 = 33.0 / Float32(42.0)
  z = (33).value // z

  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rmod__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  var i0 = (10).value % z

# CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rpow__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  var i1 = (3).value ** z

  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rlshift__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  var i2 = (1).value << z

  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rrshift__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  var i3 = (1).value >> z

  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rand__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  z = (1).value & z

  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__ror__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  z = (2).value | z

  # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__rxor__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  z = (3).value ^ z

# CHECK-LABEL: lit.func @"precedence_matmul
fn precedence_matmul(z: M) -> M:
  # CHECK-NEXT:  %[[THREE:.*]] = kgen.param.constant: {{.*}}Int = {{.*}} 3
  # CHECK-NEXT:  %[[INT_THREE:.*]] = kgen.call @"$expressions"::@M::@"__init__{{.*}}(%[[THREE]])
  # CHECK-NEXT:  %[[TWO:.*]] = kgen.param.constant: {{.*}}Int = {{.*}} 2
  # CHECK-NEXT:  %[[INT_TWO:.*]] = kgen.call @"$expressions"::@M::@"__init__{{.*}}(%[[TWO]])
  # CHECK-NEXT:  %[[NEG:.*]] = kgen.call @"$expressions"::@M::@"__neg__{{.*}}(%[[INT_TWO]])
  # CHECK-NEXT:  %[[MATMUL:.*]] = kgen.call @"$expressions"::@M::@"__matmul__{{.*}}(%[[INT_THREE]], %[[NEG]])
  # CHECK-NEXT:  %[[ADD:.*]] = kgen.call @"$expressions"::@M::@"__add__{{.*}}(%z, %[[MATMUL]])
  # CHECK-NEXT:  lit.return %[[ADD]] : !kgen.declref<@"$expressions"::@M>
  return z + M(3) @ -M(2)

# CHECK-LABEL: lit.func @"precedence_bitwise
fn precedence_bitwise(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT: %[[MUL:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__mul__{{.*}}(%a, %[[INT_TWO]])
  # CHECK-NEXT: %[[AND:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__and__{{.*}}(%[[MUL]], %b)
  # CHECK-NEXT: %[[INT_FOUR:.*]] = kgen{{.*}}#lit.struct<{value = 4}>
  # CHECK-NEXT: %[[XOR:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__xor__{{.*}}(%[[INT_FOUR]], %c)
  # CHECK-NEXT: %[[OR:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__or__{{.*}}(%[[AND]], %[[XOR]])
  # CHECK-NEXT: lit.return %[[OR]]
  return a * 2 & b | 4 ^ c

# CHECK-LABEL: @"comparisons
fn comparisons(a: Int, b: Int):
   var res: Bool
   # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__lt__{{.*}}(%a, %b)
   res = a < b
   # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__le__{{.*}}(%a, %b)
   res = a <= b
   # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__gt__{{.*}}(%a, %b)
   res = a > b
   # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__ge__{{.*}}(%a, %b)
   res = a >= b
   # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__eq__{{.*}}(%a, %b)
   res = a == b
   # CHECK: kgen.call {{.*}}@"$Int"::@Int::@"__ne__{{.*}}(%a, %b)
   res = a != b

@register_passable
struct Boolish:
  fn __copyinit__(self) -> Self: pass
  fn __bool__(self) -> Bool: return True

struct MemBoolish:
  fn __init__(inout self, value: Boolish): pass
  fn __copyinit__(inout self, existing: Self): pass
  fn __bool__(self) -> Bool: return True

# CHECK-LABEL: @"unary
fn unary(a: Bool, b: Int, c: Boolish, d: MemBoolish):
  # CHECK: %0 = kgen.call {{.*}}@"$Bool"::@Bool::@"__bool__({{.*}}$Bool::Bool)"(%a)
  # CHECK: %1 = kgen.call {{.*}}@"$Bool"::@Bool::@"__invert__({{.*}}$Bool::Bool)"(%0)
  _ = not a

  # CHECK: [[EQ:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__eq__({{.*}}$Int::Int,{{.*}}$Int::Int)"
  # CHECK: [[EQBOOL:%.*]] = kgen.call {{.*}}@"$Bool"::@Bool::@"__bool__({{.*}}$Bool::Bool)"([[EQ]])
  # CHECK:  = kgen.call {{.*}}@"$Bool"::@Bool::@"__invert__({{.*}}$Bool::Bool)"([[EQBOOL]])
  _ = not b == 0

  # CHECK: [[BOOL:%.*]] = kgen.call {{.*}}__bool__{{.*}}(%c)
  # CHECK:  = kgen.call {{.*}}@"$Bool"::@Bool::@"__invert__({{.*}}$Bool::Bool)"([[BOOL]])
  _ = not c

  # CHECK: [[BOOL:%.*]] = kgen.call {{.*}}@"__bool__{{.*}}(%d)
  # CHECK-NEXT: kgen.call {{.*}}__invert__{{.*}}([[BOOL]])
  _ = not d

# CHECK-LABEL: lit.func @"andOr
fn andOr(a: Boolish, b: Boolish, c: Bool, d: MemBoolish):
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: [[BOOL:%.*]] = kgen.call {{.*}}__bool__{{.*}}(%a)
  # CHECK: [[I1:%.*]] = kgen.call {{.*}}__mlir_i1__{{.*}}([[BOOL]])
  # CHECK: hlcf.if [[I1]] -> !kgen.declref<@"$expressions"::@Boolish> {
  # CHECK:   [[TMP:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%b)
  # CHECK:   hlcf.yield [[TMP]]
  # CHECK: } else {
  # CHECK:   [[TMP:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK:   hlcf.yield [[TMP]]
  # CHECK: }
  _ = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.  Boolish is defined with copy ctor so it must be invoked.

  # CHECK-NEXT: [[ABOOL:%.*]] = kgen.call @"$expressions"::@Boolish::@"__bool__($expressions::Boolish)"(
  # CHECK-NEXT: [[I1:%.*]] = kgen.call {{.*}}@Bool::@"__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !kgen.declref<@"$expressions"::@Boolish> {
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%b)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a or b

  # Testing two different logic'y types returns the common bool type if present.

  # CHECK-NEXT: [[ABOOL:%.*]] = kgen.call {{.*}}__bool__{{.*}}(%a)
  # CHECK-NEXT: [[I1:%.*]] = kgen.call {{.*}}__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !kgen.declref<{{.*}}@"$Bool"::@Bool> {
  # CHECK-NEXT:   hlcf.yield %c
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[ABOOL:%.*]] = kgen.call {{.*}}__init__{{.*}}([[I1]])
  # CHECK-NEXT:   hlcf.yield [[ABOOL]]
  # CHECK-NEXT: }
  _ = a and c

  # Check incompatible types that are nevertheless boolish.

  # CHECK-NEXT: [[BBOOL:%.*]] = kgen.call {{.*}}__bool__{{.*}}(%b)
  # CHECK-NEXT: [[BI1:%.*]] = kgen.call {{.*}}__mlir_i1__{{.*}}([[BBOOL]])
  # CHECK-NEXT: = hlcf.if [[BI1]] -> !kgen.declref<{{.*}}@"$Bool"::@Bool> {
  # CHECK-NEXT:    [[TMP:%.*]] = kgen.call {{.*}}@Bool::@"__init__{{.*}}([[BI1]])
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT:  } else {
  # CHECK-NEXT:    hlcf.yield %c
  # CHECK-NEXT:  }
  _ = b or c

  # Check memory-only boolish types.
  # Boolish and MemBoolish has a common type of MemBoolish.

  # CHECK-NEXT: [[DBOOL:%.*]] = kgen.call {{.*}}__bool__{{.*}}(%d)
  # CHECK-NEXT: [[DI1:%.*]] = kgen.call {{.*}}__mlir_i1__{{.*}}([[DBOOL]])
  # CHECK-NEXT: [[IFRESULT:%.*]] = lit.varlet.decl {{.*}} <@"$expressions"::@MemBoolish>
  # CHECK-NEXT: hlcf.if [[DI1]] {
  # CHECK-NEXT:   kgen.call {{.*}}__copyinit__{{.*}}([[IFRESULT]], %d)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMPMEM:%.*]] = lit.varlet.decl
  # CHECK-NEXT:   kgen.call {{.*}}__init__{{.*}}([[TMPMEM]], %b)
  # CHECK-NEXT:   kgen.call {{.*}}__copyinit__{{.*}}([[IFRESULT]], [[TMPMEM]])
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  _ = d or b

# CHECK-LABEL: lit.func @"paramAndOr{{.*}}()"
# CHECK-SAME: <[[A:.*]]: @"$expressions"::@Boolish, [[B:.*]]: @"$expressions"::@Boolish>
fn paramAndOr[a: Boolish, b: Boolish]():
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: lit.alias.decl {{.*}}c: @"$expressions"::@Boolish = <cond(apply(:<>(!kgen.declref<{{.*}}@"$Bool"::@Bool> borrow) -> i1 {{.*}}@"$Bool"::@Bool::@"__mlir_i1__({{.*}}$Bool::Bool)", apply(:<>(!kgen.declref<@"$expressions"::@Boolish> borrow) -> !kgen.declref<{{.*}}@"$Bool"::@Bool> @"$expressions"::@Boolish::@"__bool__($expressions::Boolish)", [[A]])), [[B]], [[A]])>
  alias c = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.

  # CHECK: lit.alias.decl {{.*}}d: @"$expressions"::@Boolish = <cond(apply(:<>(!kgen.declref<{{.*}}@"$Bool"::@Bool> borrow) -> i1 {{.*}}@"$Bool"::@Bool::@"__mlir_i1__({{.*}}$Bool::Bool)", apply(:<>(!kgen.declref<@"$expressions"::@Boolish> borrow) -> !kgen.declref<{{.*}}@"$Bool"::@Bool> @"$expressions"::@Boolish::@"__bool__($expressions::Boolish)", [[A]])), [[A]], [[B]])>
  alias d = a or b

# CHECK-LABEL: lit.func @"do_math
fn do_math(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %z = lit.varlet.decl "z", var = true
  var z : Int
  # CHECK-NEXT: %[[INT_5:.*]] = kgen{{.*}}#lit.struct<{value = 5}>
  # CHECK-NEXT: %[[MUL:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__mul__{{.*}}(%[[INT_5]], %a)
  # CHECK-NEXT: %[[INT_42:.*]] = kgen{{.*}}#lit.struct<{value = 42}>
  # CHECK-NEXT: %[[ADD:.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__add__{{.*}}(%[[INT_42]], %[[MUL]])
  # CHECK-NEXT: pop.store %[[ADD]], %z
  z = 42 + 5*a

  # CHECK-NEXT: %x = lit.varlet.decl "x", var = true
  # CHECK-NEXT: [[TMP:%.*]] = pop.load %z
  # CHECK-NEXT: pop.store [[TMP]], %x
  # This is checking the lexer handles \ at end of line correctly.
  var x : Int
  x = \
z

  # CHECK-NEXT: kgen.call @"$expressions"::@"noop()"()
  noop()

  # CHECK-NEXT: [[TMP:%.*]] = pop.load %x
  # CHECK-NEXT: lit.return [[TMP]]
  return x

# CHECK-LABEL: lit.func @"listValues()"
fn listValues():
  # CHECK: %[[LIST:.*]] = kgen.call {{.*}}@ListLiteral::@"__init__
  # CHECK: pop.store %[[LIST:.*]], %a
  var a = [1, 2, 2+1]
  # CHECK: %[[LIST:.*]] = kgen.call {{.*}}@ListLiteral::@"__init__
  # CHECK: pop.store %[[LIST:.*]], %a
  a = [1, 2, 2+1,]
  # CHECK: %[[LIST:.*]] = kgen.call {{.*}}@ListLiteral::@"__init__
  # CHECK: pop.store %[[LIST:.*]], %a
  a = [1, 2, 2+1]
  # CHECK: %[[LIST:.*]] = kgen.call {{.*}}@ListLiteral::@"__init__
  # CHECK: pop.store %[[LIST:.*]], %b
  var b = []

# CHECK-LABEL: lit.func @"initializers
fn initializers():
  # CHECK: %0 = kgen.param.constant: {{.*}}@"$Int"::@Int = <#lit.struct<{value = 42}>>
  # CHECK: lit.letreg.decl "a" = %0
  let a = Int{value: (42).value}

  # Issue #7343: Trailing comma ok too.
  _ = Int{value: (42).value,}

  # Issue #12067, suffix stuff ok.
  _ = Int{ value: (1).value }.value

# CHECK-LABEL: lit.func @"test_if_cond
fn test_if_cond(owned cond: Bool, memCond: MemBoolish):
    # CHECK: %i = lit.varlet.decl "i"
    # CHECK: %[[COND:.*]] = pop.load %cond_0
    # CHECK: %[[LIT_BOOLI1:.*]] = kgen.call {{.*}}__mlir_i1__{{.*}}(%[[COND]])
    # CHECK-NEXT: %[[IF_RES:.*]] = hlcf.if %[[LIT_BOOLI1]]
    # CHECK-NEXT:   %[[INT_TWO:.*]] = kgen{{.*}}= 2}
    # CHECK-NEXT:   hlcf.yield %[[INT_TWO]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %[[INT_THREE:.*]] = kgen{{.*}}= 3}
    # CHECK-NEXT:   hlcf.yield %[[INT_THREE]]
    # CHECK-NEXT: }
    # CHECK-NEXT: pop.store %[[IF_RES]], %i
    var i: Int = 2 if cond else 3

    # CHECK: [[TRUEB:%.+]] = kgen{{.*}}= true}
    # CHECK-NEXT: pop.store [[TRUEB]], %cond
    cond = True
    i += i
    if cond:     # 'if' stmt, not an 'if' expression.
        i += 1

# CHECK-LABEL: lit.func @"test_param_if_cond{{.*}}()"
# CHECK-SAME: <[[COND:.*]]: {{.*}}@"$Bool"::@Bool>
fn test_param_if_cond[cond: Bool]() -> Int:
  # CHECK: lit.alias.decl [[I_ALIAS:.*]]: {{.*}}@"$Int"::@Int = <cond(apply(:<>(!kgen.declref<{{.*}}@"$Bool"::@Bool> borrow) -> i1 {{.*}}@"$Bool"::@Bool::@"__mlir_i1__({{.*}}$Bool::Bool)", [[COND]]), #lit.struct<{value = 2}>, #lit.struct<{value = 3}>)>
  alias i = 2 if cond else 3

  # CHECK-NEXT: lit.alias.decl {{.*}}j: {{.*}}@"$FloatLiteral"::@FloatLiteral = <cond(apply(:<>(!kgen.declref<{{.*}}@"$Bool"::@Bool> borrow) -> i1 {{.*}}@"$Bool"::@Bool::@"__mlir_i1__({{.*}}$Bool::Bool)", [[COND]]), #lit.struct<{value: scalar<f64> = "2"}>, #lit.struct<{value: scalar<f64> = "3"}>)>
  alias j = 2.0 if cond else 3

  # CHECK-NEXT: %[[I:.*]] = kgen.param.constant: {{.*}}@"$Int"::@Int = <[[I_ALIAS]]>
  return i

# CHECK-LABEL: lit.func @"callable_mv[fn({{.*}}$Int::Int) -> {{.*}}$Int::Int]({{.*}}$Int::Int)"
# CHECK-SAME: <[[CALLABLE:.*]]: <>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>>(%a: !kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>
fn callable_mv[callable: fn (Int) -> Int](a: Int) -> Int:
  # CHECK-NEXT: kgen.call_param[<>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>: [[CALLABLE]]](%a)
  return callable(a)

# CHECK-LABEL: lit.func @"callable_mv_inputs{{.*}})"<
# CHECK-SAME: [[CALLABLE:.*]]: <{{.*}}@"$Int"::@Int>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>, [[B:.*]]: {{.*}}@"$Int"::@Int>(%a: !kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>
fn callable_mv_inputs[callable: fn[x: Int](Int) -> Int, b: Int](a: Int) -> Int:
  # CHECK-NEXT: kgen.call_param[<>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>: bind_signature(:<{{.*}}@"$Int"::@Int>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int> [[CALLABLE]], [[B]])](%a)
  return callable[b](a)

# CHECK-LABEL: lit.func @"takeIndexParam{{.*}}"<{{.*}}a: {{.*}}@"$Int"::@Int>() -> !kgen.declref<{{.*}}@"$Int"::@Int>
fn takeIndexParam[a: Int]() -> Int:
  return a + 1

# CHECK-LABEL: lit.func @"returnIndex()"() -> !kgen.declref<{{.*}}@"$Int"::@Int>
fn returnIndex() -> Int:
  return 0

# CHECK-LABEL: lit.func @"returnIndex2()"() -> !kgen.declref<{{.*}}@"$Int"::@Int>
fn returnIndex2() -> Int:
  # CHECK-NEXT: %0 = kgen.call @"$expressions"::@"takeIndexParam{{.*}}"<:{{.*}}@"$Int"::@Int apply(:() -> !kgen.declref<{{.*}}@"$Int"::@Int> @"$expressions"::@"returnIndex()")>() : () -> !kgen.declref<{{.*}}@"$Int"::@Int>
  # CHECK-NEXT: return %0
  return takeIndexParam[returnIndex()]()

# CHECK-LABEL: lit.func @"callInParam[fn[{{.*}}$Int::Int]({{.*}}$Int::Int) -> {{.*}}$Int::Int]()"
# CHECK-SAME: <[[CALLABLE:.*]]: <{{.*}}@"$Int"::@Int>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int>>() -> !kgen.declref<{{.*}}@"$Int"::@Int>
fn callInParam[callable: fn[x: Int](Int) -> Int]() -> Int:
  # CHECK-NEXT: %0 = kgen.call @"$expressions"::@"takeIndexParam{{.*}}()"<:{{.*}}@"$Int"::@Int apply(:<>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int> bind_signature(:<{{.*}}@"$Int"::@Int>(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) -> !kgen.declref<{{.*}}@"$Int"::@Int> [[CALLABLE]], #lit.struct<{value = 1}>), #lit.struct<{value = 1}>)>() : () -> !kgen.declref<{{.*}}@"$Int"::@Int>
  # CHECK-NEXT: return %0
  return takeIndexParam[callable[1](1)]()

# CHECK-LABEL: lit.func @"parameterExprs{{.*}}()"
# CHECK-SAME: <[[A:.*]]: {{.*}}@"$Int"::@Int, [[A2:.*]]: {{.*}}@"$Int"::@Int>
fn parameterExprs[a: Int, a2: Int]():
  # CHECK: lit.alias.decl {{.*}}b: {{.*}}@Int = <apply({{.*}}__sub__{{.*}}, [[A]], [[A]])>
  alias b = a-a
  # CHECK: lit.alias.decl {{.*}}c: {{.*}}@Int = <apply({{.*}}__add__{{.*}}, [[A]], {{.*}}42{{.*}})>
  alias c = a+42
  # CHECK: lit.alias.decl {{.*}}d: {{.*}}@Int = <apply({{.*}}__mul__{{.*}}, [[A]], [[A2]])>
  alias d = a*a2

##===----------------------------------------------------------------------===##
# Patterns, LValues and RValues
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"patterns()
fn patterns():
  # CHECK: %z2 = lit.varlet.decl "z2", var = true
  var z2: Int

  (((z2))) = 42  # Paren patterns
  # CHECK: [[TMP:%.*]] = {{.*}}constant{{.*}} 42
  # CHECK: pop.store [[TMP]], %z2

  var someInt : Int
  (someInt) += someInt
  # CHECK: %someInt = lit.varlet.decl "someInt", var = true
  # CHECK:  %1 = pop.load %someInt
  # CHECK:  %2 = kgen.call {{.*}}@"$Int"::@Int::@"__iadd__{{.*}}(%someInt, %1)

  # Discard pattern with different types.
  (_) = someInt
  # CHECK: [[TMP:%.*]] = pop.load %someInt

  (_) = 1.0

  # CHECK: %someFloat32 = lit.varlet.decl "someFloat32", var = true
  # CHECK: [[Float32:%.*]] = pop.load %someFloat32
  # CHECK: {{%.*}} = kgen.call {{.*}}__iadd__{{.*}}(%someFloat32, [[Float32]])
  var someFloat32 : Float32
  (someFloat32) += someFloat32

  # CHECK: %someSIMD = lit.varlet.decl "someSIMD", var = true
  # CHECK: [[SIMD:%.*]] = pop.load %someSIMD
  # CHECK: {{%.*}} = kgen.call @"$SIMD"::@SIMD::@"__iadd__{{.*}}(%someSIMD, [[SIMD]])
  var someSIMD : SIMD[DType.float64, 4]
  (someSIMD) += someSIMD

# CHECK-LABEL: lit.func @"byval_byref_function({{.*}}$Int::Int,{{.*}}$Int::Int&)"(%a: !kgen.declref<{{.*}}@"$Int"::@Int> borrow, %b: !pop.pointer<{{.*}}@"$Int"::@Int> byref) -> !lit.none
fn byval_byref_function(a: Int, inout b: Int):
  # CHECK-NEXT: pop.store %a, %b
  b = a

  # CHECK-NEXT: %x = lit.varlet.decl "x", var = true
  var x : Int
  # This needs to load 'b' to pass it by value for the first arg, but pass its
  # address in directly for the second.
  # CHECK: %0 = pop.load %b
  # CHECK: = kgen.call @{{.*}}::@"byval_byref_function{{.*}}(%0, %b)
  byval_byref_function(b, b)

# CHECK-LABEL: lit.func @"lvaluesAndRValues()
fn lvaluesAndRValues() -> __mlir_type.index:
  # CHECK: [[VALUE:%.*]] = index.constant 4
  # CHECK: lit.return [[VALUE]] : index
  return (4).value

# CHECK-LABEL: lit.func @"mvalueStructField()"
fn mvalueStructField():
  # CHECK: lit.alias.decl [[INT:.*]]: {{.*}}@"$Int"::@Int = <#lit.struct<{value = 4}>>
  alias int = 4
  # CHECK: lit.alias.decl {{.*}}value = <#lit.struct.extract<:{{.*}}@"$Int"::@Int [[INT]], "value">>
  alias value = int.value
  alias foldToValue = (5).value

# CHECK-LABEL: lit.func @"defTests({{.*}}, %untyped: !pop.pointer<@"$Object"::@object> owned_in_mem)
def defTests(a: Int, b: Int, untyped) -> None:
  # CHECK: [[B:%.*]] = pop.load %b_1
  # CHECK-NEXT: pop.store [[B]], %a_0
  a = b # Parameters are mutable!

##===----------------------------------------------------------------------===##
# Augmented Assignments
##===----------------------------------------------------------------------===##

def basic_assignments(a: Int, b: Int, c: M, d: M):
  # CHECK:      %a_0 = lit.varlet.decl "a", var = true
  # CHECK:      %b_1 = lit.varlet.decl "b", var = true
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__iadd__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a += b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__isub__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a -= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__imul__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a *= b
  # HECK:      [[LOAD_C:%.*]] = pop.load %c_2  : !pop.pointer<@M>
  # HECK-NEXT: [[RES:%.*]] = kgen.call @M::@"__imatmul__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%d_3, [[LOAD_C]])
  #d @= c
  # HECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # HECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__itruediv__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  #a /= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__ifloordiv__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a //= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__imod__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a %= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__ipow__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a **= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__irshift__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a >>= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__ilshift__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a <<= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__iand__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a &= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__ixor__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a ^= b
  # CHECK:      [[LOAD_B:%.*]] = pop.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = kgen.call {{.*}}@"$Int"::@Int::@"__ior__({{.*}}$Int::Int&,{{.*}}$Int::Int)"(%a_0, [[LOAD_B]])
  a |= b

  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}value = 4
  # CHECK-NEXT: pop.store [[FOUR]], %b_1
  # CHECK-NEXT: pop.store [[FOUR]], %a_0
  a = b = 4

  # Walrus
  # CHECK-NEXT: [[SEVEN:%.*]] = kgen.param.constant: {{.*}}value = 7
  # CHECK-NEXT: pop.store [[SEVEN]], %b_1
  # CHECK-NEXT: [[A:%.*]] = pop.load %a_0
  # CHECK-NEXT: kgen.call {{.*}}simpleMath{{.*}}([[A]], [[SEVEN]])
  simpleMath(a, b := 7)

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
    a = 00__0_0       # CHECK: 0
    a = 1__9_         # CHECK: 19
    a = 0x123         # CHECK: 291
    a = 0X123         # CHECK: 291
    a = 0b10101       # CHECK: 21
    a = 0B10101       # CHECK: 21
    a = 0o711         # CHECK: 457
    a = 0O711         # CHECK: 457
    b = 1.1           # CHECK: "1.10000{{.*}}"
    b = .1            # CHECK: "0.10000{{.*}}"
    b = 1.            # CHECK: "1"
    b = 1e2           # CHECK: "100"
    b = 1.1e2         # CHECK: "110"
    b = .1e2          # CHECK: "10"
    b = 1.e2          # CHECK: "100"
    b = 1e+2          # CHECK: "100"
    b = 1.1e-2        # CHECK: "0.01099{{.*}}"
    b = .1e+2         # CHECK: "10"
    b = 1.e-2         # CHECK: "0.01"
    b = 0.1           # CHECK: "0.100000{{.*}}"
    b = 0.            # CHECK: "0"
    b = 0e2           # CHECK: "0"
    b = 0.1e2         # CHECK: "10"
    b = 0.e2          # CHECK: "0"
    b = 0e+2          # CHECK: "0"
    b = 0.1e-2        # CHECK: "0.001"
    b = 0.e-2         # CHECK: "0"
    b = 12.31e+11     # CHECK: "1.231E+12"
    b = 1_2.3__1e+1_1 # CHECK: "1.231E+12"
    b = 12.31E-3      # CHECK: "0.01231"
    c = False         # CHECK: @Bool = <#lit.struct<{value: scalar<bool> = false}>>
    c = True          # CHECK: @Bool = <#lit.struct<{value: scalar<bool> = true}>>

# CHECK-LABEL: lit.func @"strings
fn strings():
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
        # CHECK: string = "123"
        return "123"
        # lit.end_func
    """other comment"""


##===----------------------------------------------------------------------===##
# Tuples
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"tuples_rv
fn tuples_rv(a: Int, b: Float32):
    # CHECK: [[PACK0:%.*]] = kgen.param.constant: !pop.pack<[]> = <<>>
    # CHECK: kgen.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK0]])
    _ = ()

    # CHECK: [[PACK1:%.*]] = pop.pack.create(%a, %b)
    # CHECK: kgen.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK1]])
    _ = (a, b)

    # CHECK: [[PACK1:%.*]] = pop.pack.create(%a, %b)
    # CHECK: kgen.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK1]])
    _ = a, b

    # CHECK: [[PACK2:%.*]] = pop.pack.create(%a)
    # CHECK: kgen.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK2]])
    _ = (a,)

    # CHECK: [[PACK2:%.*]] = pop.pack.create(%a)
    # CHECK: kgen.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK2]])
    _ = a,

    # CHECK: %c = lit.varlet.decl "c"
    # CHECK: [[PACK2:%.*]] = pop.pack.create(%a)
    # CHECK: [[TUP2:%.*]] = kgen.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK2]])
    # CHECK: pop.store [[TUP2]], %c
    var c = a,

# CHECK-LABEL: lit.func @"tuples_lv
fn tuples_lv(i0: Int, f0: Float32):
   var i1 = 1
   var i2 = 2

   # CHECK: %iTup = lit.varlet.decl "iTup"
   var iTup : Tuple[Int, Int]

   # Tuple Rvalue
   # CHECK: [[TUP:%.*]] = kgen.call {{.*}}@Tuple::@"__init__
   # CHECK: pop.store [[TUP]], %iTup
   iTup = (i1, i2)

   # Tuple LValue
   # CHECK: [[TUP:%.*]] = pop.load %iTup
   # CHECK: [[TUP2:%.*]] = kgen.call {{.*}}@"__copyinit__{{.*}}([[TUP]])
   # CHECK: [[ELT:%.*]] = kgen.call {{.*}}Tuple::@"get{{.*}}([[TUP2]])
   # CHECK-NEXT: pop.store [[ELT]], %i1
   # CHECK: [[ELT:%.*]] = kgen.call {{.*}}Tuple::@"get{{.*}}([[TUP2]])
   # CHECK-NEXT: pop.store [[ELT]], %i2
   (i1, i2) = iTup

   # Check that the swap idiom is correct, this requires producing a copy of the
   # whole RValue on the right before extracting from it.

   # CHECK: [[I2VAL:%.*]] = pop.load %i2
   # CHECK-NEXT: [[I1VAL:%.*]] = pop.load %i1
   # CHECK-NEXT: [[PACK:%.*]] = pop.pack.create([[I2VAL]], [[I1VAL]])
   # CHECK-NEXT: [[TUPRV:%.*]] = kgen.call {{.*}}__init__{{.*}}([[PACK]])
   # CHECK-NEXT: [[I1VAL:%.*]] =  kgen.call {{.*}}Tuple::@"get{{.*}}({{.*}} = 0{{.*}}([[TUPRV]])
   # CHECK-NEXT: pop.store [[I1VAL]], %i1
   # CHECK-NEXT: [[I2VAL:%.*]] =  kgen.call {{.*}}Tuple::@"get{{.*}}({{.*}} = 1{{.*}}([[TUPRV]])
   # CHECK-NEXT: pop.store [[I2VAL]], %i2
   (i1, i2) = (i2, i1)

   var f1 : Float32 = 1
   # Mixed element types should work.  Don't need check lines though.
   (i1, f1) = (i0, f0)



##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

struct WeirdArray:
  fn __getitem__(self, x: Int) -> Int:
    return 1
  fn __getitem__(self, x: Int, y: Int) -> Int:
    return 2
  fn __getitem__(self, x: Int, y: Int, z: Int) -> Int:
    return 3
  fn __getitem__(self, x: Float32, *ints: Int) -> Float32:
    return x

 fn __setitem__(self, x: Int, y: Int, value: Int): pass

# CHECK-LABEL: lit.func @"testWeirdArray
fn testWeirdArray(a: WeirdArray, idx: Int, f: Float32):
  # CHECK: kgen.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
  _ = a[idx]
  # CHECK: kgen.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx)
  _ = a[idx, idx]
  # CHECK: kgen.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx, %idx)
  _ = a[idx, idx, idx]
  # CHECK: [[VARIADIC:%.*]] = pop.variadic.create [%idx, %idx, %idx, %idx]
  # CHECK: kgen.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %f, [[VARIADIC]])
  _ = a[f, idx, idx, idx, idx]

  # CHECK: [[SEVENTEEN:%.*]] = kgen.param.constant: {{.*}} = 17
  # CHECK: kgen.call {{.*}}__setitem__{{.*}}(%a, %idx, %idx, [[SEVENTEEN]])
  a[idx, idx] = 17


struct Slicable:
    fn __init__(inout self):
        pass

    fn __getitem__(self, s: slice):
        pass

# CHECK-LABEL: lit.func @"slice_expression
fn slice_expression(a: Slicable, i: Int):
  # CHECK: %[[I0:.*]] = kgen{{.*}}none
  # CHECK: %[[I1:.*]] = kgen{{.*}}none
  # CHECK: %[[I2:.*]] = kgen{{.*}}none
  # CHECK-NEXT: call {{.*}}@slice::@"__init__{{.*}}"<{{.*}}>(%[[I0]], %[[I1]], %[[I2]])
  # CHECK-NEXT: call {{.*}}__getitem__
  a[:]
  # CHECK: %[[I0:.*]] = kgen{{.*}}none
  # CHECK: %[[I1:.*]] = kgen{{.*}}none
  # CHECK: %[[I2:.*]] = kgen{{.*}}none
  # CHECK-NEXT: call {{.*}}@slice::@"__init__{{.*}}"<{{.*}}>(%[[I0]], %[[I1]], %[[I2]])
  # CHECK-NEXT: call {{.*}}__getitem__
  a[::]
  # CHECK: %[[I0:.*]] = kgen{{.*}} 1
  # CHECK: %[[I2:.*]] = kgen{{.*}}none
  # CHECK-NEXT: call {{.*}}@slice::@"__init__{{.*}}"<{{.*}}>(%[[I0]], %i, %[[I2]])
  # CHECK-NEXT: call {{.*}}__getitem__
  a[1:i]
  # CHECK: %[[C2:.*]] = kgen{{.*}} 2
  # CHECK: %[[I1:.*]] = {{.*}}@Int::@"__add__{{.*}}"(%[[C2]], %i)
  # CHECK: %[[I0:.*]] = kgen{{.*}}none
  # CHECK: %[[I2:.*]] = kgen{{.*}} 3
  # CHECK-NEXT: call {{.*}}@slice::@"__init__{{.*}}"<{{.*}}>(%[[I0]], %[[I1]], %[[I2]])
  # CHECK-NEXT: call {{.*}}__getitem__
  a[:2+i:3]


# This is an array that has elements of MemoryOnlyInt.
struct MemoryOnlyIntArray:
  fn __getitem__(inout self, x: Int) -> MemoryOnlyInt: pass
  fn __setitem__(inout self, x: Int, owned value: MemoryOnlyInt): pass

# CHECK-LABEL: lit.func @"testMemoryOnlyIntArray
fn testMemoryOnlyIntArray(inout arr: MemoryOnlyIntArray, x: Int, owned moi: MemoryOnlyInt):
  # CHECK: %0 = lit.ownership.end.lifetime %moi
  # CHECK: kgen.call {{.*}}__setitem__{{.*}}(%arr, %x, %0)
  arr[x] = moi^
  # CHECK: [[ANON:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK: kgen.call {{.*}}__getitem__{{.*}}([[ANON]], %arr, %x)
  # CHECK: kgen.call {{.*}}__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x] = arr[x]

  # CHECK: [[ANON:%.*]] = lit.varlet.decl "__store_tmp__"
  # CHECK: kgen.call {{.*}}__getitem__{{.*}}([[ANON]], %arr, %x)
  # CHECK: [[XP:%.*]] = lit.struct.gep %__store_tmp__[x]
  # CHECK: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK: pop.store %[[C1:.*]], [[XP]]
  # CHECK: kgen.call {{.*}}__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x].x = 1

  # Initialize in memory through a temp + setitem.
  # CHECK: [[ANON:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK: kgen.call @"{{.*}}__init__{{.*}}([[ANON]],
  # CHECK: kgen.call {{.*}}"__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x] = MemoryOnlyInt(42)

  # CHECK: [[STORETMP:%.*]] = lit.varlet.decl "__store_tmp__"
  # CHECK: kgen.call {{.*}}__getitem__{{.*}}([[STORETMP]], %arr, %x)
  # CHECK: [[XP:%.*]] = lit.struct.gep [[STORETMP]][x]
  # CHECK:  pop.store {{.*}}, [[XP]]
  # CHECK: kgen.call {{.*}}__setitem__{{.*}}(%arr, %x, [[STORETMP]])
  arr[x].x += 1


# Check a load from a SIMD field works.
# CHECK-LABEL: lit.func @"testSIMDGetter
fn testSIMDGetter[type: DType](owned a: SIMD[type, 2]) -> __mlir_type[
    `!pop.scalar<`, type.value, `>`]:
  # CHECK: %a_0 = lit.varlet.decl "a"
  # CHECK: pop.store %a, %a_0
  # CHECK: %0 = pop.load %a_0
  # CHECK: %1 = kgen.param.constant: {{.*}} = 0
  # CHECK: %2 = kgen.call {{.*}}__getitem__{{.*}}(%0, %1)
  # CHECK: %3 = lit.struct.extract %2[value]
  # CHECK: lit.return %3
  return a[0].value



struct MyInlineIntInit:
    var intVal: MemoryOnlyInt
    # CHECK-LABEL: lit.func @"__init__($expressions::MyInlineIntInit=&,$expressions::MemoryOnlyInt)"
    # CHECK-SAME: (%self: !pop.pointer<@"$expressions"::@MyInlineIntInit> init_self, %intVal: !pop.pointer<@"$expressions"::@MemoryOnlyInt> borrow_in_mem) -> !lit.none
    fn __init__(inout self, intVal: MemoryOnlyInt):
        # CHECK: %0 = lit.struct.gep %self[intVal]
        # CHECK: kgen.call {{.*}}__copyinit__{{.*}}(%0, %intVal)
        self.intVal = intVal

struct IndexArray:
  fn __getitem__(inout self, x: Int) -> Int: pass
  fn __setitem__(inout self, x: Int, value: Int): pass

struct IndexArrayArray:
  fn __getitem__(inout self, x: Int) -> IndexArray: pass
  fn __setitem__(inout self, x: Int, value: IndexArray): pass

fn takeInOutInt(inout a: Int): pass

 # CHECK-LABEL: lit.func @"testWritebacks
fn testWritebacks(inout a: IndexArray, inout b: IndexArrayArray):
  # CHECK: %anonymous2A = lit.varlet.decl "anonymous*", var = true
  # CHECK-NEXT: %[[V0:.*]] = {{.*}}constant{{.*}} = 0
  # CHECK-NEXT: %[[V1:.*]] = kgen.call {{.*}}__getitem__{{.*}}(%a, %[[V0]])
  # CHECK-NEXT: pop.store %[[V1]], %anonymous2A
  # CHECK-NEXT: %[[V2:.*]] = kgen.call {{.*}}takeInOutInt{{.*}}(%anonymous2A)
  # CHECK-NEXT: %[[V3:.*]] = {{.*}}constant{{.*}} = 0
  # CHECK-NEXT: %[[V4:.*]] = pop.load %anonymous2A
  # CHECK-NEXT: %[[V5:.*]] = kgen.call {{.*}}__setitem__{{.*}}(%a, %[[V3]], %[[V4]])
  takeInOutInt(a[0]);

  # CHECK: %anonymous2A_0 = lit.varlet.decl
  # CHECK: %anonymous2A_1 = lit.varlet.decl {{.*}}: <@"$expressions"::@IndexArray>
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: %[[V4:.*]] = {{.*}}__getitem__{{.*}}(%anonymous2A_1, %b, %[[C1]])
  # CHECK-NEXT: %[[C2:.*]] = {{.*}}constant{{.*}} = 2
  # CHECK-NEXT: %[[V5:.*]] = kgen.call {{.*}}__getitem__{{.*}}(%anonymous2A_1, %[[C2]])
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: %[[V6:.*]] = kgen.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], %anonymous2A_1)
  # CHECK-NEXT: pop.store %[[V5]], %anonymous2A_0
  # CHECK-NEXT: %[[V7:.*]] = kgen.call {{.*}}takeInOutInt{{.*}}(%anonymous2A_0)
  # CHECK-NEXT: %anonymous2A_2 = lit.varlet.decl {{.*}}
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: %[[V8:.*]] = kgen.call {{.*}}__getitem__{{.*}}(%anonymous2A_2, %b, %[[C1]])
  # CHECK-NEXT: %[[C2:.*]] = {{.*}}constant{{.*}} = 2
  # CHECK-NEXT: %[[V9:.*]] = pop.load %anonymous2A_0
  # CHECK-NEXT: %[[V10:.*]] = kgen.call {{.*}}__setitem__{{.*}}(%anonymous2A_2, %[[C2]], %[[V9]])
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: %[[V11:.*]] = kgen.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], %anonymous2A_2)
  takeInOutInt(b[1][2])


@register_passable
struct RegWeirdArray:
    fn __getitem__(self, idx: Int) -> Int:
        return idx
    fn __setitem__(self, idx: Int, value: Int):
        pass


# CHECK-LABEL: lit.func @"dlValueToPValue
fn dlValueToPValue[arr: RegWeirdArray]():
    # CHECK-NEXT: lit.alias.decl {{.*}}x: {{.*}}@Int = <apply({{.*}}@RegWeirdArray::@"__getitem__{{.*}}, {{.*}}arr, #lit.struct<{value = 2}>)>
    alias x = arr[2]


@register_passable
struct ConstDynamicObject:
    fn __init__() -> Self:
        return Self{}

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
    let const_obj = ConstDynamicObject()
    # CHECK: %[[KEY:.*]] = kgen.param.constant{{.*}} "dynamic_attribute"
    # CHECK: call {{.*}}@ConstDynamicObject::@"__getattr__{{.*}}"(%const_obj, %[[KEY]])
    _ = const_obj.dynamic_attribute

    var obj = DynamicObject()
    # CHECK: %[[KEY:.*]] = kgen.param.constant{{.*}} "some_attr"
    # CHECK: call {{.*}}@DynamicObject::@"__getattr__{{.*}}"(%obj, %[[KEY]])
    _ = obj.some_attr
    # CHECK: %[[KEY:.*]] = kgen.param.constant{{.*}} "some_attr"
    # CHECK: %[[VALUE:.*]] = kgen.param.constant{{.*}} 42
    # CHECK: call {{.*}}@DynamicObject::@"__setattr__{{.*}}(%obj, %[[KEY]], %[[VALUE]])
    obj.some_attr = 42


# CHECK-LABEL: lit.func @"chained_cmp
fn chained_cmp(a: Int, b: Int, c: Int, d: Int, e: Int):
    # CHECK-NEXT: %res = lit.varlet.decl "res"
    # CHECK:      [[CMP_A_B:%.*]] = kgen.call @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK-NEXT: %[[CMP_A_B_I1:.*]] = kgen.call @{{.*}}__mlir_i1__{{.*}}([[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.if %[[CMP_A_B_I1]]
    # CHECK-NEXT:   %[[CMP_B_C:.*]] = kgen.call @{{.*}}__lt__{{.*}}(%b, %c)
    # CHECK:        %[[IF_B_C:.*]] = hlcf.if
    # CHECK-NEXT:     %[[CMP_C_D:.*]] = kgen.call @{{.*}}__lt__{{.*}}(%c, %d)
    # CHECK-NEXT:     hlcf.yield %[[CMP_C_D]]
    # CHECK-NEXT:   } else {
    # CHECK-NEXT:     hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT:   }
    # CHECK-NEXT:   hlcf.yield %[[IF_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield [[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: pop.store %[[IF_A_B]], %res : !pop.pointer<{{.*}}@"$Bool"::@Bool>
    var res = a < b < c < d

    # COM: This checks the parsing precedence between `<` and `and`.
    # CHECK:      %[[CMP_A_B:.*]] = kgen.call @{{.*}}__lt__{{.*}}(%a, %b)
    # CHECK:       %[[CMP_A_B_I1:.*]] = kgen.call @{{.*}}__mlir_i1__{{.*}}(%[[CMP_A_B]])
    # CHECK-NEXT: %[[IF_A_B:.*]] = hlcf.if %[[CMP_A_B_I1]]
    # CHECK:   %[[CMP_B_C:.*]] = kgen.call @{{.*}}__lt__{{.*}}(
    # CHECK-NEXT:   hlcf.yield %[[CMP_B_C]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[CMP_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: %[[CMP_I1:.*]] = kgen.call @{{.*}}__mlir_i1__{{.*}}(%[[IF_A_B]])
    # CHECK-NEXT: %[[IF:.*]] = hlcf.if %[[CMP_I1]]
    # CHECK-NEXT:   %[[CMP_D_E:.*]] = kgen.call @{{.*}}__lt__{{.*}}(%d, %e)
    # CHECK-NEXT:   hlcf.yield %[[CMP_D_E]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   hlcf.yield %[[IF_A_B]]
    # CHECK-NEXT: }
    # CHECK-NEXT: pop.store %[[IF]], %res : !pop.pointer<{{.*}}@"$Bool"::@Bool>
    res = a < b < c and d < e

# CHECK-LABEL: lit.func @"foo_adaptive[{{.*}}$Int::Int](){{.*}} {isAdaptive
@adaptive
fn foo_adaptive[x: Int]() -> Int:
   return 0

# CHECK-LABEL: lit.func @"foo_adaptive[{{.*}}$Int::Int]()_0{{.*}} {isAdaptive
@adaptive
fn foo_adaptive[x: Int]() -> Int:
  return 1

# CHECK-LABEL: lit.func @"test_adaptive_set
fn test_adaptive_set():
    # CHECK: lit.alias.decl {{.*}}not_bound: variadic<!kgen.signature<<{{.*}}@"$Int"::@Int>() -> !kgen.declref<{{.*}}@"$Int"::@Int>>> =
    # CHECK-SAME: <[@"$expressions"::@"foo_adaptive[{{.*}}$Int::Int]()", @"$expressions"::@"foo_adaptive[{{.*}}$Int::Int]()_0"]>
    alias not_bound = foo_adaptive.__adaptive_set
    # CHECK-NEXT: lit.alias.decl {{.*}}bound: variadic<!kgen.signature<() -> !kgen.declref<{{.*}}@"$Int"::@Int>>> =
    # CHECK-SAME: <[@"$expressions"::@"foo_adaptive[{{.*}}$Int::Int]()"<:{{.*}}@"$Int"::@Int {{.*}}1{{.*}}>, @"$expressions"::@"foo_adaptive[{{.*}}$Int::Int]()_0"<:{{.*}}@"$Int"::@Int {{.*}}1{{.*}}>]>
    alias bound = foo_adaptive[1].__adaptive_set

fn lvalue_utilities(inout a: Int):
  # Get the address of the specified physical lvalue as a pop.pointer value.
  let addr : __mlir_type[`!pop.pointer<`,Int,`>`] = __get_lvalue_as_address(a)

  # Get and use an lvalue from an address.
  __get_address_as_lvalue(addr) = 42
  let val = __get_address_as_lvalue(addr)

struct CallableStruct:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __call__(self, rhs: Int) -> Int:
        return self.value + rhs

# CHECK-LABEL: lit.func @"test_call_method()"
fn test_call_method():
    # CHECK: %[[C2:.*]] = kgen.param.constant{{.*}} 2
    # CHECK-NEXT: kgen.call {{.*}}@"__call__{{.*}}"(%{{.*}}, %[[C2]])
    let value = CallableStruct(5)
    _ = value(2)

struct MemoryType:
  fn __copyinit__(inout self, existing: Self):
    pass

@register_passable
struct RegType: pass

# CHECK-LABEL: lit.struct.decl @ParamType
# CHECK-SAME: <[[A:.*]]: {{.*}}@"$Int"::@Int>
@register_passable
struct ParamType[a: Int]: pass

# CHECK-LABEL: lit.func @"function_types
# CHECK-SAME: %float0: {{.*}}(!kgen.declref<@"$Builtin"::@"$Int"::@Int> borrow) -> !kgen.declref<@"$Builtin"::@"$Int"::@Int>
# CHECK-SAME: %float1: {{.*}}(!pop.pointer<@"$expressions"::@MemoryType> byref_result, !pop.pointer<@"$expressions"::@MemoryType> borrow_in_mem) -> !lit.none
# CHECK-SAME: %float2: {{.*}}(!kgen.declref<@"$expressions"::@RegType>) ownedresult -> !kgen.declref<@"$expressions"::@RegType>
# CHECK-SAME: %float3: {{.*}}(!pop.pointer<@"$expressions"::@MemoryType> owned_in_mem) -> !lit.none
# CHECK-SAME: %float4: {{.*}}(!pop.pointer<{{.*}}@"$Int"::@Int> byref) -> !lit.none
# CHECK-SAME: %float5: {{.*}}(!kgen.declref<{{.*}}@"$Int"::@Int> borrow) throws -> !pop.variant<@"$Builtin"::@"$Error"::@Error, !lit.none>
# CHECK-SAME: %float6: {{.*}}(!kgen.declref<@"$Builtin"::@"$Int"::@Int> borrow) throws|async|capturing -> !pop.variant<@"$Builtin"::@"$Error"::@Error, !lit.none>
# CHECK-SAME: %float7: {{.*}}(!kgen.variadic<@"$Builtin"::@"$Int"::@Int>) throws|vararg -> !pop.variant<@"$Builtin"::@"$Error"::@Error, !lit.none>
# CHECK-SAME: %float8: {{.*}}<{{.*}}@"$Int"::@Int>(!kgen.declref<@"$expressions"::@ParamType<[[A]]: {{.*}}@"$Int"::@Int = *(0,0)>> borrow) -> !lit.none
# CHECK-SAME: %float9: {{.*}}<[] -> {{.*}}@"$Int"::@Int>() -> !lit.none
# CHECK-SAME: %float10: {{.*}}<<{{.*}}@"$Int"::@Int, @"$expressions"::@ParamType<[[A]]: {{.*}}@"$Int"::@Int = *(0,0)>>() throws -> !pop.variant<@"$Builtin"::@"$Error"::@Error, !lit.none>
# CHECK-SAME: %float11: {{.*}}<<variadic<!kgen.mlirtype>>(!pop.pack<*(0,0)>) throws|async|packvararg|param_vararg -> !pop.variant<@"$Builtin"::@"$Error"::@Error, !lit.none>
# CHECK-SAME: %float12: {{.*}}<(!kgen.declref<{{.*}}@"$Int"::@Int> borrow = #lit.struct<{value = 10}>, !kgen.declref<{{.*}}@"$StringLiteral"::@StringLiteral> borrow = #lit.struct<{value: string = "foo"}>) -> !lit.none>
fn function_types(
  float0: fn(Int) -> Int,
  float1: fn(MemoryType) -> MemoryType,
  float2: fn(owned RegType) -> RegType,
  float3: fn(owned MemoryType) -> None,
  float4: fn(inout Int) -> None,
  float5: fn(Int) raises -> None,
  float6: async fn(Int) capturing raises -> None,
  float7: def(*Int) -> None,
  float8: fn[a: Int](ParamType[a]) -> None,
  float9: fn[() -> a: Int]() -> None,
  float10: def[a: Int, b: ParamType[a]]() -> None,
  float11: async def[*Ts: AnyType](* *Ts) -> None,
  float12: fn(Int = 10, StringLiteral = "foo") -> None,
): pass

alias fn_type_alias = fn() -> None

@always_inline
fn func_with_decorator(): pass


struct TwoParamsStruct[a: Int, b: Int]:
    fn __copyinit__(inout self, existing: Self):
        pass

# CHECK-LABEL: lit.func @"variadic_subscript{{.*}})"<
# CHECK-SAME: {{.*}}idx: {{.*}}@"$Int"::@Int, [[A:.*]]: variadic<{{.*}}@"$Int"::@Int>>
fn variadic_subscript[idx: Int, *a: Int](*b: Int):
    # CHECK-NEXT: lit.alias.decl {{.*}}v0: {{.*}}Int = <variadic_get(:variadic<{{.*}}@"$Int"::@Int> [[A]], 2)>
    alias v0 = a[2]
    # CHECK: pop.variadic.get %{{.*}}[%idx3]
    let v1 = a[3]
    # CHECK: %[[IDX:.*]] = kgen.call {{.*}}__index__
    # CHECK-NEXT: %[[MLIR_IDX:.*]] = kgen.call {{.*}}__mlir_index__{{.*}}%[[IDX]]
    # CHECK-NEXT: pop.variadic.get %b[%[[MLIR_IDX]]]
    let v2 = b[idx]


# CHECK-LABEL: lit.func @"variadic_memory_subscript
# CHECK-SAME: variadic<!pop.pointer<{{.*}}TwoParamsStruct<
# CHECK-SAME:   a{{.*}} = variadic_get{{.*}}a, 0
# CHECK-SAME:   b{{.*}} = variadic_get{{.*}}a, 1
fn variadic_memory_subscript[*a: Int](*b: TwoParamsStruct[a[0], a[1]]):
    # CHECK: [[V0:%.*]] = pop.variadic.get %b[%idx1]
    # CHECK: __copyinit__{{.*}}[[V0]]
    let v0 = b[1]
    # CHECK: [[V1:%.*]] = pop.variadic.get %b[%idx2]
    # CHECK: __copyinit__{{.*}}[[V1]]
    var v1 = b[2]

fn takeMemory(a: MemoryType): pass

# CHECK-LABEL: lit.func @"testConds
fn testConds(cond: __mlir_type.i1, a: MemoryType, b: MemoryType, m: M, i: Int) -> MemoryType:
  # Implicit conversions.
  # Mojo Issue #49: https://github.com/modularml/mojo/issues/49

  # CHECK-NEXT: hlcf.if %cond -> !kgen.declref<@"$expressions"::@M> {
  # CHECK-NEXT:   [[V:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%m)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[V:%.*]] = kgen.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: }
  _ = m if cond else i

  # CHECK-NEXT: hlcf.if %cond -> !kgen.declref<@"$expressions"::@M> {
  # CHECK-NEXT:   [[V:%.*]] = kgen.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[V:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%m)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: }
  _ = i if cond else m

  # Memory only conds.
  # Issue (#13379)

  # CHECK-NEXT: %anonymous2A = lit.varlet.decl
  # CHECK-NEXT: hlcf.if %cond {
  # CHECK-NEXT:   kgen.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %a)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   kgen.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %b)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  # CHECK-NEXT: kgen.call {{.*}}takeMemory{{.*}}(%anonymous2A)
  takeMemory(a if cond else b)

  # CHECK-NEXT: hlcf.if %cond {
  # CHECK-NEXT:   kgen.call {{.*}}__copyinit__{{.*}}(%__result__, %a)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   kgen.call {{.*}}__copyinit__{{.*}}(%__result__, %b)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  # CHECK-NEXT: kgen.param.constant: !lit.none = <#lit.none>
  return a if cond else b

fn testTransferWarning():
  let a = MemoryOnlyInt()

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(a^^)

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(MemoryOnlyInt()^)
