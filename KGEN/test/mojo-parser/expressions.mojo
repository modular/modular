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
    # CHECK: %1 = {{.*}}constant: {{.*}}Int = {{.*}} 1
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
  var x: FloatLiteral
  fn __init__(inout self, value: MemoryOnlyInt):
    self.x = 1.0

# CHECK-LABEL: lit.struct.decl @MemoryOnlyPair
struct MemoryOnlyPair:
  var x: MemoryOnlyInt
  var y: Int

  # CHECK: lit.func @"__copyinit__{{.*}}(%self: !lit.ref<mut !MemoryOnlyPair, {{.*}}> init_self,
  # CHECK-SAME: %other: !lit.ref<!MemoryOnlyPair, {{.*}}> borrow_in_mem)
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
  # CHECK-SAME: %self: !lit.ref<mut !MemoryOnlyPair, {{.*}}> owned_in_mem,
  # CHECK-SAME: %arg: !lit.ref<mut !MemoryOnlyInt, {{.*}}> owned_in_mem)
  fn method(owned self, owned arg: MemoryOnlyInt):
    # CHECK: %0 = lit.ref.struct.ger %self[y]
    # CHECK: %1 = lit.ref.struct.ger %arg[x]
    # CHECK: %2 = lit.ref.load %0
    # CHECK: %3 = lit.ref.load %1
    # CHECK: %4 = lit.call @"{{.*}}__add__{{.*}}"(%2, %3)
    _ = self.y+arg.x

fn inferred_function_with_memory_result[
  width: Int](x: SIMD[DType.float32, width]) -> MemoryOnlyInt: pass

# CHECK-LABEL: lit.func @"memoryOnlyOps
fn memoryOnlyOps(inout a: MemoryOnlyPair) -> MemoryOnlyPair:
  # CHECK-NEXT: %v1 = lit.varlet.decl {{.*}} var : !lit.ref<mut !MemoryOnlyPair,
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%v1, [[IMMREF]])
  var v1 = a

  # CHECK-NEXT: %v2 = lit.varlet.decl "v2" let
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%v2, [[IMMREF]])
  let v2 : MemoryOnlyPair = a

  # CHECK-NEXT: %anonymous2A = lit.varlet.decl {{.*}} synth
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, [[IMMREF]])
  _ = a

  a  # expected-warning {{'MemoryOnlyPair' value is unused}}

  # CHECK-NEXT: %regX = lit.varlet.decl {{.*}} let
  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[AX]]
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%regX, [[IMMREF]])
  let regX = a.x

  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[AX]], [[IMMREF]])
  a.x = regX

  # Pass memory only things by value as arguments.

  # CHECK-NEXT: [[TMPPAIR:%.*]] = lit.varlet.decl {{.*}}!MemoryOnlyPair
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[TMPPAIR]], [[IMMREF]])
  # CHECK-NEXT: [[TMPINT:%.*]] = lit.varlet.decl {{.*}}!MemoryOnlyInt
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[TMPINT]], [[IMMREF]])
  # CHECK-NEXT: lit.call @{{.*}}@"method{{.*}}([[TMPPAIR]], [[TMPINT]])
  a.method(regX)

  # Drill into rvalue without cloning intermediate values.
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[V2XX:%.*]] = lit.ref.struct.ger [[V2X]][x]
  # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load [[V2XX]]
  # CHECK-NEXT: lit.letreg.decl "v2xx" = [[VAL]]
  let v2xx = v2.x.x

  # Implicit conversion between memory-only types.
  # CHECK-NEXT: %mpFloat = lit.varlet.decl
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[V2X]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%mpFloat, [[IMMREF]])
  let mpFloat : MemoryOnlyFloat64 = v2.x

  # CHECK: [[TMP:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK-NEXT: lit.call @{{.*}}inferred_function_with_memory_result{{.*}}(%anonymous2A
  _ = inferred_function_with_memory_result(SIMD[DType.float32, 4]())

  # Memory-only default argument with memory-only result.
  # CHECK-NEXT: [[TMP:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK-NEXT: %[[C42:.*]] = {{.*}}constant: {{.*}}Int = {{.*}} 42
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}([[TMP]], %[[C42]])
  _ = MemoryOnlyInt()

  # CHECK-NEXT: [[IMMREF1:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[IMMREF2:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.create [[[IMMREF1]], [[IMMREF2]]]
  # CHECK-NEXT: lit.call @{{.*}}variadic{{.*}}([[VARIADIC]])
  MemoryOnlyInt.variadic(regX, regX)
  # CHECK-NEXT: lit.ownership.use [[IMMREF1]]
  # CHECK-NEXT: lit.ownership.use [[IMMREF2]]

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %v2
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%__result__, [[IMMREF]])
  # CHECK-NEXT: [[NONEVAL:%.*]] = kgen.param.constant: none = <#kgen.none>
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

    # CHECK: %f = lit.varlet.decl "f"
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
  fn __init__(value: Int) -> Self:
  # CHECK-NEXT:  = lit.struct.create(value=%value) : (!Int) -> !RegPassable
    return Self{value: value}

  fn __copyinit__(self) -> Self: pass
  fn __del__(owned self): pass
  fn __neg__(self) -> Self: pass
  fn __add__(self, rhs: Self) -> Self: pass
  fn __matmul__(self, rhs: Self) -> Self: pass
  fn __rmatmul__(lhs, self: Self) -> Self: pass

# CHECK-LABEL: lit.struct.decl @StructWithFuncParam
# CHECK-SAME: <[[PARAM:.*]][comparator]: !lit.signature
# CHECK-SAME: <"T": regtype>(!kgen.paramref<*(0,0)> borrow, |)
struct StructWithFuncParam[comparator: fn[T: AnyRegType] (T) -> None]:
    # CHECK-LABEL: lit.func @"f
    # CHECK-SAME: %self: !lit.ref<{{.*}}<:!lit.signature<<"T": regtype>(!kgen.paramref<*(0,0)>
    fn f(self):
        pass

    # CHECK-LABEL: lit.func @"g
    fn g(self):
        # CHECK: call {{.*}}[*"`self"]<:!lit.signature<<"T": regtype>(!kgen.paramref<*(0,0)> borrow, |)
        # CHECK-SAME: !lit.ref<{{.*}}<"T": regtype>(!kgen.paramref<*(0,0)> borrow, |)
        self.f()

# CHECK-LABEL: lit.func @"simpleMath
fn simpleMath(a: Int, b: Int) -> Int:
  # CHECK: %0 = lit.call {{.*}}Int::@"__mul__{{.*}}(%b, %a)
  # CHECK: %1 = lit.call {{.*}}Int::@"__sub__{{.*}}(%a, %0)
  # CHECK: lit.return %1 : !Int
  return a-b*a

# CHECK-LABEL: lit.func @"precedence_associativity
fn precedence_associativity(a: Int):
  # CHECK: %z = lit.varlet.decl "z" var
  var z: Int = 0

  # CHECK: [[SEVENTEENINT:%.*]] = kgen{{.*}}#lit.struct<{value = 17}>
  # CHECK-NEXT: lit.ref.store [[SEVENTEENINT]], %z
  z = 17  # Implicit conversion

  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT: %[[POW0:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT: %[[POW1:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0]])
  # CHECK-NEXT: lit.ref.store %[[POW1]], %z
  z = 2**(a**z)
  # CHECK-NEXT: %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT: %[[POW0:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%a, %[[Z]])
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT: %[[POW1:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[POW0]])
  # CHECK-NEXT: lit.ref.store %[[POW1]], %z
  z = 2**a**z
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[MUL:.*]] = kgen.param.constant: !Int = <{{.*}} = -6}
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[Z]], %[[MUL]])
  # CHECK-NEXT:  lit.ref.store %[[ADD]], %z
  z = z + 3 * -2
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[FLOOR_DIV:.*]] = kgen.param.constant: !Int = <{{.*}} = -2}
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[Z]], %[[FLOOR_DIV]])
  # CHECK-NEXT:  lit.ref.store %[[ADD]], %z
  z = z + 3 // -2
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[INT_THREE:.*]] = kgen{{.*}}#lit.struct<{value = 3}>
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[Z]], %[[INT_THREE]])
  # CHECK-NEXT:  %[[NEG:.*]] = kgen{{.*}}#lit.struct<{value = -2}>
  # CHECK-NEXT:  %[[MUL:.*]] =  lit.call {{.*}}Int::@"__mul__{{.*}}(%[[ADD]], %[[NEG]])
  # CHECK-NEXT:  lit.ref.store %[[MUL]], %z
  z = (z + 3) * -+2
  # CHECK-NEXT:  %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT:  %[[Z:.*]] = lit.ref.load %z
  # CHECK-NEXT:  %[[POW:.*]] = lit.call {{.*}}Int::@"__pow__{{.*}}(%[[INT_TWO]], %[[Z]])
  # CHECK-NEXT:  %[[NEG:.*]] = lit.call {{.*}}Int::@"__neg__{{.*}}(%[[POW]])
  # CHECK-NEXT:  lit.ref.store %[[NEG]], %z
  z = -2**z
  # CHECK-NEXT: [[Z:%.*]] = lit.ref.load %z
  # CHECK-NEXT: [[ONE:%.*]] = kgen{{.*}}#lit.struct<{value = 1}>
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__radd__({{.*}}$int::Int,{{.*}}$int::Int)"([[Z]], [[ONE]])
  # CHECK-NEXT: lit.ref.store [[RES]], %z
  z = Int(1).value + z

  # div tests
  # CHECK: lit.call {{.*}}__truediv__
  var r0 = Float32(33.0) / Float32(42.0)

  # CHECK: lit.call {{.*}}__truediv__
  var r1 = Float32(33.0) / 42.0

# CHECK-LABEL: lit.func @"reverse_operators
fn reverse_operators(a: Int):
  # CHECK: [[RES:%.*]] = lit.call {{.*}}Int::@"__radd__({{.*}}$int::Int,{{.*}}$int::Int)"
  var z = Int(1).value + a

  # CHECK: [[RES:%.*]] = lit.call {{.*}}Int::@"__rsub__({{.*}}$int::Int,{{.*}}$int::Int)"
  z = Int(2).value - z

  # CHECK: [[RES:%.*]] = lit.call {{.*}}Int::@"__rmul__({{.*}}$int::Int,{{.*}}$int::Int)"
  z = Int(3).value * z

  # div tests
  # CHECK: lit.call {{.*}}__rtruediv__
  # CHECK: lit.call {{.*}}Int::@"__rfloordiv__({{.*}}$int::Int,{{.*}}$int::Int)"
  var r1 = 33.0 / Float32(42.0)
  z = Int(33).value // z

  # CHECK: lit.call {{.*}}Int::@"__rmod__({{.*}}$int::Int,{{.*}}$int::Int)"
  var i0 = Int(10).value % z

# CHECK: lit.call {{.*}}Int::@"__rpow__({{.*}}$int::Int,{{.*}}$int::Int)"
  var i1 = Int(3).value ** z

  # CHECK: lit.call {{.*}}Int::@"__rlshift__({{.*}}$int::Int,{{.*}}$int::Int)"
  var i2 = Int(1).value << z

  # CHECK: lit.call {{.*}}Int::@"__rrshift__({{.*}}$int::Int,{{.*}}$int::Int)"
  var i3 = Int(1).value >> z

  # CHECK: lit.call {{.*}}Int::@"__rand__({{.*}}$int::Int,{{.*}}$int::Int)"
  z = Int(1).value & z

  # CHECK: lit.call {{.*}}Int::@"__ror__({{.*}}$int::Int,{{.*}}$int::Int)"
  z = Int(2).value | z

  # CHECK: lit.call {{.*}}Int::@"__rxor__({{.*}}$int::Int,{{.*}}$int::Int)"
  z = Int(3).value ^ z

# CHECK-LABEL: lit.func @"precedence_matmul
fn precedence_matmul(z: RegPassable) -> RegPassable:
  # CHECK-NEXT:  %[[THREE:.*]] = kgen.param.constant: {{.*}}Int = {{.*}} 3
  # CHECK-NEXT:  %[[INT_THREE:.*]] = lit.call {{.*}}@RegPassable::@"__init__{{.*}}(%[[THREE]])
  # CHECK-NEXT:  %[[TWO:.*]] = kgen.param.constant: {{.*}}Int = {{.*}} 2
  # CHECK-NEXT:  %[[INT_TWO:.*]] = lit.call {{.*}}@RegPassable::@"__init__{{.*}}(%[[TWO]])
  # CHECK-NEXT:  %[[NEG:.*]] = lit.call {{.*}}@RegPassable::@"__neg__{{.*}}(%[[INT_TWO]])
  # CHECK-NEXT:  %[[MATMUL:.*]] = lit.call {{.*}}@RegPassable::@"__matmul__{{.*}}(%[[INT_THREE]], %[[NEG]])
  # CHECK-NEXT:  %[[ADD:.*]] = lit.call {{.*}}@RegPassable::@"__add__{{.*}}(%z, %[[MATMUL]])
  # CHECK-NEXT:  lit.return %[[ADD]] : !RegPassable
  return z + RegPassable(3) @ -RegPassable(2)

# CHECK-LABEL: lit.func @"precedence_bitwise
fn precedence_bitwise(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %[[INT_TWO:.*]] = kgen{{.*}}#lit.struct<{value = 2}>
  # CHECK-NEXT: %[[MUL:.*]] = lit.call {{.*}}Int::@"__mul__{{.*}}(%a, %[[INT_TWO]])
  # CHECK-NEXT: %[[AND:.*]] = lit.call {{.*}}Int::@"__and__{{.*}}(%[[MUL]], %b)
  # CHECK-NEXT: %[[INT_FOUR:.*]] = kgen{{.*}}#lit.struct<{value = 4}>
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
  fn __copyinit__(self) -> Self: pass
  fn __bool__(self) -> Bool: return True

struct MemBoolish:
  fn __init__(inout self, value: Boolish): pass
  fn __copyinit__(inout self, other: Self): pass
  fn __bool__(self) -> Bool: return True

# CHECK-LABEL: @"unary
fn unary(a: Bool, b: Int, c: Boolish, d: MemBoolish):
  # CHECK: %0 = lit.call {{.*}}@"$bool"::@Bool::@"__bool__({{.*}}$bool::Bool)"(%a)
  # CHECK: %1 = lit.call {{.*}}@"$bool"::@Bool::@"__invert__({{.*}}$bool::Bool)"(%0)
  _ = not a

  # CHECK: [[EQ:%.*]] = lit.call {{.*}}Int::@"__eq__({{.*}}$int::Int,{{.*}}$int::Int)"
  # CHECK: [[EQBOOL:%.*]] = lit.call {{.*}}@"$bool"::@Bool::@"__bool__({{.*}}$bool::Bool)"([[EQ]])
  # CHECK:  = lit.call {{.*}}@"$bool"::@Bool::@"__invert__({{.*}}$bool::Bool)"([[EQBOOL]])
  _ = not b == 0

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%c)
  # CHECK:  = lit.call {{.*}}@"$bool"::@Bool::@"__invert__({{.*}}$bool::Bool)"([[BOOL]])
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
  # CHECK:   [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%b)
  # CHECK:   hlcf.yield [[TMP]]
  # CHECK: } else {
  # CHECK:   [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK:   hlcf.yield [[TMP]]
  # CHECK: }
  _ = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.  Boolish is defined with copy ctor so it must be invoked.

  # CHECK-NEXT: [[ABOOL:%.*]] = lit.call {{.*}}Boolish::@"__bool__{{.*}}"(
  # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !Boolish {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%b)
  # CHECK-NEXT:   hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a or b

  # Testing two different logic'y types returns the common bool type if present.

  # CHECK-NEXT: [[ABOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !Bool {
  # CHECK-NEXT:   hlcf.yield %c
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[ABOOL:%.*]] = lit.call {{.*}}__init__{{.*}}([[I1]])
  # CHECK-NEXT:   hlcf.yield [[ABOOL]]
  # CHECK-NEXT: }
  _ = a and c

  # Check incompatible types that are nevertheless boolish.

  # CHECK-NEXT: [[BBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%b)
  # CHECK-NEXT: [[BI1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BBOOL]])
  # CHECK-NEXT: = hlcf.if [[BI1]] -> !Bool {
  # CHECK-NEXT:    [[TMP:%.*]] = lit.call {{.*}}@Bool::@"__init__{{.*}}([[BI1]])
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT:  } else {
  # CHECK-NEXT:    hlcf.yield %c
  # CHECK-NEXT:  }
  _ = b or c

  # Check memory-only boolish types.
  # Boolish and MemBoolish has a common type of MemBoolish.

  # CHECK-NEXT: [[DBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%d)
  # CHECK-NEXT: [[DI1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[DBOOL]])
  # CHECK-NEXT: [[IFRESULT:%.*]] = lit.varlet.decl {{.*}} : !lit.ref<mut !MemBoolish
  # CHECK-NEXT: hlcf.if [[DI1]] {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, %d)
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMPMEM:%.*]] = lit.varlet.decl
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}([[TMPMEM]], %b)
  # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut [[TMPMEM]]
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%anonymous2A, [[IMMREF]])
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  _ = d or b

# CHECK-LABEL: lit.func @"paramAndOr{{.*}}()"
# CHECK-SAME: <[[A:.*_a]][a]: !Boolish, [[B:.*_b]][b]: !Boolish>
fn paramAndOr[a: Boolish, b: Boolish]():
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: lit.alias.decl {{.*}}c: !Boolish = <cond(apply({{.*}}@Bool::@"__mlir_i1__{{.*}}", apply({{.*}}Boolish::@"__bool__{{.*}}", [[A]])), [[B]], [[A]])>
  alias c = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.

  # CHECK: lit.alias.decl {{.*}}d: !Boolish = <cond(apply({{.*}}@Bool::@"__mlir_i1__{{.*}}", apply({{.*}}Boolish::@"__bool__{{.*}}", [[A]])), [[A]], [[B]])>
  alias d = a or b

# CHECK-LABEL: lit.func @"do_math
fn do_math(a: Int, b: Int, c: Int) -> Int:
  # CHECK-NEXT: %z = lit.varlet.decl "z" var
  var z : Int
  # CHECK-NEXT: %[[INT_5:.*]] = kgen{{.*}}#lit.struct<{value = 5}>
  # CHECK-NEXT: %[[MUL:.*]] = lit.call {{.*}}Int::@"__mul__{{.*}}(%[[INT_5]], %a)
  # CHECK-NEXT: %[[INT_42:.*]] = kgen{{.*}}#lit.struct<{value = 42}>
  # CHECK-NEXT: %[[ADD:.*]] = lit.call {{.*}}Int::@"__add__{{.*}}(%[[INT_42]], %[[MUL]])
  # CHECK-NEXT: lit.ref.store %[[ADD]], %z
  z = 42 + 5*a

  # CHECK-NEXT: %x = lit.varlet.decl "x" var
  # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %z
  # CHECK-NEXT: lit.ref.store [[TMP]], %x
  # This is checking the lexer handles \ at end of line correctly.
  var x : Int
  x = \
z

  # CHECK-NEXT: lit.call @"$expressions"::@"noop()"()
  noop()

  # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %x
  # CHECK-NEXT: lit.return [[TMP]]
  return x

# CHECK-LABEL: lit.func @"listValues()"
fn listValues():
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__
  # CHECK: lit.ref.store %[[LIST:.*]], %a
  var a = [1, 2, 2+1]
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__
  # CHECK: lit.ref.store %[[LIST:.*]], %a
  a = [1, 2, 2+1,]
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__
  # CHECK: lit.ref.store %[[LIST:.*]], %a
  a = [1, 2, 2+1]
  # CHECK: %[[LIST:.*]] = lit.call {{.*}}@ListLiteral::@"__init__
  # CHECK: lit.ref.store %[[LIST:.*]], %b
  var b = []

# CHECK-LABEL: lit.func @"initializers
fn initializers():
  # CHECK: %0 = kgen.param.constant: !Int = <#lit.struct<{value = 42}>>
  # CHECK: lit.letreg.decl "a" = %0
  let a = Int{value: Int(42).value}

  # Issue #7343: Trailing comma ok too.
  _ = Int{value: Int(42).value,}

  # Issue #12067, suffix stuff ok.
  _ = Int{ value: Int(1).value }.value

# CHECK-LABEL: lit.func @"test_if_cond
fn test_if_cond(owned cond: Bool, memCond: MemBoolish):
    # CHECK: lit.ref.store %cond, %cond_0
    # CHECK: %i = lit.varlet.decl "i"
    # CHECK: [[COND:%.*]] = lit.ref.load %cond_0
    # CHECK: %[[LIT_BOOLI1:.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[COND]])
    # CHECK-NEXT: %[[IF_RES:.*]] = hlcf.if %[[LIT_BOOLI1]]
    # CHECK-NEXT:   %[[INT_TWO:.*]] = kgen{{.*}}= 2}
    # CHECK-NEXT:   hlcf.yield %[[INT_TWO]]
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %[[INT_THREE:.*]] = kgen{{.*}}= 3}
    # CHECK-NEXT:   hlcf.yield %[[INT_THREE]]
    # CHECK-NEXT: }
    # CHECK-NEXT: lit.ref.store %[[IF_RES]], %i
    var i: Int = 2 if cond else 3

    # CHECK: [[TRUEB:%.+]] = kgen{{.*}}= true}
    # CHECK-NEXT: lit.ref.store [[TRUEB]], %cond_0
    cond = True
    i += i
    if cond:     # 'if' stmt, not an 'if' expression.
        i += 1

# CHECK-LABEL: lit.func @"test_param_if_cond{{.*}}()"
# CHECK-SAME: <[[COND:.*_cond]][cond]: !Bool>
fn test_param_if_cond[cond: Bool]() -> Int:
  # CHECK: lit.alias.decl [[I_ALIAS:.*]]: !IntLiteral = <cond(apply({{.*}}Bool::@"__mlir_i1__{{.*}}", [[COND]]), #lit.struct<{value: !kgen.int_literal = 2}>, #lit.struct<{value: !kgen.int_literal = 3}>)>
  alias i = 2 if cond else 3

  # CHECK-NEXT: lit.alias.decl {{.*}}j: !FloatLiteral = <cond(apply({{.*}}Bool::@"__mlir_i1__{{.*}}", [[COND]]), #lit.struct<{value: scalar<f64> = "2"}>, #lit.struct<{value: scalar<f64> = "3"}>)>
  alias j = 2.0 if cond else 3

  # CHECK: %[[I:.*]] = kgen.param.constant: !Int = {{.*}}IntLiteral{{.*}}[[I_ALIAS]]{{.*}}
  return i

# CHECK-LABEL: lit.func @"callable_mv[fn({{.*}}::Int, /) -> {{.*}}::Int]({{.*}}::Int)"
# CHECK-SAME: <[[CALLABLE:.*_callable]][callable]: !lit.signature<(!Int borrow, |) -> !Int>>(%a: !Int borrow) -> !Int
fn callable_mv[callable: fn (Int) -> Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call_param[!lit.signature<(!Int borrow, |) -> !Int>: [[CALLABLE]]](%a)
  return callable(a)

# CHECK-LABEL: lit.func @"callable_mv_inputs{{.*}})"<
# CHECK-SAME: [[CALLABLE:.*_callable]][callable]: !lit.signature<<"x": !Int>(!Int borrow, |) -> !Int>, [[B:.*_b]][b]: !Int>(%a: !Int borrow) -> !Int
fn callable_mv_inputs[callable: fn[x: Int](Int) -> Int, b: Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call_param[!lit.signature<(!Int borrow, |) -> !Int>: bind_signature({{.*}}[[CALLABLE]], [[B]])](%a)
  return callable[b](a)

# CHECK-LABEL: lit.func @"takeIndexParam{{.*}}"<{{.*}}[a]: !Int>() -> !Int
fn takeIndexParam[a: Int]() -> Int:
  return a + 1

# CHECK-LABEL: lit.func @"returnIndex()"() -> !Int
fn returnIndex() -> Int:
  return 0

# CHECK-LABEL: lit.func @"returnIndex2()"() -> !Int
fn returnIndex2() -> Int:
  # CHECK-NEXT: %0 = lit.call @"$expressions"::@"takeIndexParam{{.*}}"<:!Int apply({{.*}}@"$expressions"::@"returnIndex()")>()
  # CHECK-NEXT: return %0
  return takeIndexParam[returnIndex()]()

# CHECK-LABEL: lit.func @"callInParam[fn[{{.*}}::Int]({{.*}}::Int, /) -> {{.*}}::Int]()"
# CHECK-SAME: <[[CALLABLE:.*_callable]][callable]: !lit.signature<<"x": !Int>(!Int borrow, |) -> !Int>>() -> !Int
fn callInParam[callable: fn[x: Int](Int) -> Int]() -> Int:
  # CHECK-NEXT: %0 = lit.call @"$expressions"::@"takeIndexParam{{.*}}()"<:!Int apply({{.*}}bind_signature({{.*}}[[CALLABLE]], #lit.struct<{value = 1}>), #lit.struct<{value = 1}>)>()
  # CHECK-NEXT: return %0
  return takeIndexParam[callable[1](1)]()

# CHECK-LABEL: lit.func @"parameterExprs{{.*}}()"
# CHECK-SAME: <[[A:.*_a]][a]: !Int, [[A2:.*_a2]][a2]: !Int>
fn parameterExprs[a: Int, a2: Int]():
  # CHECK: lit.alias.decl {{.*}}b: !Int = <apply({{.*}}__sub__{{.*}}, [[A]], [[A]])>
  alias b = a-a
  # CHECK: lit.alias.decl {{.*}}c: !Int = <apply({{.*}}__add__{{.*}}, [[A]], {{.*}}42{{.*}})>
  alias c = a+42
  # CHECK: lit.alias.decl {{.*}}d: !Int = <apply({{.*}}__mul__{{.*}}, [[A]], [[A2]])>
  alias d = a*a2

##===----------------------------------------------------------------------===##
# Patterns, LValues and RValues
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"patterns()
fn patterns():
  # CHECK: %z2 = lit.varlet.decl "z2" var
  var z2: Int

  (((z2))) = 42  # Paren patterns
  # CHECK: [[TMP:%.*]] = {{.*}}constant{{.*}} 42
  # CHECK: lit.ref.store [[TMP]], %z2

  var someInt : Int
  (someInt) += someInt
  # CHECK: %someInt = lit.varlet.decl "someInt" var
  # CHECK:  %1 = lit.ref.load %someInt
  # CHECK:   = lit.call {{.*}}Int::@"__iadd__{{.*}}(%someInt, %1)

  # Discard pattern with different types.
  (_) = someInt
  # CHECK: [[TMP:%.*]] = lit.ref.load %someInt

  (_) = 1.0

  # CHECK: %someFloat32 = lit.varlet.decl "someFloat32" var
  # CHECK: [[Float32:%.*]] = lit.ref.load %someFloat32
  # CHECK: {{%.*}} = lit.call {{.*}}__iadd__{{.*}}(%someFloat32, [[Float32]])
  var someFloat32 : Float32
  (someFloat32) += someFloat32

  # CHECK: %someSIMD = lit.varlet.decl "someSIMD" var
  # CHECK: [[SIMD:%.*]] = lit.ref.load %someSIMD
  # CHECK: {{%.*}} = lit.call {{.*}}@"$builtin"::@"$simd"::@SIMD::@"__iadd__({{.*}}(%someSIMD, [[SIMD]])
  var someSIMD : SIMD[DType.float64, 4]
  (someSIMD) += someSIMD

# CHECK-LABEL: lit.func @"byval_byref_function({{.*}}$int::Int,{{.*}}$int::Int&)"{{.*}}(%a: !Int borrow, %b: !lit.ref<mut !Int, {{.*}}> byref) -> !kgen.none
fn byval_byref_function(a: Int, inout b: Int):
  # CHECK-NEXT: lit.ref.store %a, %b
  b = a

  # CHECK-NEXT: %x = lit.varlet.decl "x" var
  var x : Int
  # This needs to load 'b' to pass it by value for the first arg, but pass its
  # address in directly for the second.
  # CHECK: %0 = lit.ref.load %b
  # CHECK: = lit.call @{{.*}}::@"byval_byref_function{{.*}}(%0, %b)
  byval_byref_function(b, b)

# CHECK-LABEL: lit.func @"lvaluesAndRValues()
fn lvaluesAndRValues() -> __mlir_type.index:
  # CHECK: [[VALUE:%.*]] = kgen.param.constant = <4>
  # CHECK: lit.return [[VALUE]] : index
  return Int(4).value

# CHECK-LABEL: lit.func @"mvalueStructField()"
fn mvalueStructField():
  # CHECK: lit.alias.decl [[INT:.*]]: !Int = <#lit.struct<{value = 4}>>
  alias int = Int(4)
  # CHECK: lit.alias.decl {{.*}}value = <#lit.struct.extract<:!Int [[INT]], "value">>
  alias value = int.value
  alias foldToValue = Int(5).value

# CHECK-LABEL: lit.func @"defTests({{.*}}, %{{.*}}[untyped]: !lit.ref<mut !object, {{.*}}> owned_in_mem)
def defTests(a: Int, b: Int, untyped) -> None:
  # CHECK: %a_0 = lit.varlet.decl "a" imp
  # CHECK: lit.ref.store %a, %a_0
  # CHECK: %b_1 = lit.varlet.decl "b" imp : !lit.ref<mut !Int, *"`b1">
  # CHECK: lit.ref.store %b, %b_1
  # CHECK: [[B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: lit.ref.store [[B]], %a_0
  a = b # Parameters are mutable!

##===----------------------------------------------------------------------===##
# Augmented Assignments
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"basic_assignments
def basic_assignments(a: Int, b: Int, c: RegPassable, d: RegPassable):
  # CHECK:      %a_0 = lit.varlet.decl "a" imp
  # CHECK:      %b_1 = lit.varlet.decl "b" imp
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__iadd__{{.*}}(%a_0, [[LOAD_B]])
  a += b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__isub__{{.*}}(%a_0, [[LOAD_B]])
  a -= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__imul__{{.*}}(%a_0, [[LOAD_B]])
  a *= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__ifloordiv__{{.*}}(%a_0, [[LOAD_B]])
  a //= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__imod__{{.*}}(%a_0, [[LOAD_B]])
  a %= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__ipow__{{.*}}(%a_0, [[LOAD_B]])
  a **= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__irshift__{{.*}}(%a_0, [[LOAD_B]])
  a >>= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__ilshift__{{.*}}(%a_0, [[LOAD_B]])
  a <<= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__iand__{{.*}}(%a_0, [[LOAD_B]])
  a &= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__ixor__{{.*}}(%a_0, [[LOAD_B]])
  a ^= b
  # CHECK:      [[LOAD_B:%.*]] = lit.ref.load %b_1
  # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}Int::@"__ior__{{.*}}(%a_0, [[LOAD_B]])
  a |= b

  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}value = 4
  # CHECK-NEXT: lit.ref.store [[FOUR]], %b_1
  # CHECK-NEXT: lit.ref.store [[FOUR]], %a_0
  a = b = 4

  # Walrus
  # CHECK-NEXT: [[SEVEN:%.*]] = kgen.param.constant: {{.*}}value = 7
  # CHECK-NEXT: lit.ref.store [[SEVEN]], %b_1
  # CHECK-NEXT: [[A:%.*]] = lit.ref.load %a_0
  # CHECK-NEXT: lit.call {{.*}}simpleMath{{.*}}([[A]], [[SEVEN]])
  simpleMath(a, b := 7)

# Issue #20145: Walrus operator should implicitly declare variable in def functions.
# CHECK-LABEL: lit.func @"walrus_implicit_decl
def walrus_implicit_decl():
  # CHECK:      %a = lit.varlet.decl "a" imp
  # CHECK-NEXT: [[THREE:%.*]] = kgen.param.constant: {{.*}}value = 3
  # CHECK-NEXT: lit.ref.store [[THREE]], %a
  # CHECK-NEXT: [[VAR_A:%.*]] = lit.ref.load %a
  # CHECK-NEXT: lit.call {{.*}}([[THREE]], [[VAR_A]])
  simpleMath(a := 3, a)

  # CHECK:      %b = lit.varlet.decl "b" imp
  # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}value = 4
  # CHECK-NEXT: lit.ref.store [[FOUR]], %b
  if b := 4:
    print(b)

  # CHECK:      %c = lit.varlet.decl "c" imp
  # CHECK-NEXT: [[FIVE:%.*]] = kgen.param.constant: {{.*}}value = 5
  # CHECK-NEXT: lit.ref.store [[FIVE]], %c
  # CHECK:      %d = lit.varlet.decl "d" imp
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
    c = False         # CHECK: !Bool = <#lit.struct<{value: scalar<bool> = false}>>
    c = True          # CHECK: !Bool = <#lit.struct<{value: scalar<bool> = true}>>

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
        # CHECK: string = "123"
        return "123"
        # lit.end_func
    """other comment"""


##===----------------------------------------------------------------------===##
# Tuples
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"tuples_rv
fn tuples_rv(a: Int, b: Float32):
    # CHECK: [[PACK0:%.*]] = kgen.param.constant: !kgen.pack<[]> = <<>>
    # CHECK: lit.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK0]])
    _ = ()

    # CHECK: [[PACK1:%.*]] = kgen.pack.create(%a, %b)
    # CHECK: lit.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK1]])
    _ = (a, b)

    # CHECK: [[PACK1:%.*]] = kgen.pack.create(%a, %b)
    # CHECK: lit.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK1]])
    _ = a, b

    # CHECK: [[PACK2:%.*]] = kgen.pack.create(%a)
    # CHECK: lit.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK2]])
    _ = (a,)

    # CHECK: [[PACK2:%.*]] = kgen.pack.create(%a)
    # CHECK: lit.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK2]])
    _ = a,

    # CHECK: %c = lit.varlet.decl "c"
    # CHECK: [[PACK2:%.*]] = kgen.pack.create(%a)
    # CHECK: [[TUP2:%.*]] = lit.call @"{{.*}}@Tuple::@"__init__({{.*}}([[PACK2]])
    # CHECK: lit.ref.store [[TUP2]], %c
    var c = a,

# CHECK-LABEL: lit.func @"tuples_lv
fn tuples_lv(i0: Int, f0: Float32):
   var i1 = 1
   var i2 = 2

   # CHECK: %iTup = lit.varlet.decl "iTup"
   var iTup : Tuple[Int, Int]

   # Tuple Rvalue
   # CHECK: [[TUP:%.*]] = lit.call {{.*}}@Tuple::@"__init__
   # CHECK: lit.ref.store [[TUP]], %iTup
   iTup = (i1, i2)

   # Tuple LValue
   # CHECK: [[TUP:%.*]] = lit.ref.load %iTup
   # CHECK: [[TUP2:%.*]] = lit.call {{.*}}@"__copyinit__{{.*}}([[TUP]])
   # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"get{{.*}}([[TUP2]])
   # CHECK-NEXT: lit.ref.store [[ELT]], %i1
   # CHECK: [[ELT:%.*]] = lit.call {{.*}}Tuple::@"get{{.*}}([[TUP2]])
   # CHECK-NEXT: lit.ref.store [[ELT]], %i2
   (i1, i2) = iTup

   # Check that the swap idiom is correct, this requires producing a copy of the
   # whole RValue on the right before extracting from it.

   # CHECK: [[I2VAL:%.*]] = lit.ref.load %i2
   # CHECK-NEXT: [[I1VAL:%.*]] = lit.ref.load %i1
   # CHECK-NEXT: [[PACK:%.*]] = kgen.pack.create([[I2VAL]], [[I1VAL]])
   # CHECK-NEXT: [[TUPRV:%.*]] = lit.call {{.*}}__init__{{.*}}([[PACK]])
   # CHECK-NEXT: [[I1VAL:%.*]] =  lit.call {{.*}}Tuple::@"get{{.*}}({{.*}} = 0{{.*}}([[TUPRV]])
   # CHECK-NEXT: lit.ref.store [[I1VAL]], %i1
   # CHECK-NEXT: [[I2VAL:%.*]] =  lit.call {{.*}}Tuple::@"get{{.*}}({{.*}} = 1{{.*}}([[TUPRV]])
   # CHECK-NEXT: lit.ref.store [[I2VAL]], %i2
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
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
  _ = a[idx]
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx)
  _ = a[idx, idx]
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx, %idx)
  _ = a[idx, idx, idx]
  # CHECK: [[VARIADIC:%.*]] = pop.variadic.create [%idx, %idx, %idx, %idx]
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %f, [[VARIADIC]])
  _ = a[f, idx, idx, idx, idx]

  # CHECK: [[SEVENTEEN:%.*]] = kgen.param.constant: {{.*}} = 17
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%a, %idx, %idx, [[SEVENTEEN]])
  a[idx, idx] = 17

fn test_kew_getitem(a: WeirdArray, idx: Int, idx2: Int, idx3: Int):
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx)
  _ = a[x=idx]
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2)
  _ = a[y=idx2, x=idx]
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
  _ = a[z=idx3, x=idx, y=idx2]
  # CHECK: lit.call {{.*}}@WeirdArray::@"__getitem__{{.*}}(%a, %idx, %idx2, %idx3)
  _ = a[idx, z=idx3, y=idx2]

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
  # CHECK: %[[I0:.*]] = kgen{{.*}}1
  # CHECK: %[[I2:.*]] = kgen{{.*}}none
  # CHECK-NEXT: call {{.*}}@slice::@"__init__{{.*}}"<{{.*}}>(%[[I0]], %i, %[[I2]])
  # CHECK-NEXT: call {{.*}}__getitem__
  a[1:i]
  # CHECK: %[[C2:.*]] = kgen{{.*}}2
  # CHECK: %[[I1:.*]] = {{.*}}@Int::@"__add__{{.*}}"(%[[C2]], %i)
  # CHECK: %[[I0:.*]] = kgen{{.*}}none
  # CHECK: %[[I2:.*]] = kgen{{.*}}3
  # CHECK-NEXT: call {{.*}}@slice::@"__init__{{.*}}"<{{.*}}>(%[[I0]], %[[I1]], %[[I2]])
  # CHECK-NEXT: call {{.*}}__getitem__
  a[:2+i:3]


# This is an array that has elements of MemoryOnlyInt.
struct MemoryOnlyIntArray:
  fn __getitem__(inout self, x: Int) -> MemoryOnlyInt: pass
  fn __setitem__(inout self, x: Int, owned value: MemoryOnlyInt): pass

# CHECK-LABEL: lit.func @"testMemoryOnlyIntArray
fn testMemoryOnlyIntArray(inout arr: MemoryOnlyIntArray, x: Int, owned moi: MemoryOnlyInt):
  # CHECK: %0 = lit.ownership.end_lifetime %moi
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %0)
  arr[x] = moi^
  # CHECK: [[ANON:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%anonymous2A, %arr, %x)
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %anonymous2A)
  arr[x] = arr[x]

  # CHECK: [[ANON:%.*]] = lit.varlet.decl "__store_tmp__"
  # CHECK-SAME: : !lit.ref<mut !MemoryOnlyInt, *"`__store_tmp__
  # CHECK: lit.call {{.*}}__getitem__{{.*}}([[ANON]], %arr, %x)
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[ANON]][x]
  # CHECK: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK: lit.ref.store %[[C1:.*]], [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x].x = 1

  # Initialize in memory through a temp + setitem.
  # CHECK: [[ANON:%.*]] = lit.varlet.decl "anonymous*"
  # CHECK: lit.call @"{{.*}}__init__{{.*}}([[ANON]],
  # CHECK: lit.call {{.*}}"__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x] = MemoryOnlyInt(42)

  # CHECK: [[STORETMP:%.*]] = lit.varlet.decl "__store_tmp__"
  # CHECK-SAME: : !lit.ref<mut !MemoryOnlyInt, *"`__store_tmp__
  # CHECK: lit.call {{.*}}__getitem__{{.*}}([[STORETMP]], %arr, %x)
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[STORETMP]][x]
  # CHECK: lit.ref.store {{.*}}, [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[STORETMP]])
  arr[x].x += 1


# Check a load from a SIMD field works.
# CHECK-LABEL: lit.func @"testSIMDGetter
fn testSIMDGetter[type: DType](owned a: SIMD[type, 2]) -> __mlir_type[
    `!pop.scalar<`, type.value, `>`]:
  # CHECK: %a_0 = lit.varlet.decl "a"
  # CHECK: lit.ref.store %a, %a_0
  # CHECK: [[AVAL:%.*]] = lit.ref.load %a_0
  # CHECK: [[ZERO:%.*]] = kgen.param.constant: {{.*}} = 0
  # CHECK: [[GOT:%.*]] = lit.call {{.*}}__getitem__{{.*}}([[AVAL]], [[ZERO]])
  # CHECK: [[RES:%.*]] = lit.struct.extract [[GOT]][value]
  # CHECK: lit.return [[RES]]
  return a[0].value



struct MyInlineIntInit:
    var value: MemoryOnlyInt
    # CHECK-LABEL: lit.func @"__init__($expressions::MyInlineIntInit=&,$expressions::MemoryOnlyInt)"
    # CHECK-SAME: (%self: !lit.ref<mut !MyInlineIntInit, {{.*}}> init_self, |, %value: !lit.ref<!MemoryOnlyInt, {{.*}}> borrow_in_mem) -> !kgen.none
    fn __init__(inout self, value: MemoryOnlyInt):
        # CHECK: %0 = lit.ref.struct.ger %self[value]
        # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%0, %value)
        self.value = value

struct IndexArray:
  fn __getitem__(inout self, x: Int) -> Int: pass
  fn __setitem__(inout self, x: Int, value: Int): pass

struct IndexArrayArray:
  fn __getitem__(inout self, x: Int) -> IndexArray: pass
  fn __setitem__(inout self, x: Int, value: IndexArray): pass

fn takeInOutInt(inout a: Int): pass

 # CHECK-LABEL: lit.func @"testWritebacks
fn testWritebacks(inout a: IndexArray, inout b: IndexArrayArray):
  # CHECK: %anonymous2A = lit.varlet.decl "anonymous*" synth
  # CHECK-NEXT: %[[V0:.*]] = {{.*}}constant{{.*}} = 0
  # CHECK-NEXT: %[[V1:.*]] = lit.call {{.*}}__getitem__{{.*}}(%a, %[[V0]])
  # CHECK-NEXT: lit.ref.store %[[V1]], %anonymous2A
  # CHECK-NEXT: %[[V2:.*]] = lit.call {{.*}}takeInOutInt{{.*}}(%anonymous2A)
  # CHECK-NEXT: %[[V3:.*]] = {{.*}}constant{{.*}} = 0
  # CHECK-NEXT: %[[V4:.*]] = lit.ref.load %anonymous2A
  # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__setitem__{{.*}}(%a, %[[V3]], %[[V4]])
  takeInOutInt(a[0]);

  # CHECK: %anonymous2A_0 = lit.varlet.decl
  # CHECK: %anonymous2A_1 = lit.varlet.decl {{.*}}!IndexArray
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: %[[V4:.*]] = {{.*}}__getitem__{{.*}}(%anonymous2A_1, %b, %[[C1]])
  # CHECK-NEXT: %[[C2:.*]] = {{.*}}constant{{.*}} = 2
  # CHECK-NEXT: %[[V5:.*]] = lit.call {{.*}}__getitem__{{.*}}(%anonymous2A_1, %[[C2]])
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A_1
  # CHECK-NEXT: %[[V6:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], [[IMMREF]])
  # CHECK-NEXT: lit.ref.store %[[V5]], %anonymous2A_0
  # CHECK-NEXT: %[[V7:.*]] = lit.call {{.*}}takeInOutInt{{.*}}(%anonymous2A_0)
  # CHECK-NEXT: %anonymous2A_2 = lit.varlet.decl
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: %[[V8:.*]] = lit.call {{.*}}__getitem__{{.*}}(%anonymous2A_2, %b, %[[C1]])
  # CHECK-NEXT: %[[C2:.*]] = {{.*}}constant{{.*}} = 2
  # CHECK-NEXT: %[[V9:.*]] = lit.ref.load %anonymous2A_0
  # CHECK-NEXT: %[[V10:.*]] = lit.call {{.*}}__setitem__{{.*}}(%anonymous2A_2, %[[C2]], %[[V9]])
  # CHECK-NEXT: %[[C1:.*]] = {{.*}}constant{{.*}} = 1
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A_2
  # CHECK-NEXT: %[[V11:.*]] = lit.call {{.*}}__setitem__{{.*}}(%b, %[[C1]], [[IMMREF]])
  takeInOutInt(b[1][2])


@register_passable
struct RegWeirdArray:
    fn __getitem__(self, idx: Int) -> Int:
        return idx
    fn __setitem__(self, idx: Int, value: Int):
        pass


# CHECK-LABEL: lit.func @"dlValueToPValue
fn dlValueToPValue[arr: RegWeirdArray]():
    # CHECK-NEXT: lit.alias.decl {{.*}}x: !Int = <apply({{.*}}@RegWeirdArray::@"__getitem__{{.*}}, {{.*}}arr, #lit.struct<{value = 2}>)>
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
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: %[[KEY:.*]] = kgen.param.constant{{.*}} "some_attr"
    # CHECK: call {{.*}}@DynamicObject::@"__getattr__{{.*}}([[IMMREF]], %[[KEY]])
    _ = obj.some_attr
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: %[[KEY:.*]] = kgen.param.constant{{.*}} "some_attr"
    # CHECK: %[[VALUE:.*]] = kgen.param.constant{{.*}} 42
    # CHECK: call {{.*}}@DynamicObject::@"__setattr__{{.*}}([[IMMREF]], %[[KEY]], %[[VALUE]])
    obj.some_attr = 42


# CHECK-LABEL: lit.func @"chained_cmp
fn chained_cmp(a: Int, b: Int, c: Int, d: Int, e: Int):
    # CHECK-NEXT: %res = lit.varlet.decl "res"
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
# CHECK: lit.alias.decl{{.*}}chainedCmpAlias1: !Bool ={{.*}}false
alias chainedCmpAlias1 = 1 == 2 == 3 == 4 == 5
# CHECK: lit.alias.decl{{.*}}chainedCmpAlias2: !Bool ={{.*}}true
alias chainedCmpAlias2 = 1 <= 2 <= 3 <= 4 <= 5
# CHECK: lit.alias.decl{{.*}}chainedCmpAlias3: !Bool ={{.*}}false
alias chainedCmpAlias3 = 1 <= 2 <= 9 <= 4 <= 5
fn chainedCmpSemiDyn(x: Int, a: Int, b: Int, c: Int):
  # CHECK: [[XCMP:%.*]] = lit.varlet.decl "xCmp"
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
  # CHECK-NEXT:       [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}true
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
  # CHECK-NEXT:   [[TRUEPARAM:%.*]] = kgen.param.constant: !Bool = {{.*}}true
  # CHECK-NEXT:   hlcf.yield [[TRUEPARAM]]
  # CHECK-NEXT: }
  # CHECK-NEXT: lit.ref.store [[FINALRESULT]], [[XCMP]]
  var xCmp = 5 < 77 < x < 105 < 177
  # A fully deep check of this would be a lot of work, but this at least
  # shows that its not choking during parsing on a mix of dynamic and
  # parameter comparisons.  It required some care with the interaction
  # between recursive calls of emitNextCmp calls to get this to work.
  var mixedChain = 0 < 1 < a < 10 < 11 < b < 20 < 21 < c < 30 < 31


fn lvalue_utilities(inout a: Int):
  # Get the address of the specified physical lvalue as a pop.pointer value.
  let addr : __mlir_type[`!kgen.pointer<`,Int,`>`] = __get_lvalue_as_address(a)

  # Get and use an lvalue from an address.
  __get_address_as_lvalue(addr) = 42
  let val = __get_address_as_lvalue(addr)

# CHECK-LABEL: lit.func @"ref_utilities
fn ref_utilities(a: MemoryOnlyInt, inout b: MemoryOnlyInt,
                 inout c: MemoryOnlyInt,
                 cond: __mlir_type.i1):
  # Get the address of the specified physical bvalue or lvalue as a lit.ref.

  # CHECK-NEXT: %ref1 = lit.letreg.decl "ref1" = %a
  let ref1 = __get_ref_from_value(a)
  # CHECK-NEXT: %ref2 = lit.letreg.decl "ref2" = %b
  let ref2 = __get_ref_from_value(b)

  # CHECK-NEXT: [[MV:%.*]] = lit.ref.to_pointer %ref1
  # CHECK-NEXT: %ptr1 = lit.letreg.decl "ptr1" = [[MV]]
  let ptr1 = __mlir_op.`lit.ref.to_pointer`(ref1)

  # CHECK-NEXT: %localLet = lit.varlet.decl "localLet"
  let localLet = MemoryOnlyInt()
  # CHECK: %ref3 = lit.letreg.decl "ref3" = %localLet
  let ref3 = __get_ref_from_value(localLet)

  # CHECK-NEXT: %localVar = lit.varlet.decl "localVar"
  var localVar = MemoryOnlyInt()
  # CHECK: %ref4 = lit.letreg.decl "ref4" = %localVar
  let ref4 = __get_ref_from_value(localVar)

  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%ref2, %a)
  __get_value_from_ref(ref2) = a

  # CHECK-NEXT: [[COMMON:%.*]] = hlcf.if %cond -> !lit.ref<!MemoryOnlyInt, {*"`a", *"`b", *"`c"}> {
  # CHECK-NEXT:   [[COMMONINNER:%.*]] = hlcf.if %cond -> !lit.ref<!MemoryOnlyInt, {*"`a", *"`b"}> {
  # CHECK-NEXT:     [[TMP:%.*]] = kgen.rebind %ref1
  # CHECK-NEXT:     hlcf.yield [[TMP]]
  # CHECK-NEXT:   } else {
  # CHECK-NEXT:     [[TMP:%.*]] = kgen.rebind %ref2
  # CHECK-NEXT:     hlcf.yield [[TMP]]{{.*}}>
  # CHECK-NEXT:   }
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind [[COMMONINNER]]
  # CHECK-SAME:           !lit.ref<!MemoryOnlyInt, {*"`a", *"`b"}> to !lit.ref<!MemoryOnlyInt, {*"`a", *"`b", *"`c"}>
  # CHECK-NEXT:    hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = kgen.rebind %c
  # CHECK-NEXT:   hlcf.yield [[TMP:%.*]]
  # CHECK-NEXT: }
  # CHECK-NEXT: %ref5 = lit.letreg.decl "ref5" = [[COMMON]]
  let ref5 = ref1 if cond else ref2 if cond else __get_ref_from_value(c)

struct CallableStruct:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

    fn __call__(self, rhs: Int) -> Int:
        return self.value + rhs

# CHECK-LABEL: lit.func @"test_call_method()"
fn test_call_method():
    # CHECK: %[[C2:.*]] = kgen.param.constant{{.*}} 2
    # CHECK-NEXT: lit.call {{.*}}@"__call__{{.*}}(%{{.*}}, %[[C2]])
    let value = CallableStruct(5)
    _ = value(2)

struct MemoryType:
  fn __copyinit__(inout self, other: Self):
    pass

@register_passable
struct RegType: pass

# CHECK-LABEL: lit.struct.decl @ParamType
# CHECK-SAME: <[[A:.*]][a]: !Int>
@register_passable
struct ParamType[a: Int]: pass

@value
struct MemType: pass

# CHECK-LABEL: lit.func @"function_types
# CHECK-SAME: %{{.*}}: {{.*}}(!Int borrow, |) -> !Int
# CHECK-SAME: %{{.*}}: {{.*}}("__result__": !lit.ref<mut !MemoryType, {{.*}}> byref_result, !lit.ref<!MemoryType, {{.*}}> borrow_in_mem, |) -> !kgen.none
# CHECK-SAME: %{{.*}}: {{.*}}(!RegType, |) ownedresult -> !RegType
# CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<mut !MemoryType, *[0,0]> owned_in_mem, |) -> !kgen.none
# CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<mut !Int, *[0,0]> byref, |) -> !kgen.none
# CHECK-SAME: %{{.*}}: {{.*}}(!Int borrow, |) throws|ownedresult -> !kgen.variant<!Error, none>
# CHECK-SAME: %{{.*}}: {{.*}}(!Int borrow, |) throws|async|capturing|ownedresult -> !kgen.variant<!Error, none>
# CHECK-SAME: %{{.*}}: {{.*}}(!kgen.variadic<!Int> borrow) throws|vararg|ownedresult -> !kgen.variant<!Error, none>
# CHECK-SAME: %{{.*}}: {{.*}}<"a": !Int>(!kgen.declref<@"$expressions"::@ParamType<:!Int *(0,0)>{{.*}}> borrow, |) -> !kgen.none
# CHECK-SAME: %{{.*}}: {{.*}}<<"a": !Int, "b": @"$expressions"::@ParamType<:!Int *(0,0)>{{.*}}>() throws|ownedresult -> !kgen.variant<!Error, none>
# CHECK-SAME: %{{.*}}: {{.*}}<<"Ts": variadic<regtype>>(!kgen.pack<*(0,0)> borrow) throws|async|packvararg|param_vararg|ownedresult -> !kgen.variant<!Error, none>
# CHECK-SAME: %{{.*}}: {{.*}}<(!Int borrow = #lit.struct<{value = 10}>, !StringLiteral borrow = #lit.struct<{value: string = "foo"}>, |) -> !kgen.none>
# CHECK-SAME: %{{.*}}: {{.*}}<[1]("x": !lit.ref<!MemType, {{.*}}> borrow_in_mem) -> !Int>
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
  float10: def[a: Int, b: ParamType[a]]() -> None,
  float11: async def[*Ts: AnyRegType](* *Ts) -> None,
  float12: fn(Int = 10, StringLiteral = "foo") -> None,
  named: fn(x: MemType) -> Int
): pass

# CHECK-LABEL: lit.struct.decl @Mem
# CHECK-NEXT: lit.alias.decl _{{.*}}_x: regtype = <i8>
# CHECK-NEXT: lit.alias.decl _{{.*}}_B: regtype = <!lit.signature<("foo": i8 borrow) -> !kgen.none>>
struct Mem:
   alias x = __mlir_type.i8
   alias B = fn (foo: Self.x) -> None

alias fn_type_alias = fn() -> None

@always_inline
fn func_with_decorator(): pass


struct TwoParamsStruct[a: Int, b: Int]:
    fn __copyinit__(inout self, other: Self):
        pass

# CHECK-LABEL: lit.func @"variadic_subscript{{.*}})"<
# CHECK-SAME: {{.*}}[idx]: !Int, [[A:.*_a]][a]: variadic<!Int>>
fn variadic_subscript[idx: Int, *a: Int](*b: Int):
    # CHECK-NEXT: %[[LIST:.*]] = lit.call {{.*}}VariadicList{{.*}}__init__
    # CHECK-NEXT: lit.letreg.decl "b" {{.*}}%[[LIST]]
    # CHECK: lit.alias.decl {{.*}}v0: {{.*}}Int = <variadic_get(:variadic<!Int> [[A]], 2)>
    alias v0 = a[2]
    # CHECK: pop.variadic.get %{{.*}}[%index3]
    let v1 = a[3]
    # CHECK: lit.call {{.*}}__getitem__{{.*}}(%b_0,
    let v2 = b[idx]


# CHECK-LABEL: lit.func @"variadic_memory_subscript
# CHECK-SAME: variadic<!lit.ref<{{.*}}TwoParamsStruct<
# CHECK-SAME:   :!Int variadic_get({{.*}}a, 0)
# CHECK-SAME:   :!Int variadic_get({{.*}}a, 1)
fn variadic_memory_subscript[*a: Int](*b: TwoParamsStruct[a[0], a[1]]):
    # CHECK: %v0 = lit.varlet.decl
    # CHECK: [[B1ADDR:%.*]] = {{.*}}__refitem__{{.*}}(%b_0,
    # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%v0, [[B1ADDR]])
    let v0 = b[1]
    # CHECK: %v1 = lit.varlet.decl
    # CHECK: [[B2ADDR:%.*]] = {{.*}}__refitem__{{.*}}(%b_0,
    # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%v1, [[B2ADDR]])
    var v1 = b[2]

fn takeMemory(a: MemoryType): pass

# CHECK-LABEL: lit.func @"testConds
fn testConds(cond: __mlir_type.i1, a: MemoryType, b: MemoryType, m: RegPassable, i: Int) -> MemoryType:
  # Implicit conversions.
  # Mojo Issue #49: https://github.com/modularml/mojo/issues/49

  # CHECK-NEXT: hlcf.if %cond -> !RegPassable {
  # CHECK-NEXT:   [[V:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%m)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[V:%.*]] = lit.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: }
  _ = m if cond else i

  # CHECK-NEXT: hlcf.if %cond -> !RegPassable {
  # CHECK-NEXT:   [[V:%.*]] = lit.call {{.*}}__init__{{.*}}(%i)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[V:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%m)
  # CHECK-NEXT:   hlcf.yield [[V]]
  # CHECK-NEXT: }
  _ = i if cond else m

  # Memory only conds.
  # Issue (#13379)

  # CHECK-NEXT: %anonymous2A = lit.varlet.decl
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
  let a = MemoryOnlyInt()

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(a^^)

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(MemoryOnlyInt()^)

##===----------------------------------------------------------------------===##
# Keyword arguments
##===----------------------------------------------------------------------===##


fn take_kw_args(a: Int, b: Int = 2, c: Int = 3):
    pass


# CHECK-LABEL: lit.func @"test_simple_kw_args()"
fn test_simple_kw_args():
    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-DAG: %[[C:.*]] = kgen.param.constant: {{.*}}value = 3
    # CHECK: lit.call @{{.*}}@"take_kw_args{{.*}}"(%[[A]], %[[B]], %[[C]])
    take_kw_args(5, b=7)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-DAG: %[[C:.*]] = kgen.param.constant: {{.*}}value = 9
    take_kw_args(5, b=7, c=9)
    # CHECK: lit.call @{{.*}}@"take_kw_args{{.*}}"(%[[A]], %[[B]], %[[C]])

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 2
    # CHECK-DAG: %[[C:.*]] = kgen.param.constant: {{.*}}value = 9
    take_kw_args(5, c=9)
    # CHECK: lit.call @{{.*}}@"take_kw_args{{.*}}"(%[[A]], %[[B]], %[[C]])

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-DAG: %[[C:.*]] = kgen.param.constant: {{.*}}value = 9
    # CHECK: lit.call @{{.*}}@"take_kw_args{{.*}}"(%[[A]], %[[B]], %[[C]])
    take_kw_args(5, c=9, b=7)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-DAG: %[[C:.*]] = kgen.param.constant: {{.*}}value = 9
    # CHECK: lit.call @{{.*}}@"take_kw_args{{.*}}"(%[[A]], %[[B]], %[[C]])
    take_kw_args(a=5, c=9, b=7)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-DAG: %[[C:.*]] = kgen.param.constant: {{.*}}value = 9
    # CHECK: lit.call @{{.*}}@"take_kw_args{{.*}}"(%[[A]], %[[B]], %[[C]])
    take_kw_args(c=9, b=7, a=5)

@register_passable("trivial")
struct MyInt:
    var value: Int

    @always_inline("nodebug")
    fn __init__(_a: Int) -> Self:
        return Self {value: _a}


fn overloaded_kw_arg(a: Int, b: MyInt):
    pass


fn overloaded_kw_arg(a: Int, b: Int):
    pass


# CHECK-LABEL: lit.func @"test_kw_args_overload()"
fn test_kw_args_overload():
    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: !Int {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: !Int {{.*}}value = 8
    # CHECK: lit.call @{{.*}}@"overloaded_kw_arg({{.*}}::Int,{{.*}}::Int)"(%[[A]], %[[B]])
    overloaded_kw_arg(b=8, a=5)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: !Int {{.*}}value = 5
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: !MyInt {{.*}}value = 8
    # CHECK: lit.call @{{.*}}@"overloaded_kw_arg({{.*}}::Int,{{.*}}::MyInt)"(%[[A]], %[[B]])
    overloaded_kw_arg(b=MyInt(8), a=5)


fn take_kw_param_infer[A: AnyRegType, B: AnyRegType](a: A, b: B):
    pass


# COM: test parametric overload in the presence of keyword operands.
fn take_kw_param_infer[B: AnyRegType](a: StringLiteral, b: B):
    pass


# CHECK-LABEL: lit.func @"test_kw_args_param_infer()"
fn test_kw_args_param_infer():
    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 1
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value: scalar<f64> = "3.14
    # CHECK: lit.call @{{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:regtype !Int, :regtype !FloatLiteral>(%[[A]], %[[B]])
    take_kw_param_infer(1, b=3.14)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 1
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value: scalar<f64> = "3.14
    # CHECK: lit.call @{{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:regtype !Int, :regtype !FloatLiteral>(%[[A]], %[[B]])
    take_kw_param_infer(a=1, b=3.14)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 1
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value: scalar<f64> = "3.14
    # CHECK: lit.call @{{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:regtype !Int, :regtype !FloatLiteral>(%[[A]], %[[B]])
    take_kw_param_infer[Int, FloatLiteral](a=1, b=3.14)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 1
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value: scalar<f64> = "3.14
    # CHECK: lit.call @{{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:regtype !Int, :regtype !FloatLiteral>(%[[A]], %[[B]])
    take_kw_param_infer(b=3.14, a=1)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 1
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value: scalar<f64> = "3.14
    # CHECK: lit.call @{{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:regtype !Int, :regtype !FloatLiteral>(%[[A]], %[[B]])
    take_kw_param_infer[Int](b=3.14, a=1)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value: string = "hello"
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 3
    # CHECK: lit.call @"{{.*}}@"take_kw_param_infer[AnyRegType]{{.*}}"<:regtype !Int>(%[[A]], %[[B]])
    take_kw_param_infer("hello", b=3)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value: string = "hello"
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 3
    # CHECK: lit.call @"{{.*}}@"take_kw_param_infer[AnyRegType]{{.*}}"<:regtype !Int>(%[[A]], %[[B]])
    take_kw_param_infer(b=3, a="hello")


fn kw_args_callable(a: Int, b: Int = 7):
    pass


struct KwCallable:
    fn __init__(inout self):
        pass

    fn __call__(self, msg: StringLiteral, n: Int = 5):
        pass


# CHECK-LABEL: lit.func @"indirect_kw_args()"
fn indirect_kw_args():
    # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
    alias callee = kw_args_callable

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 9
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK: lit.call_param[{{.*}} [[CALLEE]]](%[[A]], %[[B]])
    callee(a=9)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 4
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 5
    # CHECK: lit.call_param[{{.*}} [[CALLEE]]](%[[A]], %[[B]])
    callee(4, b=5)

    # CHECK-DAG: %[[A:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-DAG: %[[B:.*]] = kgen.param.constant: {{.*}}value = 2
    # CHECK: lit.call_param[{{.*}} [[CALLEE]]](%[[A]], %[[B]])
    callee(b=2, a=7)

    # CHECK: %[[CALLABLE:.*]] = lit.varlet.decl {{.*}}: !lit.ref<mut !KwCallable
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %[[CALLABLE]]
    # CHECK-NEXT: %[[MSG:.*]] = kgen.param.constant: {{.*}}value: string = "woof"
    # CHECK-NEXT: %[[N:.*]] = kgen.param.constant: {{.*}}value = 7
    # CHECK-NEXT: lit.call @{{.*}}@KwCallable::@"__call__{{.*}}([[IMMREF]], %[[MSG]], %[[N]])
    KwCallable()(n=7, msg="woof")

##===----------------------------------------------------------------------===##
# Test Type Expressions
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"type_function
# CHECK-SAME: (%a: !Bool borrow) -> !kgen.anyregtype
fn type_function(a: Bool) -> AnyRegType:
    # CHECK: [[TYPE:%.*]] = hlcf.if %{{.*}} -> !kgen.anyregtype
    # CHECK-NEXT: %metatype = kgen.param.constant: metatype<{{.*}}@Int> = <!Int>
    # CHECK-NEXT: [[COERCED:%.*]] = lit.call {{.*}}(%metatype)
    # CHECK-NEXT: yield [[COERCED]]
    # CHECK-NEXT: else
    # CHECK-NEXT: %regtype = kgen.param.constant: regtype = <!Bool>
    # CHECK-NEXT: yield %regtype
    # CHECK: return [[TYPE]] : !kgen.anyregtype
    return rebind[AnyRegType](Int) if a else Bool


# CHECK-LABEL: lit.func @"static_type
# CHECK-SAME: <[[PARAM:.*]][a]: !Bool>
# CHECK-SAME: %x: !kgen.paramref<apply(:!lit.signature<("a": !Bool borrow) -> !kgen.anyregtype> {{.*}}@"type_function{{.*}}, [[PARAM]])> borrow)
fn static_type[a: Bool](x: type_function(a)):
    pass

##===----------------------------------------------------------------------===##
# Test nonmaterializable IntLiteral beyond Int bounds.
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl{{.*}}bigggNumber: !IntLiteral = <#lit.struct<{value: !kgen.int_literal = 115792089237316195423570985008687907853269984665640564039457584007913129639936}>>
alias bigggNumber = 2 << 255
fn useBigNumber() -> Int:
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 512}>>
  # CHECK-NEXT: [[DECL:%.*]] lit.letreg.decl "notSoBig" = [[VAR]] : !Int
  let notSoBig = bigggNumber // (2 << 246)
  # Easy min-int
  # CHECK-NEXT: [[VAR:%.*]] = kgen.param.constant: !Int = <#lit.struct<{value = -9223372036854775808}>>
  # CHECK: [[DECL:%.*]] lit.letreg.decl "minInt" = [[VAR]] : !Int
  let minInt = -(2<<62)
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
   # CHECK-NEXT: %x = lit.varlet.decl "x"
   # CHECK-NEXT: lit.call {{.*}}(%x, %a)
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
  _ = x^  # Consume LValue ok

  let y : Int = 100
  # expected-warning @+1 {{transfer from a trivial register value has no effect and can be removed}}
  _ = y^  # Consume RValue / BValue is not, this isn't tracked.




fn del_warnings():
  # These copy the value before destroying it, which is pointless.
  let m = MemoryOnlyInt()
  m.__del__()  # expected-warning {{explicit call to '__del__' destroys a copy of the value; consider removing this call}}
  let r = RegPassable(1)
  r.__del__()  # expected-warning {{explicit call to '__del__' destroys a copy of the value; consider removing this call}}

  # These is wierd/unneeded, but at least it does what it says.
  MemoryOnlyInt().__del__()
  RegPassable(1).__del__()

# [QoI] Generate error for obviously self recursive functions
# https://github.com/modularml/mojo/issues/222
fn self_recursive():
  self_recursive() # expected-warning {{self recursive call will cause an infinite loop}}

fn self_recursive_arg(a: Int):
  self_recursive_arg(a) # expected-warning {{self recursive call will cause an infinite loop}}

  if a != 0:
    self_recursive_arg(a-1)  # No warning

fn self_recursive_param[a: Int]():
  self_recursive_param[a]() # expected-warning {{self recursive call will cause an infinite loop}}

  @parameter
  if a != 400:
    self_recursive_param[a+1]() # No warning

fn self_recursive_impl_lifetime(inout a: MemoryOnlyInt):
  self_recursive_impl_lifetime(a) # expected-warning {{self recursive call will cause an infinite loop}}
