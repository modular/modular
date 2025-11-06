# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s | FileCheck %s

# CHECK: module {

fn noop(): pass

@register_passable("trivial")
struct DType:
    alias type = __mlir_type.`!kgen.dtype`
    var _mlir_value: Self.type

    alias float32 = __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
    alias int32 = __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`
    alias float64 = __mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self.type):
        self._mlir_value = value


alias Float32 = SIMD[DType.float32, 1]
alias Float64 = SIMD[DType.float64, 1]

# CHECK-LABEL: lit.struct.decl @SIMD
# CHECK-SAMEL <[[SIMDDT:.*]]: !DType, [[SIMDSIZE:.*]]: !Int>
# CHECK-SAME: register_passable
@register_passable("trivial")
struct SIMD[dtype: DType, size: Int]:
    alias _mlir_type = __mlir_type[
        `!pop.simd<`, size._mlir_value, `, `, dtype._mlir_value, `>`
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the vector."""

    @always_inline("nodebug")
    fn __init__(out self, *, mlir_value: Self._mlir_type):
        self._mlir_value = mlir_value

    @always_inline("nodebug")
    fn __init__(out self):
        alias res = SIMD[dtype, size](Int())
        self = res

    @always_inline
    fn __init__(out self, value: Int, /):
        var index = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<index>`
        ](value._mlir_value)
        var s = __mlir_op.`pop.cast`[_type = SIMD[dtype, 1]._mlir_type](index)

        @parameter
        if size == 1:
            self._mlir_value = rebind[Self._mlir_type](s)
        else:
            self._mlir_value = __mlir_op.`pop.simd.splat`[
                _type = Self._mlir_type
            ](s)

    @implicit
    fn __init__(out self, value: FloatLiteral, /):
        var res = __mlir_attr[
            `#pop<float_literal_convert<`, value.value, `>> : `, Self._mlir_type
        ]
        self = Self(mlir_value=res)

    fn __add__(lhs, rhs: Self) -> Self:
        while __mlir_attr.true:
            pass

    @staticmethod
    fn splat():
        pass

    @always_inline("nodebug")
    fn __truediv__(self, rhs: Self) -> Self:
        return Self(
            mlir_value=__mlir_op.`pop.div`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    fn __rtruediv__(self, value: Self) -> Self:
        return value / self

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        self = self + rhs

# CHECK-LABEL: lit.struct.decl @MemoryOnlyInt
struct MemoryOnlyInt(ImplicitlyCopyable):
  var x: Int

  # CHECK-LABEL: lit.fn @"__init__
  @implicit
  fn __init__(out self, a: Int = 42):
    # CHECK: %0 = lit.ref.struct.ger %self[x]
    # CHECK: %1 = {{.*}}constant: !Int = <{1}>
    # CHECK: lit.ref.store %1, %0
    self.x = 1
  fn __del__(deinit self): pass

  # CHECK-LABEL: lit.fn @"__copyinit__
  fn __copyinit__(out self, other: Self):
    self.x = other.x

  @staticmethod
  fn variadic(*value: MemoryOnlyInt):
    pass

fn consume(var a: MemoryOnlyInt): pass

# This type is used to test implicit conversion from MemoryOnlyInt
struct MemoryOnlyFloat64:
  var x: Float64
  @implicit
  fn __init__(out self, value: MemoryOnlyInt):
    self.x = 1.0

# CHECK-LABEL: lit.struct.decl @MemoryOnlyPair
struct MemoryOnlyPair(ImplicitlyCopyable):
  var x: MemoryOnlyInt
  var y: Int

  # CHECK: lit.fn @"__copyinit__{{.*}}(%other: !lit.ref<!MemoryOnlyPair, imm {{.*}}> read_mem,
  # CHECK-SAME: %self: !lit.ref<!MemoryOnlyPair, mut {{.*}}> byref_result)
  fn __copyinit__(out self, other: MemoryOnlyPair):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %self[x]
    # CHECK-NEXT: %1 = lit.ref.struct.ger %other[x]
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%1, %0)
    # CHECK-NEXT: [[SY:%.*]] = lit.ref.struct.ger %self[y]
    # CHECK-NEXT: [[OY:%.*]] = lit.ref.struct.ger %other[y]
    # CHECK-NEXT: [[OY_VAL:%.*]] = lit.ref.load [[OY]]
    # CHECK-NEXT: lit.ref.store [[OY_VAL]], [[SY]]
    self.x = other.x
    self.y = other.y

  # CHECK: lit.fn @"method{{.*}}(
  # CHECK-SAME: %self: !lit.ref<!MemoryOnlyPair, mut {{.*}}> owned_in_mem,
  # CHECK-SAME: %arg: !lit.ref<!MemoryOnlyInt, mut {{.*}}> owned_in_mem)
  fn method(var self, var arg: MemoryOnlyInt):
    # CHECK: %0 = lit.ref.struct.ger %self[y]
    # CHECK: %1 = lit.ref.struct.ger %arg[x]
    # CHECK: %2 = lit.ref.load %0
    # CHECK: %3 = lit.ref.load %1
    # CHECK: %4 = lit.call @{{.*}}__add__{{.*}}"(%2, %3)
    _ = self.y+arg.x

fn inferred_function_with_memory_result[
  width: Int](x: SIMD[DType.float32, width]) -> MemoryOnlyInt: pass

# CHECK-LABEL: lit.fn @"memoryOnlyOps
fn memoryOnlyOps(mut a: MemoryOnlyPair) -> MemoryOnlyPair:
  # CHECK-NEXT: %v1 = lit.var.decl {{.*}} var : !lit.ref<!MemoryOnlyPair,
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], %v1)
  var v1 = a

  # CHECK-NEXT: %v2 = lit.var.decl "v2"
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], %v2)
  var v2 : MemoryOnlyPair = a

  # CHECK-NEXT: lit.ownership.use %a
  _ = a

  a  # expected-warning {{'MemoryOnlyPair' value is unused}}

  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: %regX = lit.var.decl {{.*}}
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[AX]]
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], %regX)
  var regX = a.x

  # CHECK-NEXT: [[AX:%.*]] = lit.ref.struct.ger %a[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], [[AX]])
  a.x = regX

  # Pass memory only things by value as arguments.

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %a
  # CHECK-NEXT: [[TMPPAIR:%.*]] = lit.var.decl {{.*}}!MemoryOnlyPair
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[IMMREF]], [[TMPPAIR]])
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[TMPINT:%.*]] = lit.var.decl {{.*}}!MemoryOnlyInt
  # CHECK-NEXT: lit.call @{{.*}}@"__copyinit__{{.*}}([[IMMREF]], [[TMPINT]])
  # CHECK-NEXT: lit.call @{{.*}}@"method{{.*}}([[TMPPAIR]], [[TMPINT]])
  a.method(regX)

  # Drill into rvalue without cloning intermediate values.
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[V2XX:%.*]] = lit.ref.struct.ger [[V2X]][x]
  # CHECK-NEXT: %v2xx = lit.var.decl "v2xx"
  # CHECK-NEXT: [[VAL:%.*]] = lit.ref.load [[V2XX]]
  # CHECK-NEXT: lit.ref.store [[VAL]], %v2xx
  var v2xx = v2.x.x

  # Implicit conversion between memory-only types.
  # CHECK-NEXT: %mpFloat = lit.var.decl
  # CHECK-NEXT: [[V2X:%.*]] = lit.ref.struct.ger %v2[x]
  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[V2X]]
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}([[IMMREF]], %mpFloat)
  var mpFloat : MemoryOnlyFloat64 = v2.x

  # CHECK-NEXT: [[SIMDVAL:%.*]] = lit.call {{.*}}SIMD::@"__init__{{.*}}()

  # CHECK: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call @{{.*}}inferred_function_with_memory_result{{.*}}([[SIMDVAL]], [[TMP]])
  _ = inferred_function_with_memory_result(SIMD[DType.float32, 4]())
  # CHECK-NEXT: lit.ownership.use [[TMP]]

  # Memory-only default argument with memory-only result.
  # CHECK-NEXT: %[[C42:.*]] = {{.*}}constant: !Int = <{42}>
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-NEXT: lit.call @{{.*}}__init__{{.*}}(%[[C42]], [[TMP]])
  _ = MemoryOnlyInt()
  # CHECK-NEXT: lit.ownership.use [[TMP]]

  # CHECK-NEXT: [[IMMREF1:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[IMMREF2:%.*]] = lit.ref.immut %regX
  # CHECK-NEXT: [[VARIADIC:%.*]] = pop.variadic.create [[[IMMREF1]], [[IMMREF2]]]
  # CHECK-NEXT: lit.call @{{.*}}variadic{{.*}}([[VARIADIC]])
  MemoryOnlyInt.variadic(regX, regX)

  # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %v2
  # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], %__result__)
  # CHECK-NEXT: [[NONEVAL:%.*]] = kgen.param.constant: none = <#kgen.none>
  # CHECK-NEXT: lit.return [[NONEVAL]]
  return v2

struct DirectInit:
  fn __init__(out self):
    pass

fn direct_call_init():
  var value: DirectInit
  # This is a call of a static method on an instance, so 'value' is unused.
  # expected-warning @+1 {{'DirectInit' value is unused}}
  value.__init__()

struct DummyFunc:
    @implicit
    fn __init__(out self, f: def(Int)):
        pass

fn func_arg_conversion(f: DummyFunc): pass

# CHECK-LABEL: lit.fn @"implicit_func_conversion()"
fn implicit_func_conversion():
    def take_int(x: Int):
        pass

    # CHECK: %f = lit.var.decl "f"
    # CHECK: [[CLOSURE:%.*]] = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}([[CLOSURE]], %f)
    var f: DummyFunc = take_int
    # CHECK: [[CLOSURE:%.*]] = kgen.create_closure
    # CHECK: call {{.*}}DummyFunc::@"__init__{{.*}}([[CLOSURE]], %__call_result_tmp__)
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %__call_result_tmp__
    # CHECK: call {{.*}}func_arg_conversion{{.*}}([[IMMREF]])
    func_arg_conversion(take_int)

# CHECK-LABEL: lit.struct.decl @RegPassable
@register_passable
struct RegPassable(ImplicitlyCopyable):
  var value: Int
  # CHECK-LABEL: lit.fn @"__init__
  # CHECK-NEXT: %self = lit.var.decl "self" initoutarg
  # CHECK-NEXT: [[VALREF:%.*]] = lit.ref.struct.ger %self[value]
  # CHECK-NEXT: lit.ref.store %value, [[VALREF]]
  # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
  # CHECK-NEXT: lit.return [[TMP]]
  @implicit
  fn __init__(out self, value: Int):
    self.value = value

  fn __del__(deinit self): pass
  fn __neg__(self) -> Self: pass
  fn __add__(self, rhs: Self) -> Self: pass
  fn __matmul__(self, rhs: Self) -> Self: pass
  fn __rmatmul__(lhs, rhs: Self) -> Self: pass

# CHECK-LABEL: lit.struct.decl @StructWithFuncParam<comparator: !lit.generator
# CHECK-SAME: <"T": type>(!kgen.param<*(0,0)>, |)
struct StructWithFuncParam[comparator: fn[T: AnyTrivialRegType] (T) -> None]:
    # CHECK-LABEL: lit.fn @"f
    # CHECK-SAME: %self: !lit.ref<{{.*}}<:!lit.generator<<"T": type>(!kgen.param<*(0,0)>
    fn f(self):
        pass

    # CHECK-LABEL: lit.fn @"g
    fn g(self):
        # CHECK: call {{.*}}[imm *"self`2x"]<:!lit.generator<<"T": type>(!kgen.param<*(0,0)>, |)
        # CHECK-SAME: !lit.ref<{{.*}}<"T": type>(!kgen.param<*(0,0)>, |)
        self.f()

# CHECK-LABEL: lit.fn @"simpleMath
fn simpleMath(a: Int, b: Int) -> Int:
  # CHECK: %0 = lit.call {{.*}}Int::@"__mul__{{.*}}(%b, %a)
  # CHECK: %1 = lit.call {{.*}}Int::@"__sub__{{.*}}(%a, %0)
  # CHECK: lit.return %1 : !Int
  return a-b*a

# CHECK-LABEL: lit.fn @"precedence_associativity
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


struct LHS:
  @implicit
  fn __init__(out self, value: Int):
    pass

struct RHS(Movable, ImplicitlyCopyable):
  fn __radd__(self, lhs: LHS) -> RHS: return self
  fn __rsub__(self, lhs: LHS) -> RHS: return self
  fn __rmul__(self, lhs: LHS) -> RHS: return self
  fn __rfloordiv__(self, lhs: LHS) -> RHS: return self
  fn __rmod__(self, lhs: LHS) -> RHS: return self
  fn __rpow__(self, lhs: LHS) -> RHS: return self
  fn __rlshift__(self, lhs: LHS) -> RHS: return self
  fn __rrshift__(self, lhs: LHS) -> RHS: return self
  fn __rand__(self, lhs: LHS) -> RHS: return self
  fn __ror__(self, lhs: LHS) -> RHS: return self
  fn __rxor__(self, lhs: LHS) -> RHS: return self

# CHECK-LABEL: lit.fn @"reverse_operators
fn reverse_operators(a: RHS):
  # CHECK: lit.call {{.*}}RHS::@"__radd__(expressions::RHS,expressions::LHS)"
  var z = Int(1) + a

  # CHECK: lit.call {{.*}}RHS::@"__rsub__(expressions::RHS,expressions::LHS)"
  z = Int(2) - z

  # CHECK: lit.call {{.*}}RHS::@"__rmul__(expressions::RHS,expressions::LHS)"
  z = Int(3) * z

  # div tests
  # CHECK: lit.call {{.*}}__rtruediv__
  # CHECK: lit.call {{.*}}RHS::@"__rfloordiv__(expressions::RHS,expressions::LHS)"
  var r1 = 33.0 / Float32(42.0)
  z = Int(33) // z

  # CHECK: lit.call {{.*}}RHS::@"__rmod__(expressions::RHS,expressions::LHS)"
  var i0 = Int(10) % z

  # CHECK: lit.call {{.*}}RHS::@"__rpow__(expressions::RHS,expressions::LHS)"
  var i1 = Int(3) ** z

  # CHECK: lit.call {{.*}}RHS::@"__rlshift__(expressions::RHS,expressions::LHS)"
  var i2 = Int(1) << z

  # CHECK: lit.call {{.*}}RHS::@"__rrshift__(expressions::RHS,expressions::LHS)"
  var i3 = Int(1) >> z

  # CHECK: lit.call {{.*}}RHS::@"__rand__(expressions::RHS,expressions::LHS)"
  z = Int(1) & z

  # CHECK: lit.call {{.*}}RHS::@"__ror__(expressions::RHS,expressions::LHS)"
  z = Int(2) | z

  # CHECK: lit.call {{.*}}RHS::@"__rxor__(expressions::RHS,expressions::LHS)"
  z = Int(3) ^ z

# CHECK-LABEL: lit.fn @"precedence_matmul
fn precedence_matmul(z: RegPassable) -> RegPassable:
  # CHECK:  [[THREE:%.*]] = kgen.param.constant: !Int = <{3}>
  # CHECK-NEXT:  [[THREERP:%.*]] = lit.call {{.*}}@RegPassable::@"__init__{{.*}}([[THREE]])
  # CHECK-NEXT:  [[TWO:%.*]] = kgen.param.constant: !Int = <{2}>
  # CHECK-NEXT:  [[TWORP:%.*]] = lit.call {{.*}}@RegPassable::@"__init__{{.*}}([[TWO]])
  # CHECK-NEXT:  [[TWOTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[TWORP]], [[TWOTMP]]
  # CHECK-NEXT:  [[TWOTMP_IMM:%.*]] = lit.ref.immut [[TWOTMP]]
  # CHECK-NEXT:  [[NEG:%.*]] = lit.call {{.*}}@RegPassable::@"__neg__{{.*}}([[TWOTMP_IMM]])

  # CHECK-NEXT:  [[THREETMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[THREERP]], [[THREETMP]]

  # CHECK-NEXT:  [[NEGTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[NEG]], [[NEGTMP]]
  # CHECK-NEXT:  [[THREETMP_IMM:%.*]] = lit.ref.immut [[THREETMP]]
  # CHECK-NEXT:  [[NEGTMP_IMM:%.*]] = lit.ref.immut [[NEGTMP]]
  # CHECK-NEXT:  [[MATMUL:%.*]] = lit.call {{.*}}@RegPassable::@"__matmul__{{.*}}([[THREETMP_IMM]], [[NEGTMP_IMM]])
  # CHECK-NEXT:  [[MMTMP:%.*]] = lit.var.decl "anonymous*"
  # CHECK-NEXT:  lit.ref.store [[MATMUL]], [[MMTMP]]
  # CHECK-NEXT:  [[MMTMP_IMM:%.*]] = lit.ref.immut [[MMTMP]]
  # CHECK-NEXT:  [[ADD:%.*]] = lit.call {{.*}}@RegPassable::@"__add__{{.*}}(%z, [[MMTMP_IMM]])
  # CHECK-NEXT:  lit.return [[ADD]] : !RegPassable
  return z + RegPassable(3) @ -RegPassable(2)

# CHECK-LABEL: lit.fn @"precedence_bitwise
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

trait Boolable:
    fn __bool__(self) -> Bool:
        ...

@register_passable
struct Boolish(Boolable, ImplicitlyCopyable):
  fn __copyinit__(out self, existing: Self): pass
  fn __bool__(self) -> Bool: return True

struct MemBoolish(ImplicitlyCopyable):
  @implicit
  fn __init__(out self, value: Boolish): pass
  fn __copyinit__(out self, other: Self): pass
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

# CHECK-LABEL: lit.fn @"andOr1
fn andOr1(a: Boolish, b: Boolish):
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: [[BOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BOOL]])
  # CHECK: hlcf.if [[I1]] -> !Boolish {
  # CHECK:   = lit.call {{.*}}__copyinit__{{.*}}(%b)
  # CHECK:   hlcf.yield
  # CHECK: } else {
  # CHECK:   [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK:   hlcf.yield [[TMP]]
  # CHECK: }
  _ = a and b


# CHECK-LABEL: lit.fn @"andOr2
fn andOr2(a: Boolish, b: Boolish):
  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.  Boolish is defined with copy ctor so it must be invoked.

  # CHECK: [[ABOOL:%.*]] = lit.call {{.*}}Boolish::@"__bool__{{.*}}(
  # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}@Bool::@"__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !Boolish {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}Boolish::@"__copyinit__{{.*}}(%a)
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: } else {
  # CHECK:        [[TMP:%.*]] = lit.call {{.*}}Boolish::@"__copyinit__{{.*}}(%b)
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a or b

# CHECK-LABEL: lit.fn @"andOr3
fn andOr3(a: Boolish, c: Bool):
  # Testing two different logic'y types returns the common bool type if present.

  # CHECK: [[ABOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%a)
  # CHECK-NEXT: [[I1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[ABOOL]])
  # CHECK-NEXT:  = hlcf.if [[I1]] -> !Bool {
  # CHECK-NEXT:   hlcf.yield %c
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}([[I1]])
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK-NEXT: }
  _ = a and c

# CHECK-LABEL: lit.fn @"andOr4
fn andOr4(b: Boolish, c: Bool):
  # Check incompatible types that are nevertheless boolish.
  # CHECK: [[BBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%b)
  # CHECK-NEXT: [[BI1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[BBOOL]])
  # CHECK-NEXT: = hlcf.if [[BI1]] -> !Bool {
  # CHECK-NEXT:   [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}([[BI1]])
  # CHECK:        hlcf.yield [[TMP]]
  # CHECK: } else {
  # CHECK-NEXT: hlcf.yield %c : !Bool
  # CHECK-NEXT: }
  _ = b or c

# CHECK-LABEL: lit.fn @"andOr2
fn andOr2(b: Boolish, d: MemBoolish):
  # Check memory-only boolish types.
  # Boolish and MemBoolish has a common type of MemBoolish.

  # CHECK: [[DBOOL:%.*]] = lit.call {{.*}}__bool__{{.*}}(%d)
  # CHECK-NEXT: [[DI1:%.*]] = lit.call {{.*}}__mlir_i1__{{.*}}([[DBOOL]])
  # CHECK-NEXT: [[IFRESULT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!MemBoolish
  # CHECK-NEXT: hlcf.if [[DI1]] {
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}(%d, [[ANON:%.*]])
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: } else {
  # CHECK-NEXT:   [[TMPMEM:%.*]] = lit.var.decl
  # CHECK-NEXT:   lit.call {{.*}}__init__{{.*}}(%b, [[TMPMEM]])
  # CHECK-NEXT:   [[IMMREF:%.*]] = lit.ref.immut [[TMPMEM]]
  # CHECK-NEXT:   lit.call {{.*}}__copyinit__{{.*}}([[IMMREF]], [[ANON]])
  # CHECK-NEXT:   hlcf.yield
  # CHECK-NEXT: }
  _ = d or b

# CHECK-LABEL: lit.fn @"paramAndOr{{.*}}"<a: !Boolish, b: !Boolish>
fn paramAndOr[a: Boolish, b: Boolish]():
  # Short circuiting AND returns second operand when the first is false-y, first
  # otherwise.

  # CHECK: lit.alias.decl *"c{{.*}}": !Boolish = <cond(
  # CHECK-SAME: apply({{.*}}Boolish::@"__bool__{{.*}}"), store_to_mem(a)), "_mlir_value">{{.*}}, b, a)>
  alias c = a and b

  # Short circuiting OR returns first operand when it is true-y, second
  # otherwise.

  # CHECK: lit.alias.decl *"d{{.*}}": !Boolish = <cond({{.*}}apply({{.*}}Boolish::@"__bool__{{.*}}"), store_to_mem(a)), "_mlir_value">{{.*}}, a, b)>
  alias d = a or b

# CHECK-LABEL: lit.fn @"do_math
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

# CHECK-LABEL: lit.fn @"test_if_cond
fn test_if_cond(var cond: Bool, memCond: MemBoolish):
    # CHECK: %i = lit.var.decl "i"
    # CHECK: %[[COND:.*]] = lit.ref.load %cond
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
    # CHECK-NEXT: lit.ref.store [[TRUEB]], %cond
    cond = True
    i += i
    if cond:     # 'if' stmt, not an 'if' expression.
        i += 1

# CHECK-LABEL: lit.fn @"test_param_if_cond{{.*}}"<cond: !Bool>
fn test_param_if_cond[cond: Bool]() -> Int:
  # CHECK-NEXT: lit.alias.decl [[I_ALIAS:.*]]: !Int = <cond(#lit.struct.extract<:!Bool cond, "_mlir_value">, {2}, {3})>
  alias i = 2 if cond else 3

  # CHECK-NEXT: lit.alias.decl *"j{{.*}} = <cond({{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">
  # CHECK-SAME: :!pop.float_literal #pop.float_literal<2|1>{{.*}}:!pop.int_literal 3>
  alias j = 2.0 if cond else 3

  # CHECK-NEXT: %[[I:.*]] = kgen.param.constant: !Int = <sugar_alias(*"i`", cond(#lit.struct.extract<:!Bool cond, "_mlir_value">, {2}, {3}))>
  return i

# CHECK-LABEL: lit.fn @"callable_mv[fn(::Int, /) -> ::Int](::Int)"
# CHECK-SAME: <callable: !lit.generator<(!Int, |) -> !Int>>(%a: !Int) -> !Int
fn callable_mv[callable: fn (Int) -> Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call[!lit.generator<(!Int, |) -> !Int>: callable](%a)
  return callable(a)

# CHECK-LABEL: lit.fn @"callable_mv_inputs{{.*}})"<
# CHECK-SAME: callable: !lit.generator<<"x": !Int>(!Int, |) -> !Int>, b: !Int>(%a: !Int) -> !Int
fn callable_mv_inputs[callable: fn[x: Int](Int) -> Int, b: Int](a: Int) -> Int:
  # CHECK-NEXT: lit.call[!lit.generator<(!Int, |) -> !Int>: bind_params({{.*}}callable, b)](%a)
  return callable[b](a)

# CHECK-LABEL: lit.fn @"takeIndexParam{{.*}}"<a: !Int>() -> !Int
fn takeIndexParam[a: Int]() -> Int:
  return a + 1

# CHECK-LABEL: lit.fn @"returnIndex()"() -> !Int
fn returnIndex() -> Int:
  return 0

# CHECK-LABEL: lit.fn @"returnIndex2()"() -> !Int
fn returnIndex2() -> Int:
  # CHECK-NEXT: %0 = lit.call @{{.*}}takeIndexParam{{.*}}"<:!Int apply({{.*}}@{{.*}}returnIndex()")>()
  # CHECK-NEXT: return %0
  return takeIndexParam[returnIndex()]()

# CHECK-LABEL: lit.fn @"callInParam[fn[::Int](::Int, /) -> ::Int]()"
# CHECK-SAME: <callable: !lit.generator<<"x": !Int>(!Int, |) -> !Int>>() -> !Int
fn callInParam[callable: fn[x: Int](Int) -> Int]() -> Int:
  # CHECK-NEXT: %0 = lit.call @{{.*}}takeIndexParam{{.*}}()"<:!Int apply({{.*}}bind_params({{.*}}callable, {1}), {1})>()
  # CHECK-NEXT: return %0
  return takeIndexParam[callable[1](1)]()

# CHECK-LABEL: lit.fn @"parameterExprs{{.*}}()"
# CHECK-SAME: <a: !Int, a2: !Int>
fn parameterExprs[a: Int, a2: Int]():
  # CHECK: lit.alias.decl *"b{{.*}}": !Int = <{0}>
  alias b = a-a
  # CHECK: lit.alias.decl *"c{{.*}}": !Int = <{{.*}}{_mlir_value = add(#lit.struct.extract<:!Int a, "_mlir_value">, 42)}
  alias c = a+42
  # CHECK: lit.alias.decl *"d{{.*}}": !Int = <{{.*}}{_mlir_value = mul(#lit.struct.extract<:!Int a, "_mlir_value">, #lit.struct.extract<:!Int a2, "_mlir_value">)}
  alias d = a*a2

##===----------------------------------------------------------------------===##
# Patterns, LValues and RValues
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"patterns()
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
  (_) = 1.0

  # CHECK: %someFloat32 = lit.var.decl "someFloat32" var
  # CHECK: [[Float32:%.*]] = lit.ref.load %someFloat32
  # CHECK: {{%.*}} = lit.call {{.*}}__iadd__{{.*}}(%someFloat32, [[Float32]])
  var someFloat32 : Float32
  (someFloat32) += someFloat32

  # CHECK: %someSIMD = lit.var.decl "someSIMD" var
  # CHECK: [[SIMD:%.*]] = lit.ref.load %someSIMD
  # CHECK: {{%.*}} = lit.call {{.*}}@expressions::@SIMD::@"__iadd__({{.*}}(%someSIMD, [[SIMD]])
  var someSIMD : SIMD[DType.float64, 4]
  (someSIMD) += someSIMD

# CHECK-LABEL: lit.fn @"byval_byref_function(::Int,::Int&)"{{.*}}(%a: !Int, %b: !lit.ref<!Int, mut {{.*}}> mut) -> !kgen.none
fn byval_byref_function(a: Int, mut b: Int):
  # CHECK-NEXT: lit.ref.store %a, %b
  b = a

  # CHECK-NEXT: %x = lit.var.decl "x" var
  var x : Int
  # This needs to load 'b' to pass it by value for the first arg, but pass its
  # address in directly for the second.
  # CHECK: [[TMP:%.*]] = lit.ref.load %b
  # CHECK: = lit.call @{{.*}}::@"byval_byref_function{{.*}}([[TMP]], %b)
  byval_byref_function(b, b)

# CHECK-LABEL: lit.fn @"lvaluesAndRValues()
fn lvaluesAndRValues() -> __mlir_type.index:
  # CHECK: [[VALUE:%.*]] = kgen.param.constant = <4>
  # CHECK: lit.return [[VALUE]] : index
  return Int(4)._mlir_value

# CHECK-LABEL: lit.fn @"mvalueStructField()"
fn mvalueStructField():
  # CHECK: lit.alias.decl [[INT:.*]]: !Int = <{4}>
  alias Index = Int(4)
  # CHECK: lit.alias.decl *"value{{.*}}" = <4>
  alias value = Index._mlir_value
  alias foldToValue = Int(5)._mlir_value

