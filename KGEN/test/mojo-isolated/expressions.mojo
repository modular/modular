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

    fn __truediv__(self, rhs: Self) -> Self:
        while __mlir_attr.true:
            pass

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
