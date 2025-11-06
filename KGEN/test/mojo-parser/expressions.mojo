# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -verify-diagnostics -import-mojo %s | FileCheck %s

from memory import Pointer
from collections.string import StaticString

# CHECK: module {

fn noop(): pass

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

# CHECK-LABEL: lit.fn @"simpleMath
fn simpleMath(a: Int, b: Int) -> Int:
  # CHECK: %0 = lit.call {{.*}}Int::@"__mul__{{.*}}(%b, %a)
  # CHECK: %1 = lit.call {{.*}}Int::@"__sub__{{.*}}(%a, %0)
  # CHECK: lit.return %1 : !Int
  return a-b*a

##===----------------------------------------------------------------------===##
# Augmented Assignments
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"basic_assignments
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
# CHECK-LABEL: lit.fn @"walrus_implicit_decl
def walrus_implicit_decl():
  # CHECK:      %d = lit.var.decl "d" imp
  # CHECK:      %c = lit.var.decl "c" imp
  # CHECK:      %b = lit.var.decl "b" imp
  # CHECK:      %a = lit.var.decl "a" imp

  # CHECK-NEXT: [[THREE:%.*]] = kgen.param.constant: !Int = <{3}>
  # CHECK-NEXT: lit.ref.store [[THREE]], %a
  # CHECK-NEXT: [[VAR_A:%.*]] = lit.ref.load %a
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}simpleMath{{.*}}([[THREE]], [[VAR_A]])
  _ = simpleMath(a := 3, a)
  # CHECK-NEXT: lit.ownership.use [[TMP]]

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

# CHECK-LABEL: lit.fn @"literals
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
    alias b = 1_2.3__1e+1_1 # CHECK: #pop.float_literal<1231000000000|1>
    c = False         # CHECK: !Bool = <{:i1 0}>
    c = True          # CHECK: !Bool = <{:i1 1}>

# CHECK-LABEL: lit.fn @"_strings
fn _strings():
   """
      Various tests on strings
   """

    var a = ""                 # CHECK: ""
    # CHECK: "hello world"
    var a2 = "hello \
world"

    # COM: match newline hex values via regex since they vary between OSs
    # CHECK: "hello \\{{[\\0-9A-Z]+}}world"
    var a3 = r"hello \
world"

    # CHECK:  "1'{{(\\0D)?}}\0A2"
    var a4 = """1'
2"""

    # CHECK:  "1\222"
    var a5 = '''1"\
2'''

    # CHECK:   "1\22{{(\\0D)?}}\0A2"
    var a6 = '''1"
2'''

    # CHECK:   "1\22\0A2"
    var a7 = '1"\n2'

    # CHECK: "hello concat world"
    var a8 = "hello " "concat " "world"

    var a9 = "Hello"            # CHECK: "Hello"
    var a10 = "Hello 'world'"    # CHECK: "Hello 'world'"
    var a11 = "A\x42"            # CHECK: "AB"
    var a12 = "A\x423"           # CHECK: "AB3"
    var a13 = "A\102"            # CHECK: "AB"
    var a14 = "A\1023"           # CHECK: "AB3"

    # COM: the MLIR textual representation escapes strings, so below \ is \\ and " is \"
    var a15 = 'Hello "world"'    # CHECK: "Hello \22world\22"
    var a16 = r"A\x42"           # CHECK: "A\\x42"
    var a17 = R"A\x42"           # CHECK: "A\\x42"
    var a18 = r"AB\\"            # CHECK: "AB\\\\"
    var a19 = r"A\x"             # CHECK: "A\\x"
    var a20 = "AB\\"             # CHECK: "AB\\"
    var a21 = r"A\"B"            # CHECK: "A\\\22B"
    var a22 = r'A\'B'            # CHECK: "A\\'B"
    var a23 = "A\"B"             # CHECK: "A\22B"
    var a24 = 'A\'B'             # CHECK: "A'B"
    var a25 = r"A\zB"            # CHECK: "A\\zB"

    # Issue #201: https://github.com/modular/mojo/issues/201
    # CHECK: lit.fn *"hello{{.*}} {
    fn hello() -> StaticString:
        # CHECK: kgen.param.constant: {{.*}}@StringLiteral<:string "123"> = <*?>
        return "123"
        # lit.end_fn
    # expected-warning @+1 {{'StringLiteral["other comment"]' value is unused}}
    """other comment"""


##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

# This is an array that has elements of MemoryOnlyInt.
struct MemoryOnlyIntArray:
  fn __getitem__(mut self, x: Int) -> MemoryOnlyInt: pass
  fn __setitem__(mut self, x: Int, var value: MemoryOnlyInt): pass

# CHECK-LABEL: lit.fn @"testMemoryOnlyIntArray
fn testMemoryOnlyIntArray(mut arr: MemoryOnlyIntArray, x: Int, var moi: MemoryOnlyInt):
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %moi)
  arr[x] = moi^
  # CHECK: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, %__call_result_tmp__
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, %__call_result_tmp__
  arr[x] = arr[x]

  # CHECK: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK-SAME: : !lit.ref<!MemoryOnlyInt, mut *"__call_result_tmp__`
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[ANON]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[ANON]][x]
  # CHECK: %[[C1:.*]] = {{.*}}constant: !Int = <{1}>
  # CHECK: lit.ref.store %[[C1:.*]], [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x].x = 1

  # Initialize in memory through a temp + setitem.
  # CHECK: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
  # CHECK: lit.call @{{.*}}__init__{{.*}}({{.*}}, [[ANON]])
  # CHECK: lit.call {{.*}}"__setitem__{{.*}}(%arr, %x, [[ANON]])
  arr[x] = MemoryOnlyInt(42)

  noop() # CHECK: lit.call {{.*}}noop{{.*}}

  # This is yuck, we're rematerializing the base for the rewrite back multiple
  # times: see the "Generalizing Mojo Writeback to Refs" doc in notion.

  # CHECK: [[STORETMP:%.*]] = lit.var.decl "__call_result_tmp__" {{.*}} : !lit.ref<!MemoryOnlyInt,
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[STORETMP]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[STORETMP]][x]
  # CHECK: lit.ref.load [[XP]]
  # CHECK: lit.call {{.*}}Int::@"__iadd__
  # CHECK: [[STORETMP:%.*]] = lit.var.decl "__call_result_tmp__" {{.*}} : !lit.ref<!MemoryOnlyInt,
  # CHECK: lit.call {{.*}}__getitem__{{.*}}(%arr, %x, [[STORETMP]])
  # CHECK: [[XP:%.*]] = lit.ref.struct.ger [[STORETMP]][x]
  # CHECK: lit.ref.store {{.*}}, [[XP]]
  # CHECK: lit.call {{.*}}__setitem__{{.*}}(%arr, %x, [[STORETMP]])
  arr[x].x += 1

# CHECK-LABEL: lit.struct.decl @MyInlineIntInit
struct MyInlineIntInit:
    var value: MemoryOnlyInt
    # CHECK-LABEL: lit.fn @"__init__(expressions::MemoryOnlyInt)"
    # CHECK-SAME: (%value: !lit.ref<!MemoryOnlyInt, imm {{.*}}> read_mem, ?, %self: !lit.ref<!MyInlineIntInit, mut {{.*}}> byref_result) -> !kgen.none
    @implicit
    fn __init__(out self, value: MemoryOnlyInt):
        # CHECK: %0 = lit.ref.struct.ger %self[value]
        # CHECK: lit.call {{.*}}__copyinit__{{.*}}(%value, %0)
        self.value = value

@register_passable
struct ConstDynamicObject:
    fn __init__(out self):
        return

    fn __getattr__(self, name: StringLiteral) -> Int:
        return 0

struct DynamicObject:
    fn __init__(out self):
        pass

    fn __getattr__(self, name: StringLiteral) -> Int:
        return 0

    fn __setattr__(self, name: StringLiteral, value: Int):
        pass


# CHECK-LABEL: lit.fn @"dynamic_attribute()"
fn dynamic_attribute():
    # CHECK: %const_obj = lit.var.decl "const_obj"
    var const_obj = ConstDynamicObject()
    # CHECK: call {{.*}}@ConstDynamicObject::@"__getattr__{{.*}}<:string "dynamic_attribute">(
    _ = const_obj.dynamic_attribute

    var obj = DynamicObject()
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: call {{.*}}@DynamicObject::@"__getattr__{{.*}}<:string "some_attr">([[IMMREF]],
    var a = obj.some_attr

    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %obj
    # CHECK: %[[VALUE:.*]] = kgen.param.constant: !Int = <{42}>
    # CHECK: call {{.*}}@DynamicObject::@"__setattr__{{.*}}<:string "some_attr">([[IMMREF]], {{.*}}, %[[VALUE]])
    obj.some_attr = 42


struct CallableStruct:
    var value: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value

    fn __call__(self, rhs: Int) -> Int:
        return self.value + rhs

# CHECK-LABEL: lit.fn @"test_call_method()"
fn test_call_method():
    # CHECK: %[[C2:.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: lit.call {{.*}}@"__call__{{.*}}(%{{.*}}, %[[C2]])
    var value = CallableStruct(5)
    _ = value(2)

struct MemoryType:
  fn __copyinit__(out self, other: Self):
    pass

@register_passable
struct RegType: pass

# CHECK-LABEL: lit.struct.decl @ParamType
# CHECK-SAME: <a: !Int>
@register_passable("trivial")
struct ParamType[a: Int]: pass

# CHECK-LABEL: lit.fn @"function_types
fn function_types[
  # CHECK-SAME: p0: {{.*}}<<"a": !Int>(!lit.struct<#ParamType <:!Int *(0,0)>{{.*}}>, |) -> !kgen.none
  p0: fn[a: Int](ParamType[a]) -> None,

  # CHECK-SAME: p1: {{.*}}<<"a": !Int, "b": {{.*}}@ParamType<:!Int *(0,0)>>[2](?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> i1
  p1: def[a: Int, b: ParamType[a]]() -> None,

  # CHECK-SAME: p2: {{.*}}"Ts": variadic<!AnyType> pos_vararg>{{.*}}(!lit.ref<{{.*}}@VariadicPack<:!Bool {:i1 0}, {{.*}}origin<0> = *[0,0]}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> *(0,0)>, imm *[0,1]> read_mem|pack_vararg, ?, "__result__": !lit.ref<none, mut *[0,2]> byref_result) async
  p2: async fn[*Ts: AnyType](* *Ts) -> None,
](
  # CHECK-SAME: %{{.*}}: {{.*}}(!Int, |) -> !Int
  float0: fn(Int) -> Int,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!MemoryType, imm {{.*}}> read_mem, |, ?, "__result__": !lit.ref<!MemoryType, mut {{.*}}> byref_result) -> !kgen.none
  float1: fn(MemoryType) -> MemoryType,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!RegType, mut *[0,0]> owned_in_mem, |) -> !RegType
  float2: fn(var RegType) -> RegType,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!MemoryType, mut *[0,0]> owned_in_mem, |) -> !kgen.none
  float3: fn(var MemoryType) -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!lit.ref<!Int, mut *[0,0]> mut, |) -> !kgen.none
  float4: fn(mut Int) -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!Int, |, ?, "__error__": !lit.ref<!Error, mut *[0,0]> byref_error, "__result__": !lit.ref<none, mut *[0,1]> byref_result) throws -> i1
  float5: fn(Int) raises -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!Int, |, ?, "__result__": !lit.ref<none, mut *[0,0]> byref_result) async|capturing -> !kgen.none
  float6: async fn(Int) capturing -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}(!kgen.variadic<!Int> pos_vararg, ?, {{.*}}) throws -> i1
  float7: def(*Int) -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}<(!Int = {10}, {{.*}}StringLiteral <:string "foo">
  # CHECK-SAME: , |) -> !kgen.none>
  float12: fn(Int = 10, StaticString = "foo") -> None,

  # CHECK-SAME: %{{.*}}: {{.*}}<[1]("x": !lit.ref<!MemoryType, imm {{.*}}> read_mem) -> !Int>
  named: fn(x: MemoryType) -> Int
): pass

# CHECK-LABEL: lit.struct.decl @Mem
# CHECK:         lit.alias.decl *"x{{.*}}": type = <i8>
# CHECK-NEXT:    lit.alias.decl *"B{{.*}}": type = <!lit.generator<("foo": i8) -> !kgen.none>>
struct Mem:
   alias x = __mlir_type.i8
   alias B = fn (foo: Self.x) -> None

alias fn_type_alias = fn() -> None

@always_inline
fn func_with_decorator(): pass


struct TwoParamsStruct[a: Int, b: Int](ImplicitlyCopyable):
    pass

# CHECK-LABEL: lit.fn @"variadic_subscript{{.*}}"<idx: !Int, a: variadic<!Int> pos_vararg>
fn variadic_subscript[idx: Int, *a: Int](*b: Int):
    # CHECK-NEXT: %b_0 = lit.var.decl "b"
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}VariadicList{{.*}}__init__{{.*}}(%b)
    # CHECK-NEXT: lit.ref.store [[TMP]], %b_0
    # CHECK: lit.alias.decl *"v0{{.*}}": {{.*}}Int = <variadic_get(:variadic<!Int> a, 2)>
    alias v0 = a[2]

    # CHECK: %v1 = lit.var.decl "v1"
    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <variadic_get(:variadic<!Int> a, 3)>
    # CHECK: lit.ref.store [[TMP]], %v1
    var v1 = a[3]
    # CHECK: %[[LIST:.*]] = lit.ref.load %b_0
    # CHECK: lit.call {{.*}}__getitem__{{.*}}(%[[LIST]],
    var v2 = b[idx]


# CHECK-LABEL: lit.fn @"variadic_memory_subscript
# CHECK-SAME: variadic<!lit.ref<{{.*}}TwoParamsStruct<
# CHECK-SAME:   :!Int variadic_get({{.*}}a, 0)
# CHECK-SAME:   :!Int variadic_get({{.*}}a, 1)
fn variadic_memory_subscript[*a: Int](*b: TwoParamsStruct[a[0], a[1]]):
    # CHECK: %b_0 = lit.var.decl
    # CHECK: [[IMMREF:%.*]] = lit.ref.immut %b_0 :
    # CHECK: [[B1REF:%.*]] = {{.*}}__getitem__{{.*}}([[IMMREF]],
    # CHECK: %v0 = lit.var.decl
    # CHECK: lit.call {{.*}}__copyinit__{{.*}}([[B1REF]], %v0)
    var v0 = b[1]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %b_0 :
    # CHECK: [[B2REF:%.*]] = {{.*}}__getitem__{{.*}}([[IMMREF]],
    # CHECK: %v1 = lit.var.decl
    # CHECK: lit.call {{.*}}__copyinit__{{.*}}([[B2REF]], %v1)
    var v1 = b[2]

fn testTransferWarning():
  var a = MemoryOnlyInt()

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(a^^)

  # expected-warning @+1 {{transfer from an owned value has no effect}}
  consume(MemoryOnlyInt()^)


##===----------------------------------------------------------------------===##
# Test nonmaterializable IntLiteral beyond Int bounds.
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"bigggNumber{{.*}}@IntLiteral<:!pop.int_literal 115792089237316195423570985008687907853269984665640564039457584007913129639936> = <*?>
alias bigggNumber = 2 << 255
fn useBigNumber() -> Int:
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <{512}>
  var notSoBig = bigggNumber // (2 << 246)
  # Easy min-Index
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <{-9223372036854775808}>
  var minInt = -(2<<62)
  return notSoBig


@register_passable("trivial")
struct IndexList[size: Int]:
    @implicit
    fn __init__(out self, *elements: Int):
        pass

    fn __setitem__(mut self, val: Int):
        pass

# Issue 23233 https://github.com/modularml/modular/issues/23233
fn setitemParamToDLValue():
  alias x = 3
  var coords = IndexList[3](0)
  # The main check is just that it's not erroring.
  # CHECK: [[VAR:%.*]] = kgen.param.constant: !Int = <{-3}>
  # CHECK: lit.call {{.*}}IndexList{{.*}}__setitem__{{.*}}[[VAR]]
  coords[1] = -x

# https://github.com/modular/mojo/issues/734
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
# Parameter inference
##===----------------------------------------------------------------------===##

# Test that parameter inference can handle this.
fn dependent_call_it[dtype: DType](ptr: UnsafePointer[Scalar[dtype]]):
   dependent_callee(ptr, 0.0)
# This requires substitution to realize that storage.type.type == DType
fn dependent_callee[dtype: DType](storage: UnsafePointer[Scalar[dtype]],
                   pad_value: Scalar[storage.type.dtype]):
   pass

# This requires handling of VariadicAttr in parameter inference.
fn variadic_attr_caller(*inputs: Tuple[Int]):
   variadic_attr_callee[Int](inputs)
fn variadic_attr_callee[key_type: ImplicitlyCopyable & Movable](
       inputs: VariadicListMem[Tuple[key_type], _]
    ):
  pass

# Test that parameter inference works with implicit conversions - in this case
# that we can infer the parameters of 'thing_taking_reference' even though x
# needs to be built as a Pointer.
fn thing_taking_ref[
  type: AnyType,
  //,
  origin: Origin[_]
](ref [origin] arg: type): pass

fn thing_taking_ref2[type: AnyType](ref arg: type): pass

fn thing_taking_pointer2[type: AnyType](arg: Pointer[type, _]): pass

# CHECK-LABEL: lit.fn @"test_thing_taking_reference
fn test_thing_taking_reference(mut x: String):
  # CHECK-NEXT: lit.call {{.*}}thing_taking_ref{{.*}}(%x)
  thing_taking_ref(x)
  # CHECK-NEXT: lit.call {{.*}}thing_taking_ref2{{.*}}(%x)
  thing_taking_ref2(x)
  # CHECK-NEXT: lit.call {{.*}}@Pointer::@"__init__($1%)"{{.*}}
  thing_taking_pointer2(Pointer(to=x))

struct StructWithStaticMethods:
   @staticmethod
   fn _init_op_state(state: Pointer[Int, _], foo: Int): pass
   fn thing(self):
     var x = 42
     Self._init_op_state(Pointer(to=x), x)

fn infer_through_alias():
  alias MyType = MemoryOnlyInt
  _ = MyType(4)


# CHECK-LABEL: lit.fn @"infer_address_space
fn infer_address_space[
    mut: __mlir_type.i1,
    origin: Origin[mut]._mlir_type
](a: Pointer[Int, origin, AddressSpace(4)]._mlir_type):
  # Show that we can infer the address space parameter of Pointer from a
  # !lit.ref.

  # CHECK: lit.call {{.*}}@Pointer::@"__init__($1%)"{{.*}}(%a)
  var x = Pointer(to=__get_litref_as_mvalue(a))


# https://linear.app/modularml/issue/MOCO-584/[references]-we-cannot-bind-litref-in-parameter-context
# [References] We cannot bind !lit.ref in parameter context
struct ThingWithMethodReferenceSelf:
    fn method(ref a: Self):
      pass

# CHECK-LABEL: lit.fn @"testThingWithMethodReferenceSelf
fn testThingWithMethodReferenceSelf[a: ThingWithMethodReferenceSelf]():
    # CHECK-NEXT: lit.alias.decl *"sizzle`": none =
    # CHECK-SAME: <apply(:!lit.generator<("a": !lit.ref<!ThingWithMethodReferenceSelf,
    # CHECK-SAME:     <:i1 0, :origin<0> #lit.any.origin>,
    # CHECK-SAME:     rebind(:!lit.ref<!ThingWithMethodReferenceSelf, imm #lit.comptime.origin> store_to_mem(a)))>
    alias sizzle = a.method()
