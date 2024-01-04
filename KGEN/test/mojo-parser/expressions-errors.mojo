# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Conversions
##===----------------------------------------------------------------------===##

def invalid_conversion(a: Int, b: __mlir_type.index):
  b = a # expected-error {{cannot implicitly convert 'Int' value to 'index' in assignment}}

  # expected-error @+1 {{cannot use initializer syntax on MLIR type 'index'}}
  _ = __mlir_type.index(4)

struct NotBoolConvertible:
  pass

# Issue #6600
fn negBuiltinType(x: __mlir_type.f64) :
    # expected-error @+1 {{'f64' does not implement the '__neg__' method}}
    _ = -x

# expected-note @+1 {{function declared here}}
fn some_fn_take_int(a: Int): pass
fn some_fn_ret_int() -> Int: return 42

# Issue #11288
fn test_overload_set():
  # expected-error @+1 {{invalid call to 'some_fn_take_int': argument #0 cannot be converted from 'fn() -> Int' to 'Int'}}
  some_fn_take_int(some_fn_ret_int)


struct MemType: pass


fn test_func_type():
    # expected-error @below {{fn(Int, /) -> Int}}
    alias float0: fn(Int) -> Int = test_func_type
    # expected-error @below {{async fn() -> None}}
    alias float1: async fn() -> None = test_func_type
    # expected-error @below {{fn[Int]() -> MemType}}
    alias float2: fn[a: Int]() -> MemType = test_func_type
    # expected-error @below {{fn[Int](owned Int, /) -> MemType}}
    alias float3: fn[a: Int](owned Int) -> MemType = test_func_type
    # expected-error @below {{fn[Int](inout *Int) -> None}}
    alias float4: fn[a: Int](inout *Int) -> None = test_func_type
    # expected-error @below {{fn(owned *MemType) raises capturing -> None}}
    alias float5: def(*MemType) capturing -> None = test_func_type
    # expected-error @below {{fn[*AnyRegType](owned * *$0) capt}}
    alias float6: fn[*Ts: AnyRegType](owned* *Ts) capturing -> None = test_func_type
    # expected-error @below {{fn[AnyRegType](inout *$0) capturing -> None}}
    alias float7: fn[T: AnyRegType](inout *T) capturing -> None = test_func_type

    # expected-error @below {{unnamed argument cannot follow named argument}}
    alias f1: fn (a: Int, StringLiteral) -> Int
    # expected-error @below {{unnamed argument cannot follow '/' or '*'}}
    alias f2: fn (Int, /, StringLiteral) -> Int
    # expected-error @below {{unnamed argument cannot follow '/' or '*'}}
    alias f3: fn (*, StringLiteral) -> Int
    # expected-error @below {{unnamed argument must be positional-only}}
    alias f4 = fn (Int, b: Int) capturing -> Int

    # expected-error @below {{unnamed parameter cannot follow named parameter}}
    alias f5: fn [a: Int, StringLiteral]() -> Int
    # expected-error @below {{unnamed parameter cannot follow '/' or '*'}}
    alias f6: fn [Int, /, StringLiteral] -> Int
    # expected-error @below {{unnamed parameter cannot follow '/' or '*'}}
    alias f7: fn [*, StringLiteral] -> Int = test_func_type
    # expected-error @below {{unnamed parameter must be positional-only}}
    alias f8 = fn [Int, b: Int] capturing -> Int

    alias type = DType.float32
    # expected-error @below {{SIMD[DType(type.value), 32]}}
    alias value: SIMD[type.value, 32] = SIMD[DType.float32, 32]()

##===----------------------------------------------------------------------===##
# LValue and RValues
##===----------------------------------------------------------------------===##

fn assignRValue():
  42 = 17 # expected-error {{expression must be mutable in assignment}}

struct LValuesRvalues:
  fn __init__(inout self): pass
  fn __copyinit__(inout self, existing: Self): pass

  def normalMethod(self): pass
  # expected-note @+1 {{function declared here}}
  def mutatingMethod(inout self) -> None: pass
  # expected-note @+1 {{function declared here}}
  def takesByRef(self, inout x: LValuesRvalues): pass

  def normalMethod3(self, a: Float32): pass

struct MemoryOnlyPair:
  var x: Int
  var y: Int
  fn __init__(inout self):
    self.x = 0
    self.y = 0
  fn __copyinit__(inout self, existing: Self):
    self.x = existing.x
    self.y = existing.y

struct NonCopyable:
  fn __init__(inout self): pass

# expected-note @+1 {{function declared here}}
fn generic_on_type_bad[T: AnyRegType](a: T): pass

# expected-note @+1 {{function declared here}}
fn generic_on_type_bad_raises[T: AnyRegType]() raises -> T: pass

fn generic_on_type_ok[T: AnyRegType](): pass

def testLValuesRvalues() -> None:
  # Test with lvalues
  var lv: LValuesRvalues
  lv.normalMethod()
  lv.mutatingMethod()

  # Partial application.
  # expected-error @+1 {{TODO: partial application to mutable base isn't supportable without a lifetime model}}
  lv.mutatingMethod

  # Test with rvalues
  LValuesRvalues().normalMethod()
  LValuesRvalues().mutatingMethod()  # expected-error {{invalid use of mutating method on rvalue of type 'LValuesRvalues'}}

  # expected-error @+1 {{method argument #0 must be mutable in order to pass as a by-ref argument}}
  LValuesRvalues().takesByRef(LValuesRvalues())

  # We can not implicitly declare things on the RHS
  lv += unknown2 # expected-error {{use of unknown declaration 'unknown2'}}

  lv.normalMethod3(1.0)

  var nc1 = NonCopyable()
  let nc2 = NonCopyable()

  let nc3 = nc1 # expected-error {{'NonCopyable' is not copyable because it has no '__copyinit__'}}
  let nc4 = nc2 # expected-error {{'NonCopyable' is not copyable because it has no '__copyinit__'}}

  let mpPair = MemoryOnlyPair()

  # expected-error @+1 {{invalid call to 'generic_on_type_bad': argument #0 cannot bind AnyRegType type to memory-only type 'MemoryOnlyPair'}}
  generic_on_type_bad[MemoryOnlyPair](mpPair)

  # For issue https://github.com/modularml/mojo/issues/910
  # expected-error @+1 {{invalid call to 'generic_on_type_bad_raises': result cannot bind AnyRegType type to memory-only type 'Variant[Error, MemoryOnlyPair]'}}
  generic_on_type_bad_raises[MemoryOnlyPair]()

  # This should be allowed.
  generic_on_type_ok[MemoryOnlyPair]()

# expected-note @+1 {{function declared here}}
fn badRef(inout val: Int):
  var x = Float32(1.0)
  # expected-error-re @+1 {{invalid call to 'badRef': l-value of type 'SIMD[{{.*}}f32{{.*}}]' cannot be converted to reference of type 'Int'}}
  badRef(x)

fn unused_values():
  var x : Int = 42

  _ = 4+4 # OK: Explicitly ignored.
  # expected-warning @+1 {{'IntLiteral' value is unused}}
  4+4  # MValue

  _ = x # OK: Explicitly ignored.
  # expected-warning @+1 {{'Int' value is unused}}
  x  # LValue

  _ = x+1 # OK: Explicitly ignored.
  # expected-warning @+1 {{'Int' value is unused}}
  x+1 # DRValue

  # expected-warning @+1 {{function pointer was formed but not called, did you forget '()'s?}}
  testLValuesRvalues
  _ = testLValuesRvalues # OK


fn func_with_static_param[x: Int]() -> Int:
  return x

fn dynamic_used_as_param() -> Int:
  let x = 5
  # expected-error @+1 {{cannot use a dynamic value in call parameter}}
  return func_with_static_param[x]()

@value
struct StructWithField:
  var x : Int

fn dynamic_used_as_param_2() -> Int:
  var w = StructWithField(3)
  # expected-error @+1 {{cannot use a dynamic value in call parameter}}
  return func_with_static_param[w.x]()

fn higher_order_int_func[func: fn (Int) capturing -> Int]() -> Int:
  return func(3)

fn use_non_parameter_func() -> Int:
  let val = 8
  fn my_nested_func(x: Int) -> Int:
    return val + x
  # expected-error @+1 {{cannot use a dynamic value in call parameter}}
  print(higher_order_int_func[my_nested_func]())

##===----------------------------------------------------------------------===##
# Keyword arguments
##===----------------------------------------------------------------------===##

# expected-note @+1 {{function declared here}}
fn var_func(s: StringLiteral, *args: Int): pass

# expected-note @+1 {{function declared here}}
fn pack_func[*Ts: AnyRegType](*args: *Ts): pass

# expected-note @+1 {{function declared here}}
fn take_kw_args(i: Int, j: Int = 7): pass

fn test_kw_args():
  # expected-error @+2 {{duplicate keyword argument 'j'}}
  # expected-note @+1 {{previously specified here}}
  take_kw_args(j = 42, j = 43)
  # expected-error @+1 {{positional argument follows keyword argument}}
  take_kw_args(j = 42, 1)

fn test_kw_args_2():
  # expected-error @+1 {{unexpected keyword argument: 'args'}}
  var_func("boo", args=3)
  # expected-error @+1 {{unexpected keyword argument: 'args'}}
  pack_func("boo", args=2)
  # expected-error @+1 {{unexpected keyword argument: 'z'}}
  take_kw_args(8, z=13)
  # expected-error @+1 {{unexpected keyword arguments: 'x', 'z'}}
  take_kw_args(8, x=11, z=13)
  # expected-error @+1 {{argument #0 ('i') passed both as positional and keyword operand}}
  take_kw_args(8, i=11)

##===----------------------------------------------------------------------===##
# Tuples
##===----------------------------------------------------------------------===##

fn bad_tuple(a: Int):
  _ = (a, a, b)  # expected-error {{use of unknown declaration 'b'}}

  var c: Int
  var d: Int
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, Int, Int]' value to 'Tuple[Int, Int]' in assignment}}
  (c, d) = (a, a, a)
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int]' value to 'Tuple[Int, Int]' in assignment}}
  (c, d) = (a,)
  # expected-error @+1 {{cannot implicitly convert 'Int' value to 'Tuple[Int, Int]' in assignment}}
  (c, d) = a

  var iTup : Tuple[Int, Int]
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, FloatLiteral]' value to 'Tuple[Int, Int]' in assignment}}
  iTup = (1, 2.0)


def tuple_return():
  return 32, 17 # expected-error {{cannot implicitly convert 'Tuple[Int, Int]' value to 'object' in return value}}


##===----------------------------------------------------------------------===##
# Other Specific expression forms
##===----------------------------------------------------------------------===##

@register_passable
struct WeirdBoolish:
  fn __bool__(self) -> Bool: return False
  fn __copyinit__(self) -> Self: pass;

fn badParamAnd[a: Bool, b: WeirdBoolish]():
  #expected-error @+1 {{value of type 'Bool' is not compatible with value of type 'WeirdBoolish'}}
  alias c = a and b

# expected-error @+1 {{'Self' type may only be used inside a struct or trait}}
fn badSelf(a: Self):
  # expected-error @+1 {{'Self' type may only be used inside a struct or trait}}
  var x: Self.field

# Structs convertible to each other.
struct Conv1:
  fn __init__(inout self, value: Conv2): pass
struct Conv2:
  fn __init__(inout self, value: Conv1): pass

@register_passable
struct MyIntPair:
  var a: Int
  var b: Int

fn dict_expression(a: Int):
  # expected-error @+1 {{TODO: cannot emit dictionary literals yet}}
  _ = {}
  # expected-error @+1 {{TODO: cannot emit dictionary literals yet}}
  _ = {a: 4}
  # expected-error @+1 {{TODO: cannot emit dictionary literals yet}}
  let dict = {a: 4, "b": 17}
  # expected-error @+1 {{TODO: cannot emit dictionary literals yet}}
  _ = {a: 4, **dict, "b": 17}

  # expected-error @+1 {{TODO: dictionary comprehension parsing}}
  let comprehension = {key:value for (key,value) in dict.items()}

  # Dictionary subscripts.

  # expected-error @+1 {{cannot use a dynamic value in type}}
  _ = a{1: 2, **dict}


  # expected-error @+1 {{MLIR types may not be initialized with this syntax}}
  _ = __mlir_type.index{value: 4}

  # expected-error @+1 {{type initializer requires keys to be bare field names}}
  _ = MyIntPair{"a": 4}

  # expected-error @+1 {{field 'a' specified multiple times}}
  _ = MyIntPair{a: 4, a: 4}
  # expected-error @+1 {{cannot expand into initializer list}}
  _ = MyIntPair{a: a, **a}
  # expected-error @+1 {{no value for field 'b' specified}}
  _ = MyIntPair{a: 4}
  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int' in field initializer}}
  _ = MyIntPair{a: 4.0, b: 4}
  _ = MyIntPair{a: 4, b: 4}

fn dict_parse_errors(a: Int):
  # expected-error @+1 {{dictionary comprehension must start with single key:value pair}}
  _ = {key:value, key:value for (key,value) in dict.items()}



fn bad_exprs(cond: Bool, Float32: Float32, c1: Conv1, c2: Conv2):
  # expected-error-re @+1 {{value of type 'SIMD[{{.*}}f32{{.*}}]' is not compatible with value of type 'Conv1'}}
  _ = Float32 if cond else c1

  # expected-error @below {{ambiguous merge: left value has type 'Conv1' and right value has type 'Conv2', and both convert to each other}}
  # expected-note @below {{you could disambiguate by casting the left value to 'Conv2'}}
  # expected-note @below {{or cast the right value to 'Conv1'}}
  _ = c1 if cond else c2

def bad_assignment0(a: Int, b: Int):
   # expected-error @+1 {{cannot implicitly convert 'None' value to 'Int' in assignment}}
   a = b += b

def bad_assignment1(a: Int, b: Int):
   # expected-error @+1 {{expected ')' in parenthesized expression}}
   a = (b += b)

fn bad_walrus_implicit_decl_in_fn():
  # expected-error @+1 {{use of unknown declaration 'a'}}
  if a := 4:
    pass

fn unused_assignments():
  var a = 1
  a = a  # ok of course.
  a := a # expected-warning {{'Int' value is unused}}

async fn async_function() -> Int:
    return 0

# See Issue #15578
def doIs(a: Int, b: Int):
  # expected-error @+1 {{'Int' does not implement the '__is__' method}}
  if a is b:
    pass

def doIsNot(a: Int, b: Int):
  # expected-error @+1 {{'Int' does not implement the '__isnot__' method}}
  if a is not b:
    pass

def hasResultParam[() -> a: Int]():
  pass

def aliasCallNoBind():
  alias callee = hasResultParam
  # expected-error @below {{invalid indirect call: callee has 1 unbound result parameter}}
  callee()

##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

struct WeirdArray:
  # expected-note @+1 {{function declared here}}
  fn __getitem__(self, x: Int) -> Int:
    return 1

struct MultiSetItem:
  # expected-note @+1 {{candidate declared here}}
  fn __setitem__(self, x: Int, y: Int): pass
  # expected-note @+1 {{candidate declared here}}
  fn __setitem__(self, x: Int, y: Float32): pass

struct IncompatElementTypes:
  fn __getitem__(self, x: Int) -> Int: pass
  fn __setitem__(self, x: Int, y: Float32): pass

fn testSubscripts(a: WeirdArray, b: MultiSetItem, c: IncompatElementTypes):
  # expected-error @+1 {{invalid call to '__getitem__': index cannot be converted from 'FloatLiteral' to 'Int'}}
  _ = a[1.0]

  # expected-error @+1 {{invalid call to '__getitem__': callee expects 2 arguments, but 3 were specified}}
  _ = a[1, 2]

  # expected-error @+1 {{expression must be mutable in assignment}}
  a[1] = 4

  # expected-error @+1 {{'MultiSetItem' has overloaded __setitem__ implementations, which isn't supported}}
  b[1] = 4

  let tmp : Int = c[1]
  # expected-error-re @+1 {{cannot implicitly convert 'SIMD[f32, 1]' value to 'Int' in assignment}}
  c[1] = Float32(4.0)
  c[1] = tmp

  # expected-error @+1 {{keyword operands for __setitem__ not supported yet}}
  c[x=1] = 4


struct GetAttrNotString:
    fn __init__(inout self):
        pass

    # expected-note @below {{function declared here}}
    fn __getattr__(self, idx: Int) -> Int:
        return 0

fn invalid_getattr():
    let obj = GetAttrNotString()
    # expected-error @below {{invalid call to '__getattr__': attribute name cannot be converted from 'StringLiteral' to 'Int}}
    obj.attr


##===----------------------------------------------------------------------===##
# __adaptive_set errors
##===----------------------------------------------------------------------===##

# expected-note @+1 {{declared here}}
fn bar[x: __mlir_type.index]() -> Int:
        return 1

fn test_adaptive_set():
    # expected-error @+1 {{cannot form a reference to non @adaptive declaration of 'bar'}}
    alias bad = bar.__adaptive_set
    # expected-error @+1 {{'IntLiteral' value has no attribute '__adaptive_set'}}
    alias bad_int= (5).__adaptive_set

struct GetSettable:
  fn __getitem__(self, x: Int) -> Int: pass
  fn __setitem__(self, x: Int, y: Int): pass


fn lvalue_utilities(a: __mlir_type.index, inout b: GetSettable):
  # expected-error @+1 {{expression must be mutable}}
  let addr : __mlir_type.`!kgen.pointer<index>` = __get_lvalue_as_address(a)

  # expected-error @+1 {{cannot use a dynamic LValue}}
  _ = __get_lvalue_as_address(b[1])

  # Get and use an lvalue from an address.
  __get_address_as_lvalue(addr) = 42

  let addr2 : __mlir_type.index
  # expected-error @+1 {{operand must have '!kgen.pointer<T>' type, not 'index'}}
  __get_address_as_lvalue(addr2) = 42

struct NoSelfCtor:
  var x: Int
  fn __init__(inout self: Self, x: Int):
    self.x = x

fn test_int_to_int_error(a: Int, b: NoSelfCtor):
  # expected-error @+1 {{cannot construct 'NoSelfCtor' with itself, you can remove the constructor call}}
  _ = NoSelfCtor(NoSelfCtor(a))

  # expected-error @+1 {{cannot construct 'GetAttrNotString' from 'Int' value in assignment}}
  _ = GetAttrNotString(a)


##===----------------------------------------------------------------------===##
# lambda not supported yet
##===----------------------------------------------------------------------===##

def testLambda():
  # expected-error @+1 {{Mojo doesn't support lambda expressions yet}}
  _ = lambda x, y: x+y

def testLambda2():
  # expected-error @+1 {{Mojo doesn't support lambda expressions yet}}
  _ = lambda (x: Int, y: Float) raises: x+y

def testInExpr(x, y):
  # expected-error @+1 {{'in' operation is not yet supported}}
  _ = x in y
  # expected-error @+1 {{'not in' operation is not yet supported}}
  _ = x not in y



struct CopyAndInitMemType:
  fn __init__(inout self): pass
  fn __copyinit__(inout self, other: Self): pass
  # expected-note @+1 {{function declared here}}
  fn __le__(self, other: Self) -> Self: return self
  fn __mlir_i1__(self) -> __mlir_type.i1: pass

fn getaddr_mem():
  var x = CopyAndInitMemType()
  # https://github.com/modularml/mojo/issues/912
  # expected-error @+1 {{operand must have '!kgen.pointer<T>' type, not 'CopyAndInitMemType'}}
  __get_address_as_lvalue(x)

fn compare_mem_result():
  var x = CopyAndInitMemType()
  # https://github.com/modularml/mojo/issues/1115
  # expected-error @+1 {{chained comparison operator does not currently support memory-only return types}}
  x <= x <= x

# Issue #27654: Parser crash: Assertion failed: Types should match
fn return_metatype_problem() -> CopyAndInitMemType:
  # expected-error @+1 {{cannot implicitly convert 'CopyAndInitMemType' value to 'CopyAndInitMemType' in return value}}
  return CopyAndInitMemType

fn test_bad_ref(a: Int, b: CopyAndInitMemType):
  # expected-error @+1 {{cannot get a reference to a register value}}
  _ = __get_ref_from_value(a)

  let bref = __get_ref_from_value(b) # ok
  # expected-error @+1 {{expression must be mutable in assignment}}
  __get_value_from_ref(bref) = CopyAndInitMemType()

  # expected-error @+1 {{invalid call to '__le__': right side cannot be converted from 'ref[*"`b"] CopyAndInitMemType' to 'CopyAndInitMemType'}}
  _ = b <= bref
