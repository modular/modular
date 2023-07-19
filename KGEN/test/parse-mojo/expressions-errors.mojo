# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s

from DType import DType
from SIMD import Float32, SIMD
from Object import object
from String import String

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
    # expected-error @below {{fn(Int) -> Int}}
    alias float0: fn(Int) -> Int = test_func_type
    # expected-error @below {{async fn() -> None}}
    alias float1: async fn() -> None = test_func_type
    # expected-error @below {{fn[Int]() -> MemType}}
    alias float2: fn[a: Int]() -> MemType = test_func_type
    # expected-error @below {{fn[Int](owned Int) -> MemType}}
    alias float3: fn[a: Int](owned Int) -> MemType = test_func_type
    # expected-error @below {{fn[Int](inout *Int) -> None}}
    alias float4: fn[a: Int](inout *Int) -> None = test_func_type
    # expected-error @below {{fn(*MemType) raises capturing -> None}}
    alias float5: def(*MemType) capturing -> None = test_func_type
    # expected-error @below {{fn[*AnyType](owned * *$0) capt}}
    alias float6: fn[*Ts: AnyType](owned* *Ts) capturing -> None = test_func_type
    # expected-error @below {{fn[AnyType](inout *$0) capturing -> None}}
    alias float7: fn[T: AnyType](inout *T) capturing -> None = test_func_type

    alias type = DType.float32
    # expected-error @below {{SIMD[DType(type.value), 32]}}
    alias value: SIMD[type.value, 32] = SIMD[DType.float32, 32]()

fn hello_str() -> String:
    return "hello"

# expected-error @below {{TODO: memory-primary results are not supported in parameter expressions.}}
alias x = hello_str()

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

struct MemoryPrimaryPair:
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
fn generic_on_type_bad[T: __mlir_type.`!kgen.mlirtype`](a: T): pass

fn generic_on_type_ok[T: __mlir_type.`!kgen.mlirtype`](): pass

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

  let nc3 = nc1 # expected-error {{value of type 'NonCopyable' cannot be copied into its destination}}
  let nc4 = nc2 # expected-error {{value of type 'NonCopyable' cannot be copied into its destination}}

  let mpPair = MemoryPrimaryPair()

  # expected-error @+1 {{invalid call to 'generic_on_type_bad': argument #0 cannot bind generic !mlirtype to memory-only type 'MemoryPrimaryPair'}}
  generic_on_type_bad[MemoryPrimaryPair](mpPair)

  # This should be allowed.
  generic_on_type_ok[MemoryPrimaryPair]()

# expected-note @+1 {{function declared here}}
fn badRef(inout val: Int):
  var x = Float32(1.0)
  # expected-error-re @+1 {{invalid call to 'badRef': l-value of type 'SIMD[{{.*}}f32{{.*}}]' cannot be converted to reference of type 'Int'}}
  badRef(x)

fn unused_values():
  var x : Int = 42

  _ = 4+4 # OK: Explicitly ignored.
  # expected-warning @+1 {{'Int' value is unused}}
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


def no_unused_values_in_def():
  var x : Int = 42
  4+4  # MValue
  x  # LValue
  x+1 # DRValue
  testLValuesRvalues

  _ # expected-error {{discard pattern requires an initializing expression}}

##===----------------------------------------------------------------------===##
# Keyword arguments
##===----------------------------------------------------------------------===##

fn takeKeywordArgs(i: Int, j: Int): pass

def testKWargs():
  # expected-error @+1 {{keyword arguments are not supported yet}}
  takeKeywordArgs(j = 42, i = 1)
  # expected-error @+1 {{positional argument follows keyword argument}}
  takeKeywordArgs(j = 42, 1)

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

# expected-error @+1 {{'Self' type may only be used inside a type}}
fn badSelf(a: Self):
  # expected-error @+1 {{'Self' type may only be used inside a type}}
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

fn unused_assignments():
  var a = 1
  a = a  # ok of course.
  a := a # expected-warning {{'Int' value is unused}}

async fn async_function() -> Int:
    return 0

fn call_async_fn_in_param():
    # expected-error @below {{cannot call async function in alias initializer}}
    alias result = async_function()
    alias awaitable: Coroutine[Int]
    # expected-error @below {{cannot await in alias initializer}}
    alias await_it = await awaitable
    # expected-error @below {{cannot await inside a non-async function}}
    await async_function()

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
    # expected-error @+1 {{'Int' value has no attribute '__adaptive_set'}}
    alias bad_int= (5).__adaptive_set

struct GetSettable:
  fn __getitem__(self, x: Int) -> Int: pass
  fn __setitem__(self, x: Int, y: Int): pass


fn lvalue_utilities(a: __mlir_type.index, inout b: GetSettable):
  # expected-error @+1 {{expression must be mutable}}
  let addr : __mlir_type.`!pop.pointer<index>` = __get_lvalue_as_address(a)

  # expected-error @+1 {{cannot use a dynamic LValue in this operator}}
  _ = __get_lvalue_as_address(b[1])

  # Get and use an lvalue from an address.
  __get_address_as_lvalue(addr) = 42

  let addr2 : __mlir_type.index
  # expected-error @+1 {{operand must have '!pop.pointer<T>' type, not 'index'}}
  __get_address_as_lvalue(addr2) = 42


fn test_int_to_int_error(a: Int):
  # expected-error @+1 {{cannot construct 'Int' with itself, you can remove the constructor call}}
  _ = Int(a)

  # expected-error @+1 {{cannot construct 'GetAttrNotString' from 'Int' value in assignment}}
  _ = GetAttrNotString(a)
