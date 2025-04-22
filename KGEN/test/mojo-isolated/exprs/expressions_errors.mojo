# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics

@register_passable
struct SomeNonTrivRegPassable: pass

fn takes_pos_or_kw_arg(i: Index, j: Index):
    pass


fn test_duplicate_kw_arg(x: Index):
    takes_pos_or_kw_arg(
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword argument 'j'}}
    )


fn test_pos_after_kw_arg(x: Index):
    takes_pos_or_kw_arg(
        j=x,
        x,  # expected-error {{positional argument follows keyword argument}}
    )


fn takes_pos_or_kw_param[i: Index, j: Index]():
    pass


fn test_duplicate_kw_param[x: Index]():
    takes_pos_or_kw_param[
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{duplicate keyword parameter 'j'}}
    ]


fn test_pos_after_kw_param[x: Index]():
    takes_pos_or_kw_param[
        j=x,
        x,  # expected-error {{positional parameter follows keyword parameter}}
    ]()


fn invalid_with():
    # expected-error @below {{use of unknown declaration 'bogus'}}
    with bogus() as foo:
        foo.something()


struct SomeType:
    pass


fn mem_type_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating an `alias` instead of a `var}}
    var type = SomeType


fn reg_type_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating an `alias` instead of a `var}}
    var type = Int


trait SomeTrait:
    pass


fn trait_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating an `alias` instead of a `var}}
    var type = SomeTrait


fn reg_type_func() -> AnyTrivialRegType:
    # expected-error @below {{dynamic type values not permitted yet}}
    return Int


fn mem_type_func() -> AnyType:
    # expected-error @below {{dynamic type values not permitted yet}}
    return SomeType


fn takes_reg_type(t: AnyTrivialRegType):
    pass


fn test_takes_reg_type():
    # expected-error @below {{use of unknown declaration 'takes_type'}}
    takes_type(Int)


fn takes_mem_type(t: AnyType):
    pass


fn test_takes_mem_type():
    # expected-error @below {{use of unknown declaration 'takes_type'}}
    takes_type(SomeType)

# MOCO-56: Mojo produces weird error when mut function is used in non mutating function
struct SomethingWithInferredParam[T: CollectionElement]:
  pass
# expected-note @+1 {{function declared here}}
fn SomethingWithInferredParamCallee(mut v: SomethingWithInferredParam):
  pass

fn SomethingWithInferredParamCaller(v: SomethingWithInferredParam):
  # expected-error @+1 {{must be mutable in order to pass to a mutating argument}}
  SomethingWithInferredParamCallee(v)

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
    # expected-error @below {{fn[Int](mut *Int) -> None}}
    alias float4: fn[a: Int](mut *Int) -> None = test_func_type
    # expected-error @below {{fn(*MemType) raises capturing -> None}}
    alias float5: def(*MemType) capturing -> None = test_func_type
    # expected-error @below {{'fn[*AnyType](owned * *$0) capturing -> None'}}
    alias float6: fn[*Ts: AnyType](owned* *Ts) capturing -> None = test_func_type
    # expected-error @below {{'fn[*AnyType](owned * *$0) capturing -> None'}}
    alias float6a: fn[*Ts: AnyType](owned* *Ts) capturing -> None = test_func_type
    # expected-error @below {{fn[AnyTrivialRegType](mut *$0) capturing -> None}}
    alias float7: fn[T: AnyTrivialRegType](mut *T) capturing -> None = test_func_type

    # expected-error @below {{unnamed argument cannot follow named argument}}
    alias f1: fn (a: Int, Int) -> Int
    # expected-error @below {{unnamed argument cannot follow '/' or '*'}}
    alias f2: fn (Int, /, Int) -> Int
    # expected-error @below {{unnamed argument cannot follow '/' or '*'}}
    alias f3: fn (*, Int) -> Int
    # expected-error @below {{unnamed argument must be positional-only}}
    alias f4 = fn (Int, b: Int) capturing -> Int

    # expected-error @below {{unnamed parameter cannot follow named parameter}}
    alias f5: fn [a: Int, Int]() -> Int
    # expected-error @below {{unnamed parameter cannot follow '/' or '*'}}
    alias f6: fn [Int, /, Int] -> Int
    # expected-error @below {{unnamed parameter cannot follow '/' or '*'}}
    alias f7: fn [*, Int] -> Int = test_func_type
    # expected-error @below {{unnamed parameter must be positional-only}}
    alias f8 = fn [Int, b: Int] capturing -> Int

##===----------------------------------------------------------------------===##
# LValue and RValues
##===----------------------------------------------------------------------===##

fn mutArg(a: Int):
  a = a  # expected-error {{expression must be mutable in assignment}}

fn assignRValue():
  42 = 17 # expected-error {{expression must be mutable in assignment}}

struct LValuesRvalues:
  fn __init__(out self): pass
  fn __copyinit__(out self, existing: Self): pass

  def normalMethod(self): pass
  # expected-note @+1 {{function declared here}}
  def mutatingMethod(mut self) -> None: pass
  # expected-note @+1 {{function declared here}}
  def takesByRef(self, mut x: LValuesRvalues): pass

  def normalMethod3(self, a: FloatDyn): pass

struct MemoryOnlyPair:
  var x: Int
  var y: Int
  fn __init__(out self):
    self.x = 0
    self.y = 0
  fn __copyinit__(out self, existing: Self):
    self.x = existing.x
    self.y = existing.y

struct NonCopyable:
  fn __init__(out self): pass

fn generic_on_type_ok[T: AnyTrivialRegType](): pass

def testLValuesRvalues() -> None:
  # Test with lvalues
  var lv: LValuesRvalues
  lv.normalMethod()
  lv.mutatingMethod()

  # Partial application.
  # expected-error @below {{cannot emit closure for method 'mutatingMethod'}}
  # expected-note @below {{computing member method closure is not yet supported}}
  # expected-note @below {{did you forget '()'s?}}
  lv.mutatingMethod

  # Test with rvalues
  LValuesRvalues().normalMethod()
  LValuesRvalues().mutatingMethod()  # expected-error {{invalid use of mutating method on rvalue of type 'LValuesRvalues'}}

  # expected-error @+1 {{method argument #0 must be mutable in order to pass to a mutating argument}}
  LValuesRvalues().takesByRef(LValuesRvalues())

  # We can not implicitly declare things on the RHS
  lv += unknown2 # expected-error {{use of unknown declaration 'unknown2'}}

  lv.normalMethod3(1.0)

  var nc1 = NonCopyable()
  var nc2 = NonCopyable()

  var nc3 = nc1 # expected-error {{'NonCopyable' is not copyable because it has no '__copyinit__'}}
  var nc4 = nc2 # expected-error {{'NonCopyable' is not copyable because it has no '__copyinit__'}}

  var mpPair = MemoryOnlyPair()

  # expected-error @+1 {{cannot implicitly convert 'AnyStruct[MemoryOnlyPair]' value to 'AnyTrivialRegType' in alias initializer}}
  alias T: AnyTrivialRegType = MemoryOnlyPair

# expected-note @+1 {{function declared here}}
fn badRef(mut val: Int):
  var x = FloatDyn(1.0)
  # expected-error @+1 {{invalid call to 'badRef': l-value of type 'FloatDyn' cannot be converted to reference of type 'Int'}}
  badRef(x)

struct PythonObject: pass
fn getPythonObject() -> PythonObject: pass

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

  # No warning.
  getPythonObject()

  # No warning.
  try:
    unused_values2()
  except e:
    pass

def unused_values2():
  # No warning.
  getPythonObject()

  # expected-warning @+1 {{'Int' value is unused}}
  4+1

def no_unused_values_in_def():
  var x : Int = 42
  4+4  # expected-warning {{'Int' value is unused}}
  x    # expected-warning {{'Int' value is unused}}
  x+1  # expected-warning {{'Int' value is unused}}
  testLValuesRvalues # expected-warning {{function pointer was formed but not called, did you forget '()'s?}}

  _ # expected-error {{cannot read from discard pattern '_'}}

  # expected-error @+1 {{cannot read from discard pattern '_'}}
  var abc = _

  # expected-error @+1 {{cannot read from discard pattern '_'}}
  var bcd = *_

  _ = *x # expected-error {{can't use starred expression here}}

fn func_with_static_param[x: Int]() -> Int:
  return x

fn dynamic_used_as_param() -> Int:
  var x = 5
  # expected-error @+1 {{cannot use a dynamic value in call parameter}}
  return func_with_static_param[x]()

@value
struct StructWithField:
  var x : Int

fn dynamic_used_as_param_2() -> Int:
  var w = StructWithField(3)
  # expected-error @+1 {{cannot use a dynamic value in call parameter}}
  return func_with_static_param[w.x]()

fn higher_order_int_func[func: fn (Int) escaping -> Int]() -> Int:
  return func(3)

fn use_non_parameter_func() -> Int:
  var val = 8
  fn my_nested_func(x: Int) -> Int:
    return val + x
  # expected-error @+1 {{cannot use a dynamic value in call parameter}}
  var result: Int = higher_order_int_func[my_nested_func]()

##===----------------------------------------------------------------------===##
# Tuples
##===----------------------------------------------------------------------===##

fn bad_tuple(a: Int):
  _ = (a, a, b)  # expected-error {{use of unknown declaration 'b'}}

  var c: Int
  var d: Int
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, Int, Int]' value to 'Tuple[Int, Int]'}}
  (c, d) = (a, a, a)
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int]' value to 'Tuple[Int, Int]'}}
  (c, d) = (a,)
  # expected-error @+1 {{cannot implicitly convert 'Int' value to 'Tuple[Int, Int]'}}
  (c, d) = a

  var iTup : Tuple[Int, Int]
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, FloatDyn]' value to 'Tuple[Int, Int]'}}
  iTup = (1, 2.0)


def tuple_return() -> Int:
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, Int]' value to 'Int'}}
  return 32, 17


# Issue https://github.com/modular/mojo/issues/1917
# Do not crash in tuple creation if element has syntax error.
# expected-error @below {{expected '(' for argument list}}
fn bad_func return fn() -> __mlir_type.index


##===----------------------------------------------------------------------===##
# Other Specific expression forms
##===----------------------------------------------------------------------===##

@register_passable
struct WeirdBoolish:
  fn __bool__(self) -> Bool: return False
  fn __copyinit__(out self, existing: Self): pass;

fn badParamAnd[a: Bool, b: WeirdBoolish]():
  #expected-error @+1 {{value of type 'Bool' is not compatible with value of type 'WeirdBoolish'}}
  alias c = a and b

# expected-error @+1 {{'Self' type may only be used inside a struct or trait}}
fn badSelf(a: Self):
  var x: Self.field

# Structs convertible to each other.
struct Conv1:
  @implicit
  fn __init__(out self, value: Conv2): pass
struct Conv2:
  @implicit
  fn __init__(out self, value: Conv1): pass

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
  var dict = {a: 4, "b": 17}
  # expected-error @+1 {{TODO: cannot emit dictionary literals yet}}
  _ = {a: 4, **dict, "b": 17}

  # expected-error @+1 {{TODO: dictionary comprehension parsing}}
  var comprehension = {key:value for (key,value) in dict.items()}

  # Dictionary subscripts.

  # expected-error @+1 {{cannot use a dynamic value in type}}
  _ = a{1: 2, **dict}

  # expected-error @+1 {{TODO: cannot emit dictionary literals yet}}
  _ = MyIntPair{"a": 4}

fn dict_parse_errors(a: Int):
  # expected-error @+1 {{dictionary comprehension must start with single key:value pair}}
  _ = {key:value, key:value for (key,value) in dict.items()}



fn bad_exprs(cond: Bool, x: Error, c1: Conv1, c2: Conv2):
  # expected-error @+1 {{value of type 'Error' is not compatible with value of type 'Conv1'}}
  _ = x if cond else c1

  # expected-error @below {{ambiguous merge: left value has type 'Conv1' and right value has type 'Conv2', and both convert to each other}}
  # expected-note @below {{you could disambiguate by casting the left value to 'Conv2'}}
  # expected-note @below {{or cast the right value to 'Conv1'}}
  _ = c1 if cond else c2

def bad_assignment0(a: Int, b: Int):
   # expected-error @+1 {{cannot implicitly convert 'None' value to 'Int'}}
   a = b += b

def bad_assignment1(a: Int, b: Int):
   # expected-error @+1 {{expected ')' in parenthesized expression}}
   a = (b += b)

fn bad_walrus_implicit_decl_in_fn():
  # Implicit definition in an 'fn' is ok.
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

##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

struct ConvertFromInt:
    fn __init__(out self): pass
    @implicit
    fn __init__(out self, value: Int): pass

struct IncompatElementTypes:
  fn __getitem__(self, x: Int) -> Int: pass
  fn __setitem__(self, x: Int, y: ConvertFromInt): pass

fn test_subscript_implicit_conversion(c: IncompatElementTypes):
  var tmp : Int = c[1]
  # expected-error @+1 {{cannot implicitly convert 'ConvertFromInt' value to 'Int'}}
  c[1] = ConvertFromInt()  # FIXME: This should work
  c[1] = tmp

struct GetAttrNotString:
    # expected-note @below {{function declared here}}
    fn __init__(out self):
        pass

    # expected-note @below {{function declared here}}
    fn __getattr__(self, idx: Int) -> Int:
        return 0

fn invalid_getattr():
    var obj = GetAttrNotString()
    # expected-error @below {{invalid call to '__getattr__': attribute name cannot be converted from 'StringLiteral["attr"]' to 'Int}}
    obj.attr


struct GetSettable:
  fn __getitem__(self, x: Int) -> Int: pass
  fn __setitem__(self, x: Int, y: Int): pass

struct NoSelfCtor:
  var x: Int
  fn __init__(out self, x: Int):
    self.x = x

fn test_int_to_int_error(a: Int, b: NoSelfCtor):
  # expected-error @+1 {{cannot construct 'NoSelfCtor' with itself, you can remove the constructor call}}
  _ = NoSelfCtor(NoSelfCtor(a))

  # expected-error @+1 {{invalid initialization: expected at most 0 positional arguments, got 1}}
  _ = GetAttrNotString(a)



##===----------------------------------------------------------------------===##
# lambda not supported yet
##===----------------------------------------------------------------------===##

def testLambda():
  # expected-error @+1 {{Mojo doesn't support lambda expressions yet}}
  _ = lambda x, y: x+y

def testLambda2():
  # expected-error @+1 {{Mojo doesn't support lambda expressions yet}}
  _ = lambda (x: Int, y: Float64) raises: x+y

def testInExpr(x: Int, y: Int):
  # expected-error @+1 {{'Int' does not implement the '__contains__' method}}
  _ = x in y
  # expected-error @+1 {{'Int' does not implement the '__contains__' method}}
  _ = x not in y

##===----------------------------------------------------------------------===##
# References and Transfer
##===----------------------------------------------------------------------===##

struct CopyAndInitMemType:
  fn __init__(out self): pass
  fn __copyinit__(out self, other: Self): pass
  # expected-note @+1 {{function declared here}}
  fn __le__(self, other: Self) -> Self: return self
  fn __mlir_i1__(self) -> __mlir_type.i1: pass

fn compare_mem_result():
  var x = CopyAndInitMemType()
  # https://github.com/modular/mojo/issues/1115
  # expected-error @+1 {{chained comparison operator does not currently support memory-only return types}}
  x <= x <= x

fn test_bad_ref(a: Int, b: CopyAndInitMemType):

  var bref = Pointer(to=b) # ok

  # expected-error @+1 {{invalid call to '__le__': right side cannot be converted from 'Pointer[CopyAndInitMemType, b]' to 'CopyAndInitMemType'}}
  _ = b <= bref

fn transfer_diags[param: String](borrowed_arg: CopyAndInitMemType, obj: SomeNonTrivRegPassable, *vararg: String):
  var mem3 = CopyAndInitMemType()

  # Test pointless transfers from RValues and trivial values.
  # These should warn and not create IR transfers.

  # First transfer is ok.
  _ = mem3^
  _ = mem3^^ # expected-warning {{transfer from an owned value has no effect and can be removed}}

  # Already an rvalue.
  _ = CopyAndInitMemType()^ # expected-warning {{transfer from an owned value has no effect and can be removed}}

  var someInt = 4
  _ = someInt^ # expected-warning {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}

  var someInt2 = 4
  someInt2 = 4
  _ = someInt2^ # expected-warning {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}

  # MOCO-757: Transfer ^ of read-only arg leads to double free
  # expected-error @+1 {{cannot transfer out of immutable reference}}
  _ = borrowed_arg^

  # expected-error @+1 {{cannot transfer out of immutable reference}}
  _ = obj^

  # expected-error @+1 {{cannot transfer from a parameter expression; did you want to introduce a local 'var'?}}
  _ = param^

  # DLValue.
  # expected-error @+1 {{expression does not designate a value with an origin}}
  _ = vararg[1]^

# Issue #1708: https://github.com/modular/mojo/issues/1708
# Issue #1699: https://github.com/modular/mojo/issues/1699
# Issue #30790: https://github.com/modularml/modular/issues/30790
struct SomeThing:
    fn overloaded[a: Int](self, b: Int) -> Int: pass
fn testSomeThing(a: SomeThing):
  # expected-error @below {{cannot emit closure for method 'overloaded'}}
  # expected-note @below {{computing member method closure is not yet supported}}
  # expected-note @below {{did you forget '()'s?}}
   a.overloaded[4] / 1.0

# Test invalid references that cannot bind to potentially-register_passable
# argument values.
# Issue #32603: References to read-only args in generics miscompile when instantiated on regpassable types
fn get_ref_to_bad_argument[T: AnyType](a: T, *args: T):
  # These are all fine since they are not returned.
  _ = Pointer(to=a)
  _ = __origin_of(a)
  _ = __get_mvalue_as_litref(a)
  # This is okay. The VariadicListMem has a origin.
  _ = Pointer(to=args)
  _ = Pointer(to=args[0])

@register_passable
struct NonTrivialReg:
  pass

fn get_ref_to_reg_variadic(*args: NonTrivialReg):
  _ = Pointer(to=args[0])

fn variadic_int(*x: Int) -> Bool: pass

# https://github.com/modularml/modular/issues/34675
fn invalid_call_variadic_int(a: Int):
    @parameter
    # expected-error @+1 {{cannot use dynamic value in '@parameter if' condition}}
    if variadic_int(a, a):
        pass

fn test_bad_ref_errors[T: AnyType](a: Pointer[T, _], b: Pointer[T, _]):
  # expected-error @below {{cannot implicitly convert 'T' value to 'Pointer[T, origin]'}}
  var x : Pointer[T, b.origin] = a[]

  # expected-error @below {{cannot implicitly convert 'T' value to 'Pointer[T, MutableAnyOrigin]'}}
  var y : Pointer[T,  __mlir_attr.`#lit.any.origin<1>: !lit.origin<1>`, a.address_space] = a[]

fn test_subscript_conflict(a: Int):
  # expected-error @below {{duplicate keyword parameter 'idx'}}
  # expected-note @below {{previously specified here}}
  _ = a[idx=4, idx=7]


struct Addable:
    fn __add__(self, other: Self): pass # expected-note {{function declared here}}
fn test(a: Pointer[Addable, _], b: Addable):
    # FIXME: This shouldn't mention is_mutable since it is an implicit parameter.
    # expected-error @+1 {{invalid call to '__add__': right side cannot be converted from 'Pointer[Addable, origin]' to 'Addable'}}
    _ = b+a


# Verify that we can propagate parametric mutability through field accesses.
struct ThingWithFields:
  var field: Int

fn field_sensitive_origins(a: ThingWithFields)
    -> Pointer[ThingWithFields, __origin_of(a.field)]:

  # expected-error @+1 {{'ThingWithFields' value has no attribute 'field_abc'}}
  _ = __origin_of(a.field_abc)
  # expected-error @+1 {{MLIR type 'index' has no attributes}}
  _ = __origin_of(Index.field_abc)

  # expected-error @+1 {{cannot implicitly convert 'ThingWithFields' value to 'Pointer[ThingWithFields, a.field]'}}
  return a


fn bad_named_return(out output: String):
   output = "emplaced!"
   # expected-note @below {{remove the expression if the return slot is already initialized}}
   # expected-error @below {{'return' in function with a named return slot should not return the slot itself}}
   return output


fn bad_named_return2(out output: Int):
   output = 42
   # expected-note @below {{remove the expression if the return slot is already initialized}}
   # expected-error @below {{'return' in function with a named return slot should not return the slot itself}}
   return output

fn unbound_function_type():
  # expected-error @below {{function type missing required origin set parameter}}
  var f: fn() [_] -> None


# Crash converting mvalue of #lit.any.origin origin to Pointer with specific one.
# https://github.com/modular/mojo/issues/1921
struct SomeStruct:
  fn refBindingToImmortal(mut self, ptr: UnsafePointer[Int])
      -> Pointer[Int, __origin_of(self)]:
    # expected-error @below {{cannot implicitly convert 'Pointer[Int, MutableAnyOrigin]' value to 'Pointer[Int, self]'}}
    return Pointer(to=ptr[])

# Various type printing cases.
#
struct HasKWOnlyParam[*, kwplz: Int]: pass

# expected-note @+1 {{function declared here}}
fn test_kw_only[a: Int](arg: HasKWOnlyParam[kwplz=a]):
  # expected-error @+1 {{cannot be converted from 'HasKWOnlyParam[kwplz=a]' to 'HasKWOnlyParam[kwplz=42]'}}
  test_kw_only[42](arg)

struct HasMultipleOnlyParam[x: Int = 1, y: Int = 4]: pass

# expected-note @+1 {{function declared here}}
fn test_mixedkw_only[a: Int](arg: HasMultipleOnlyParam[1, a]):
  # expected-error @+1 {{cannot be converted from 'HasMultipleOnlyParam[y=a]' to 'HasMultipleOnlyParam[y=42]'}}
  test_mixedkw_only[42](arg)

fn int_fn(arg: Int) -> Int: return arg+1
struct HasDependent[x: Int, y: Int = int_fn(x)]: pass

# expected-note @+1 {{function declared here}}
fn test_dependent[a: Int](arg: HasDependent[a], arg2: HasDependent[a, 4]):
  # expected-error @+1 {{argument #0 cannot be converted from 'HasDependent[a]' to 'HasDependent[42]'}}
  test_dependent[42](arg, arg2)
  # expected-error @below {{argument #1 cannot be converted from 'HasDependent[a]' to 'HasDependent[a, 4]'}}
  # expected-note @below {{types parameters include unfolded expression at parser time; try rebinding to a consistent type?}}
  test_dependent(arg, arg)

struct HasIntParam[p: Int]:
  fn __init__(out self): # expected-note {{function declared here}}
     pass

# MOCO-846: Poor error message when type conversion fails due to IntLiteral materialization

# expected-note @below {{function declared here}}
fn take_dep_args[width: Int, x: IntLiteral](a: HasIntParam[width], b: HasIntParam[width * 4]):
  # expected-error @below {{cannot be converted from 'HasIntParam[(value * 4)]' to 'HasIntParam[(value * 4)]'}}
  take_dep_args[x, x](HasIntParam[x](), HasIntParam[x*4]())

fn test_signature():
  # expected-error @+1 {{cannot implicitly convert 'fn() -> HasIntParam[1]' value to 'fn(x: HasIntParam[1]) -> None' in 'var' initializer}}
  var x : fn(x: HasIntParam[1])->None = HasIntParam[1].__init__

  # expected-error @+1 {{use of unknown declaration 'UndefinedStruct'}}
  var y : fn(x: UndefinedStruct)->None = HasIntParam[1].__init__

  var z : fn(out x: HasIntParam[1]) = HasIntParam[1].__init__

  var str : HasIntParam[1]
  # expected-error @+1 {{invalid call to '__init__': expected at most 0 positional arguments, got 1}}
  HasIntParam[1].__init__(str)

fn bad_union[ao: Origin[True]](ref [ao] a: String, mut b: String) -> ref [a, b] String:
    var c: String
    # expected-error @below {{cannot return reference with incompatible origin: 'c' vs '{b, ao._mlir_origin}'}}
    return c

# https://github.com/modular/mojo/issues/3829
fn apply_in_memory[o: ImmutableOrigin](f: fn(ref[o] x: SomeNonTrivRegPassable) -> None, x: SomeNonTrivRegPassable):
# expected-error @below {{argument #0 cannot be converted from 'SomeNonTrivRegPassable' to ref 'SomeNonTrivRegPassable'}}
# expected-note @below {{operand origin 'x' doesn't match expected origin 'o'}}
    f(x)


# Problems binding SRValues to ref arguments.
# https://github.com/modular/mojo/issues/3830

fn getSomeNonTrivRegPassable() -> SomeNonTrivRegPassable: pass
# expected-note @below {{function declared here}}
fn direct3830(ref a: SomeNonTrivRegPassable, ref[a] b: SomeNonTrivRegPassable): pass
fn test3830():
    # expected-error @below {{invalid call to 'direct3830':}}
    # expected-note @below {{failed to infer parameter #1, parameter inferred to two different values}}
    direct3830(getSomeNonTrivRegPassable(), getSomeNonTrivRegPassable())

fn test3830_1[o: ImmutableOrigin](f: fn(ref[o] x: SomeNonTrivRegPassable) -> None):
    # expected-error @below {{invalid indirect call: argument #0 cannot be converted from 'SomeNonTrivRegPassable' to ref 'SomeNonTrivRegPassable'}}
    # expected-note @below {{operand origin '*"anonymous*"' doesn't match expected origin 'o'}}
    f(getSomeNonTrivRegPassable())

fn test3830_2[o: ImmutableOrigin](f: fn(ref[o] x: Int) -> None, x: Int):
    # expected-error @below {{invalid indirect call: argument #0 cannot be converted from 'Int' to ref 'Int'}}
    # expected-note @below {{operand origin '*"anonymous*"' doesn't match expected origin 'o'}}
    f(x)




##===----------------------------------------------------------------------===##
# MergeWith
##===----------------------------------------------------------------------===##

struct TypeA:
    fn __merge_with__[other_type: __type_of(TypeB)](self) -> TypeB:
        pass
    fn __merge_with__[other_type: __type_of(TypeC)](self) -> Int:
        pass

struct TypeB:
    fn __merge_with__[other_type: __type_of(TypeA)](self) -> Int:
        pass

struct TypeC:
    pass


# CHECK-LABEL: lit.fn @"test_mergewith
fn test_mergewith(cond: Bool, a: TypeA, b: TypeB, c: TypeC):
  # expected-error @+2 {{value of types 'TypeA' and 'TypeB' have '__merge_with__' methods that disagree on common type}}
  # expected-note @+1 {{one returns 'TypeB' and the other returns 'Int'}}
  _ = a if cond else b

  # expected-error @+2 {{value of types 'TypeA' and 'TypeC' cannot be merged to type 'Int'}}
  # expected-note @+1 {{'TypeC' does not implicitly convert to 'Int'}}
  _ = a if cond else c
