# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Functions
##===----------------------------------------------------------------------===##

struct NotBoolConvertible: pass
# expected-note @+1 {{function declared here}}
fn test_bool_context(a: NotBoolConvertible): pass

fn voidReturningFn(): pass
fn badCall():
  # expected-error @+1 {{invalid call to 'test_bool_context': argument #0 cannot be converted from 'None' to 'NotBoolConvertible'}}
  test_bool_context(voidReturningFn())


fn missing_ret_val() -> __mlir_type.index:
  return # expected-error {{cannot implicitly convert 'None' value to '__mlir_type.index' in return value}}

fn ret_type_mismatch() -> __mlir_type.index:
  return 4.0 # expected-error {{cannot implicitly convert 'FloatLiteral[4]' value to '__mlir_type.index' in return value}}

async fn testAsyncVoid(): pass
async fn testAsyncInt() -> Int: return 42

fn callsWith():
  testAsyncVoid() # expected-warning {{awaitable 'Coroutine[None, {}]' value was never awaited}}
  testAsyncInt() # expected-warning {{awaitable 'Coroutine[Int, {}]' value was never awaited}}


struct ThingWithStaticMethod:
   @staticmethod
   fn splat(x: Int): # expected-note {{function declared here}}
     pass

fn testThingWithStaticMethod():
  # expected-error @+1 {{invalid call to 'splat': argument #0 cannot be converted from 'FloatLiteral[4]' to 'Int'}}
  ThingWithStaticMethod.splat(4.0)


def top_level_fn(a: Int):
    # expected-error @below {{TODO: closures cannot have parameters}}
    fn bar[b: Int]() -> Int:
      return a

def use_non_copyable_type(a: ThingWithStaticMethod):
  pass

def test_use_non_copyable_type(var b: ThingWithStaticMethod):
  use_non_copyable_type(b^)


@fieldwise_init
struct MemType(ImplicitlyCopyable, Movable):
    pass

# COM: Issue https://github.com/modularml/modular/issues/37758 where the
# COM: key test is that the below is not crashing due to assertion violation.
# expected-error @+1 {{expected ':' in function definition}}
fn missingColon(x: Int)
  return x

fn out1(a: Int, out b: Int): pass

# expected-error @+1 {{'out' convention may not be variadic}}
fn bad_out2(out *b: Int): pass

# expected-error @+1 {{expected ']' for parameter list}}
fn bad_out3[out x: Int](): pass

# expected-error @+1 {{function may not have multiple 'out' arguments}}
fn bad_out4(out a: Int, out b: Int): pass

# expected-error @+1 {{function cannot have both an 'out' argument and an explicit result type}}
fn bad_out5(out a: Int) -> Int: pass

# expected-error @+1 {{function cannot have both an 'out' argument and an explicit result type; remove the '-> None' to fix it}}
fn bad_out6(out self) -> None: pass

struct BadInitResult:
  # expected-error @+1 {{__init__ method must return Self type with 'out' argument}}
  fn __init__(mut self) raises -> None:
    pass

struct BadInitType:
    # expected-error @below {{__init__ method must return Self type with 'out' argument}}
    # expected-error @below {{self argument must be present in instance method}}
    fn __init__():
        pass

# expected-error @+1 {{argument type must be specified}}
def defaultArgumentUntyped(a=1):
    pass

##===----------------------------------------------------------------------===##
# Default Arguments, VarArgs, and Packs
##===----------------------------------------------------------------------===##

# COM: Issue https://github.com/modular/mojo/issues/1091
fn missing_arg_type_or_default(
    a: Int = 9,
    # expected-error @+2 {{required positional argument follows optional positional argument}}
    # expected-error @+1 {{argument type must be specified}}
    b,
    c: Int,  # expected-error {{required positional argument follows optional positional argument}}
    d: Int = 0,
    # expected-error @+2 {{required positional argument follows optional positional argument}}
    # expected-error @+1 {{argument type must be specified}}
    e,
    # expected-error @+1 {{argument type must be specified}}
    **kwargs,
):
    pass

def missing_default(
    a: Int=9,
    b: Int,  # expected-error {{equired positional argument follows optional positional argument}}
    c: Int=0,
    d: Int,  # expected-error {{required positional argument follows optional positional argument}}
):
    pass

# expected-error @+1 {{use of unknown declaration 'unknown'}}
fn defaultArgumentUnknownDeclaration(a: Int = unknown): pass

# expected-error @+1 {{cannot use a dynamic value in default argument}}
fn defaultArgumentReferencesArgument(a: Int = 0, b: Int = a): pass

# expected-error @+1 {{cannot implicitly convert 'FloatLiteral[1]' value to 'Int'}}
fn defaultArgumentBadType(a: Int = 1.0): pass

# expected-error @+1 {{'mut' arguments may not have defaults}}
fn byref_default(mut x: Int = 2): pass

# expected-error @below {{'**' marker must be at end of argument list}}
fn starStarLast(**a: Int, b: Int): pass

# expected-error @below {{'**' marker must be at end of argument list}}
fn twoStarStar(**a: Int, **b: Int): pass

# expected-error @+1 {{expected argument name}}
fn starSpaceStar(* *a: Int): pass

# expected-error @+1 {{variadic arguments may not have defaults}}
fn noDefaultVariadics(*a: Int = 42): pass

# expected-note @+1 {{function declared here}}
fn exampleVariadic(a: FloatLiteral, *b: Int): pass
# expected-note @+1 {{function declared here}}
fn exampleVariadicAndKeyword(*a: Int, b: Int): pass
# expected-note @+1 {{function declared here}}
fn exampleByRefVariadic(a: FloatLiteral, mut *b: Int): pass
# expected-note @+1 {{function declared here}}
fn parameterizedVariadic[T: __mlir_type.`!kgen.type`](*args: T): pass

fn ownedPack[*Ts: AnyType](var *args: *Ts): pass
fn ownedVariadic(var *args: Inner): pass
fn ownedVariadicReg(var *args: WrongType): pass


# expected-note @+1 {{struct declared here}}
struct ParameterizedStruct[T: __mlir_type.`!kgen.type`]:
    # expected-note @+1 {{function declared here}}
    def __init__(out self, *args: Self.T):
        pass

@fieldwise_init
struct TestTuple[*Ts: AnyTrivialRegType]:
    # expected-note @+1 {{function declared here}}
    fn test[i: Int, j: Int](self):
        pass

fn badCalls(arg: Int):
  # expected-error @+1 {{argument #1 cannot be converted from 'FloatLiteral[1]' to 'Int'}}
  exampleVariadic(1.0, 1.0)
  # expected-error @+1 {{argument #3 cannot be converted from 'FloatLiteral[1]' to 'Int'}}
  exampleVariadic(1.0, 1, 2, 1.0)
  # expected-error @+1 {{argument #3 cannot be converted from 'FloatLiteral[4]' to 'Int'}}
  exampleVariadicAndKeyword(1, 2, 3, b=4.0)

  var x: Int
  var y : FloatDyn
  # expected-error @+1 {{invalid call to 'exampleByRefVariadic': argument #2 must be mutable in order to pass to a mutating argument}}
  exampleByRefVariadic(1.0, x, arg)
  # expected-error-re @+1 {{invalid call to 'exampleByRefVariadic': l-value of type 'FloatDyn' cannot be converted to reference of type 'Int'}}
  exampleByRefVariadic(1.0, x, y)
  # expected-error @+1 {{argument #2 must be mutable in order to pass to a mutating argument}}
  exampleByRefVariadic(1.0, x, 1)

  # The user hasn't provided any arguments that could be used to infer `T`.
  # expected-error @below {{failed to infer parameter 'T', it isn't used in any argument}}
  parameterizedVariadic()
  # expected-error @below {{failed to infer parameter 'T' of parent struct 'ParameterizedStruct', it isn't used in any argument}}
  var z = ParameterizedStruct()

  # We can't infer `T` with two arguments of different types.
  # expected-error @below {{invalid call to 'parameterizedVariadic': failed to infer parameter 'T', it inferred to two different values: 'Int' and 'FloatDyn'}}
  # expected-note @below {{try `rebind` them to one type if they will be concretized to the same type}}
  parameterizedVariadic(1, 2.0)

  # expected-error @below {{invalid call to 'test': failed to infer parameter 'j', it isn't used in any argument}}
  TestTuple[Int, FloatLiteral]().test[1]()

fn badError(a: ParameterizedStruct[Int]):
  # expected-error @+1 {{cannot implicitly convert 'ParameterizedStruct[Int]' value to 'ParameterizedStruct[Bool]'}}
  var b: ParameterizedStruct[Bool] = a

# expected-note @below {{candidate declared here}}
fn overloadedFunc(x: Int): pass
# expected-note @below {{candidate declared here}}
fn overloadedFunc(x: Int, y: Int): pass

# expected-note @below {{function declared here}}
fn takeFuncArgument(f: Int): pass

fn callWithOverloadedArg():
  # expected-error @below {{invalid call to 'takeFuncArgument': argument #0 cannot be converted from unknown overload to}}
  # expected-error @below {{cannot convert function to non-function type 'Int'}}
  # expected-note @below {{try resolving the overloaded function first}}
  takeFuncArgument(overloadedFunc)

# expected-error @+1 {{unexpected token in expression}}
fn invalidStarExpression(*x: *): pass

# expected-error @+1 {{pack argument type list must reference a variadic list}}
fn invalidPackType(*x: *Int): pass

fn invalidParameterPack[*Ts: AnyType]():
  # expected-error @+1 {{parameters may not be variadic packs}}
  fn invalid[*Us: *Ts](): pass

# expected-error @+2 {{only variadic arguments' types can be unpacked}}
# expected-note @+1 {{'x' is not a variadic argument}}
fn invalidArgumentUnpack[*Ts: AnyType](x: *Ts): pass

# expected-error @+1 {{argument already has a convention specified}}
fn invalidOwned(var var x: Int): pass

# expected-note @+1 {{function declared here}}
fn examplePack[*Ts: AnyType](*args: *Ts):
  pass

fn packArgOverload():
  pass

fn packArgOverload(x: Int):
  pass


# expected-note @+1 {{function declared here}}
fn first_and_rest[T: AnyTrivialRegType, *Ts: AnyType](*values: *Ts):
    pass


fn unresolvedPackCall[*t : AnyType](var *args: *t):
  # expected-error @below {{invalid call to 'examplePack': assigning 0 operands to an unresolvable variadic pack argument}}
  var _ = examplePack[*t]()

fn badPackCalls(value: Int):
  # expected-error @+1 {{invalid call to 'examplePack': callee with non-empty variadic pack argument expects 1 positional operand, but 2 were specified}}
  examplePack[Int](1, 2)
  # expected-error @+1 {{invalid call to 'examplePack': callee with non-empty variadic pack argument expects 2 positional operands, but 1 was specified}}
  examplePack[Int, FloatDyn](1)
  # expected-error-re @below {{invalid call to 'examplePack': argument #1 cannot be converted from '__mlir_type.index' to 'FloatDyn'}}
  examplePack[Int, FloatDyn](1, Int(2)._mlir_value)
  # expected-warning @below {{could not infer parameter type for this value, because it is not concrete}}
  # expected-error @below {{invalid call to 'examplePack': failed to infer parameter 'Ts', it isn't used in any argument}}
  examplePack(packArgOverload)
  # expected-error @below {{invalid call to 'first_and_rest': failed to infer parameter 'T', it isn't used in any argument}}
  first_and_rest(value)

struct TestPackErrorMessage[*Ts: AnyType]:
    # expected-error @below {{'self' argument must have type 'TestPackErrorMessage[Ts]', but actually has type 'VariadicPack[False, args, AnyType, Ts]'}}
    # expected-error @below {{__init__ method must return Self type with 'out' argument}}
    fn __init__(*args: *Self.Ts):
         pass

# expected-error @+1 {{variadic pack elements declared as 'AnyTrivialRegType' are removed, please declare elements as 'AnyType' instead of 'AnyTrivialRegType'}}
fn badAnyRegPack[*Ts: AnyTrivialRegType](*args: *Ts):
  pass

# always_inline("builtin")

# expected-error @+2 {{'@always_inline("builtin")' does not support this argument convention}}
@always_inline("builtin")
async fn always_inline_builtin_1(): pass

# expected-error @+2 {{'@always_inline("builtin")' does not support this argument convention}}
@always_inline("builtin")
fn always_inline_builtin_2(a: MemType): pass

# expected-error @+2 {{'@always_inline("builtin")' does not support this argument convention}}
@always_inline("builtin")
fn always_inline_builtin_3() raises: pass

@always_inline("builtin")
fn always_inline_builtin_4(a: Bool):
  # expected-error @+1 {{'@always_inline("builtin")' does not support MLIR operation hlcf.elif}}
  if a:
     pass

# expected-note @+1 {{function declared here}}
fn simple_constraints[x: Int, y: Int]()
  # expected-note @+1 {{constraint declared here}}
  where x > 1
  # expected-note @+1 {{constraint declared here}}
  where y < 10:
    pass

# expected-note @below {{cannot evaluate call to non-builtin function declared here}}
fn unfoldable_predicate(y: Int) -> Bool:
  return y > 2

# expected-note @below {{function declared here}}
# expected-note @below {{cannot prove constraint}}
fn unprovable_constraints[x: Int, y: Int]()
  # expected-note @+1 {{constraint declared here}}
  where x > 1
  # expected-note @+1 {{constraint declared here}}
  where unfoldable_predicate(y):
    pass

fn test_constraints():
  # expected-error @+1 {{violated constraint}}
  simple_constraints[0, 0]()
  # expected-error @+1 {{violated constraint}}
  simple_constraints[2, 11]()
  # expected-error @+1 {{violated constraints}}
  simple_constraints[0, 11]()

  # expected-error @+1 {{violated constraint}}
  unprovable_constraints[0, 0]()
  # expected-error @below {{invalid call to 'unprovable_constraints': lacking evidence to prove correctness}}
  # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
  unprovable_constraints[2, 0]()

# expected-note @below {{cannot prove constraint}}
struct ConstraintStruct[
  # expected-note @below {{constraint declared here}}
  a: Int where a > 0
]:
    pass

# expected-error @below {{invalid bindings for 'ConstraintStruct': lacking evidence to prove correctness}}
fn use_constraint_struct[x: Int, cs: ConstraintStruct[x]]():
    pass

# expected-error @below {{default value violated constraint}}
fn violated_default_constraint[x: Int where x > 3 = 1]():
    pass

# There should NOT be any errors / warnings here.
fn unprovable_default_constraint[x: Int = 3, y: Int where x + y > 3 = 1]():
    pass

##===----------------------------------------------------------------------===##
# Function Overloading
##===----------------------------------------------------------------------===##

# expected-note @+1 {{previous definition here}}
def fn_redecl(): pass
# expected-error @+1 {{redefinition of function 'fn_redecl' with identical signature}}
def fn_redecl(): pass

# expected-note @+1 {{previous definition here}}
def fn_redecl2() -> Int: pass
# expected-error @+1 {{redefinition of function 'fn_redecl2' cannot overload on return type only}}
def fn_redecl2() -> FloatDyn: pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #0 cannot be converted from 'TestOverloading' to 'Int'}}
# expected-note @below {{candidate not viable: expected at most 1 positional argument, got 2}}
fn overloadIntFloat32(a: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note-re @below {{candidate not viable: argument #0 cannot be converted from 'TestOverloading' to 'FloatDyn'}}
# expected-note @below {{candidate not viable: expected at most 1 positional argument, got 2}}
fn overloadIntFloat32(a: FloatDyn): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: missing 1 required positional argument: 'b'}}
# expected-note-re @below {{candidate not viable: argument #1 cannot be converted from 'FloatDyn' to 'Int'}}
fn overloadIntFloat32(a: Int, b: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: missing 1 required positional argument: 'b'}}
# expected-note @below {{argument #1 must be mutable in order to pass to a mutating argument}}
fn overloadIntFloat32(a: Int, mut b: FloatDyn): pass

# expected-note @below {{candidate not viable: missing 2 required positional arguments: 'b', 'c'}}
# expected-note @below {{candidate not viable: missing 1 required positional argument: 'c'}}
# expected-note @below {{candidate declared here}}
fn overloadIntFloat32(a: Int, mut b: FloatDyn, c: Int, *args: Int): pass

struct TestOverloading:
  var a: Int   # expected-note {{cannot overload with this non-function definition}}
  fn a(self):  # expected-error {{invalid redefinition of 'a'}}
    pass

  fn test(self, a: Int, b: FloatDyn):
    # expected-note @below {{did you mean to call it?}}
    # expected-error @below {{cannot form a reference to overloaded declaration}}
    var bad = overloadIntFloat32

    # expected-error @+1 {{no matching function in call}}
    overloadIntFloat32(self)
    # expected-error @+1 {{no matching function in call}}
    overloadIntFloat32(a, b)

@fieldwise_init
struct OverloadedKwArgs:
    var vals: List[Int]

    # expected-note @below {{previous definition here}}
    fn __getitem__(self, idx: Int) -> Int:
        return self.vals[idx]

    # expected-error @below {{redefinition of function '__getitem__' cannot overload on return type only}}
    fn __getitem__(self, *, idx: Int) -> Bool:
        return self.vals[idx] > 0

# Test that static methods don't get dispatched if their first arg is self type.
struct StructWithStaticMethod:
    fn __init__(out self): pass

    # expected-note @+2 {{function declared here}}
    @staticmethod
    fn bar(mut f: StructWithStaticMethod): pass


fn test_static_overload():
    var a = StructWithStaticMethod()
    # expected-error @below {{invalid call to 'bar': missing 1 required positional argument: 'f'}}
    a.bar()


# expected-note @+1 {{function declared here}}
fn takesAtLeastOneInt(x: Int, *y: Int): pass
fn badTakesAtLeastOneInt():
  # expected-error @+1 {{invalid call to 'takesAtLeastOneInt': missing 1 required positional argument: 'x'}}
  takesAtLeastOneInt()


# COM: Issue #23007
# expected-note @+1 {{function declared here}}
fn too_few_pos_only(a: Int, b: Int, /, msg: Int = 2): pass

fn test_too_few_pos_only(a: Int, msg: Int = 3):
  # expected-error @+1 {{invalid call to 'too_few_pos_only': missing 1 required positional argument: 'b'}}
  too_few_pos_only(a, msg=msg)


# COM: Issue #23007
# expected-note @+1 {{function declared here}}
fn missing_args(a: Int, b: Int, c: Int = 2, d: Int = 2): pass

fn test_missing_args():
  # expected-error @+1 {{invalid call to 'missing_args': missing 2 required positional arguments: 'a', 'b'}}
  _ = missing_args(c=1, d=1)


struct ConvertibleFromInt:
  @implicit
  fn __init__(out self, value: Int):
    pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #1 cannot be converted from 'ConvertibleFromInt' to 'Int'}}
fn ambiguousConversions(a: ConvertibleFromInt, b: Int): pass
# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #0 cannot be converted from 'ConvertibleFromInt' to 'Int'}}
fn ambiguousConversions(a: Int, b: ConvertibleFromInt): pass

fn testAmbiguousConversions(a: Int, b: ConvertibleFromInt):
  ambiguousConversions(a, b) # ok
  ambiguousConversions(b, a) # ok
  # expected-error @+1 {{ambiguous call to 'ambiguousConversions', each candidate requires 1 implicit conversion, disambiguate with an explicit cast}}
  ambiguousConversions(a, a)
  # expected-error @+1 {{no matching function in call}}
  ambiguousConversions(b, b)

  var localFn = testAmbiguousConversions  # ok
  localFn(a, b)
  localFn(a, a)
  localFn(1, b)
  # expected-error @+1 {{invalid indirect call: argument #0 cannot be converted from 'ConvertibleFromInt' to 'Int'}}
  localFn(b, b)

# COM: https://github.com/modular/mojo/issues/1530
# COM: Do not crash when explicitly unbound parameter cannot be deduced due to missing arguments.
struct Parametric[a: Int]: pass

# expected-note @below {{function declared here}}
fn takes_same_arg_types[x: Int](a: Parametric[x], b: Parametric[x]): pass

fn test_param_deduction_failure[
    func: fn[y: Int] (c: Parametric[y], d: Parametric[y]) -> None,
](u: Int, v: Int):
    # expected-error @+1 {{missing 1 required positional argument: 'b'}}
    takes_same_arg_types[_](u)

    # expected-error @below {{invalid call to 'takes_same_arg_types': failed to infer parameter 'x', it isn't used in any argument}}
    takes_same_arg_types[_](u, v)

    # expected-error @+1 {{missing 1 required positional argument: 'd'}}
    func[_](u)

    # TODO: This is because we're not inferring signatures correctly
    # expected-error @below {{invalid indirect call: failed to infer parameter 'y', it isn't used in any argument}}
    func[_](u, v)

struct InitOverloaded:
  # expected-note @below {{argument #0 cannot be converted from 'StringLiteral["foo"]' to 'Int'}}
  # expected-note @below {{argument #0 cannot be converted from 'Parametric[1]' to 'Int'}}
  fn __init__(out self, a: Int): pass
  # expected-note @below {{argument #0 cannot be converted from 'StringLiteral["foo"]' to '__mlir_type.index'}}
  # expected-note @below {{argument #0 cannot be converted from 'Parametric[1]' to '__mlir_type.index'}}
  fn __init__(out self, a: __mlir_type.index): pass

fn testOverloadInitError(a: InitOverloaded, b: Parametric[1], c: Int):
  # expected-error @+1 {{cannot construct 'InitOverloaded' with itself, you can remove the constructor call}}
  _ = InitOverloaded(a)

  # expected-error @+1 {{no matching function in initialization}}
  _ = InitOverloaded(b)

  # This is ok
  _ = InitOverloaded(c)

  # expected-error @+1 {{no matching function in initialization}}
  _ = InitOverloaded("foo")

  # Ambiguous initializer list assigning to discard pattern needs to be an error.
  # expected-error @+1 {{cannot emit initializer list without a contextual type}}
  _ = {a = 1, b = 2}


##===----------------------------------------------------------------------===##
# Structs
##===----------------------------------------------------------------------===##

# expected-note @below {{originally resolving it here}}
struct Rec[
  # expected-error @below {{attempt to resolve a recursive reference to declaration}}
  param: Rec]:
  pass


struct Rec1[# expected-note {{originally resolving it here}}
 # expected-note @below {{referenced through this use}}
  p1: Rec2]: pass

struct Rec2[
  # expected-error @below {{attempt to resolve a recursive reference to declaration}}
  p2: Rec1]:


# expected-error @+1 {{'def' statement must be on its own line}}
struct Struct: def foo(mut self): pass

struct ReturnFromStruct:
  # expected-error @+1 {{cannot return from this context}}
  return 42

struct ReDef: pass # expected-note {{conflicts with this previous struct declaration}}
struct ReDef: pass # expected-error {{invalid redefinition of 'ReDef'}}

# Ambiguous Lookup Case for Referencing Redefined Struct (ALCFRRS)
# This tests that we don't crash or anything when we reference a redefined
# struct.
fn reference_redefined_struct(arg: ReDef):
  pass

struct StructMemberRedefinition:
  var x : __mlir_type.index  # expected-note {{previous definition here}}
  var x : __mlir_type.index  # expected-error {{invalid redefinition of 'x'}}

struct SpecialFunctions:
  # expected-error @+1 {{'__new__' is not supported on structs}}
  fn __new__() -> Self:
    pass

  # expected-error @+1 {{'__add__' requires 2 operands}}
  fn __add__(self):
    pass

  # This is ok, SpecialFunctions may be a reference semantic struct like Object.
  # Issue #7573: Cannot have LValue or RValue overloads of special methods
  fn __iadd__(a: SpecialFunctions, b: SpecialFunctions): pass

  # expected-error @+1 {{'__iadd__' result type must be elided (or None)}}
  fn __iadd__(mut self, rhs: SpecialFunctions) -> SpecialFunctions: pass

  fn failures(self):
    self+self # Supports this, even though it isn't valid.  Shouldn't crash.
    self*self # expected-error {{'SpecialFunctions' does not implement the '__mul__' method}}

  # expected-error @+1 {{'__del__' cannot be declared as raising an exception}}
  fn __del__(deinit self) raises:
     pass

struct TestOwnedDeinitWarnings:
  # expected-warning @+1 {{'owned' has been deprecated, use 'deinit' instead}}
  fn __del__(owned self): pass

  # expected-warning @+1 {{'owned' has been deprecated, use 'deinit' instead}}
  fn __moveinit__(out self, owned x: TestOwnedDeinitWarnings): pass

  # expected-warning @+1 {{'owned' has been deprecated, use 'var' instead}}
  fn method(owned x): pass

struct TestVarDeinitErrors:
  # expected-error @+1 {{the 'self' argument should be declared 'deinit'}}
  fn __del__(var self): pass

  # expected-error @+1 {{the 'existing' argument should be declared 'deinit'}}
  fn __moveinit__(out self, var x: String): pass


@register_passable
struct WrongType:
  # expected-error @+2 {{__init__ method must return Self type with 'out' argument}}
  # expected-error @+1 {{'self' argument must have type 'WrongType', but actually has type 'None'}}
  def __init__(self: None): pass

  # expected-error @+1 {{'self' argument must have type 'WrongType', but actually has type 'Int'}}
  fn __init__(out self: Int): pass

  # expected-error @+1 {{existing value argument must be passed as 'read'}}
  fn __copyinit__(out self, mut existing: Self): pass

  # TODO: Should err.
  fn __copyinit__(out self, existing: Int): pass

  # expected-error @+1 {{'@register_passable' types may not have a '__moveinit__' method, they are always movable by copying a register}}
  fn __moveinit__(out self, deinit existing: Self): pass


struct WrongSelfType[a: Int]:
  # expected-error @+1 {{'self' argument must have type 'WrongSelfType[a]', but actually has type 'Int'}}
  fn badMethod(self: Int): pass
  fn goodMethod(mut self: WrongSelfType[Self.a]): pass

  # Issue #13358
  # expected-error @+1 {{'__copyinit__' requires 1 operand}}
  fn __copyinit__(out self, other: Self, moar: Int): pass

  # expected-error @+1 {{'__add__' requires 2 operands}}
  fn __add__(self): pass

  fn __pow__(self, exp: Int): pass

  fn __pow__(self, exp: Int, mod: Int): pass

  # expected-error @+1 {{'__pow__' requires at least 2 operands}}
  fn __pow__(self): pass

  # expected-error @+1 {{'__pow__' requires at most 3 operands}}
  fn __pow__(self, exp: Int, mod: Int, extra: Int): pass

# Issue #6587: [Lit] Recursive constructors crash kgen
struct BadInit[size: __mlir_type.index]:
  @implicit
  fn __init__(out self, elem: BadInit[Int(1)._mlir_value]):
    var x : __mlir_type[`!pop.simd<`, Self.size, `, FloatDyn>`]
    # expected-error @+1 {{cannot implicitly convert '__mlir_type.`!pop.simd<size, FloatDyn>`' value to 'BadInit[size]'}}
    self = x

struct MLIRAttrWithinStruct:
  # expected-error @below {{expressions are not supported in struct bodies}}
  __mlir_attr.`#index<cmp_predicate eq>`


# In register structs may only have stored properties of other in-reg values.
struct InMemStruct: pass

# expected-error @+2 {{all members of '@register_passable' struct must themselves be '@register_passable'}}
@register_passable
struct InRegStruct:
  var x: Int # ok
  # expected-error @+1 {{cannot synthesize __moveinit__ because field 'y' has non-copyable and non-movable type 'InMemStruct'}}
  var y: InMemStruct # expected-note {{'y' declared with type 'InMemStruct'}}

struct OtherInMemStruct:
  var x: Int # ok
  var y: InMemStruct # ok


@register_passable("trivial")
struct InvalidMember:
  var x: __mlir_type.index
  # expected-error @+1 {{'@register_passable' types may not have a '__moveinit__' method, they are always movable by copying a register}}
  fn __moveinit__(out self, deinit existing: Self): pass
  # expected-error @+1 {{trivial types may not have a '__copyinit__' method, they are always trivially copyable}}
  fn __copyinit__(out self, existing: Self): pass
  # expected-error @+1 {{trivial types may not have a '__del__' method, they are always trivially destroyable}}
  fn __del__(deinit self): pass

def noop():  # expected-error {{expected body statements; use 'pass' if none is required}}

struct BadDtor1:
  fn __del__(self): # expected-error {{'self' argument must be passed as 'deinit'}}
    pass

  # expected-error @+1 {{only arguments of Self type may be marked 'deinit'}}
  fn bad1(self, deinit x: Int): pass
  # expected-error @+1 {{deinit arguments may not be variadic}}
  fn bad2(deinit *self): pass

# expected-error @+1 {{only struct methods may be 'deinit'}}
fn invalid_deinit(deinit self: Int):
    pass

struct GoodDtor:
   fn __del__(deinit self): pass
   fn explicit_dtor(deinit self): pass
   fn explicit_dtor2(deinit self, deinit other: Self): pass
   fn normal_var(var self): pass

struct GoodDtor2[A: Int]:
   fn explicit_dtor(deinit self, deinit other: GoodDtor2[0]): pass

fn test_deinit_fn_types():
  var fp1 : fn(var self: GoodDtor) -> None
  fp1 = GoodDtor.__del__
  fp1 = GoodDtor.explicit_dtor
  fp1 = GoodDtor.normal_var

  # expected-error @+1 {{'deinit' is not supported in function types, use 'var' instead}}
  var fp2 : fn(deinit self: GoodDtor) -> None

@fieldwise_init
struct CantSynthesize(ImplicitlyCopyable, Movable):
# expected-error @below {{cannot synthesize fieldwise init because field 'x' has non-copyable and non-movable type 'InMemStruct'}}
# expected-error @below {{cannot synthesize __moveinit__ because field 'x' has non-copyable and non-movable type 'InMemStruct'}}
# expected-error @below {{cannot synthesize __copyinit__ because field 'x' has non-copyable type 'InMemStruct'}}
  var x : InMemStruct


@fieldwise_init
struct ResolveErrorIsBubbled:
   var x: Int
   @implicit
   fn __init__(out self, x: unknown): # expected-error {{use of unknown declaration 'unknown'}}
      pass

fn function_with_struct():
  struct Foo: # expected-error {{struct inside a function not supported here}}
    var x: Int

# https://github.com/modularml/modular/issues/12598
struct not_nested_struct[*Ts: AnyType]:
    @implicit
    fn __init__(out self, *args: *Self.Ts):
        pass
fn function_with_struct2():
    var s1 = not_nested_struct()  # ok
    struct S2[*Ts: AnyType]: # expected-error {{struct inside a function not supported here}}
        @implicit
        fn __init__(out self, *args: *Ts):
            pass
    var s2 = S2() # In issue https://github.com/modularml/modular/issues/12598 this was crashing.

# https://github.com/modularml/modular/issues/33557
struct HasBadCtor:
    var v: Int
    # expected-error @below {{function cannot have both an 'out' argument and an explicit result type}}
    fn __init__(out self, v: Int) -> Self:
        self.v = v
def useBadCtor():
    # Note that the key thing we're checking for here is that this does NOT have
    # a spurious error about HasBadCtor not being constructable from IntLiteral
    var fromBadCtor = HasBadCtor(123)

struct NotRegisterPassable:
    fn __init__(out self):
        pass

# https://github.com/modularml/modular/issues/34551
# Don't crash on emitting methods when the struct itself is erroneous.

@fieldwise_init
@register_passable
struct Outer34551(ImplicitlyCopyable, Movable): # expected-error {{all members of '@register_passable' struct must themselves be '@register_passable'}}
    # expected-error @below {{cannot synthesize __moveinit__ because field '_inner' has non-copyable and non-movable type 'NotRegisterPassable'}}
    # expected-error @below {{cannot synthesize __copyinit__ because field '_inner' has non-copyable type 'NotRegisterPassable'}}
    # expected-note @below {{'_inner' declared with type 'NotRegisterPassable'}}
    var _inner: NotRegisterPassable
    fn __init__(out self):
        self._inner = NotRegisterPassable()
    # The key point of this test is that these errors break an invariant needed
    # for emission, so previously it would crash while emitting this __del__.
    fn __del__(deinit self):
        _ = self._inner ^

@register_passable
struct StructWithoutBody:
    pass

@fieldwise_init
@register_passable
struct OkayStruct(ImplicitlyCopyable):
# expected-error @below {{cannot synthesize __copyinit__ because field 'begin' has non-copyable type 'StructWithoutBody'}}
    var begin: StructWithoutBody


@fieldwise_init
@register_passable
struct ExplicitlyCopyableStructWithNonCopyableBody(Copyable):
# expected-error @below {{cannot synthesize __copyinit__ because field 'begin' has non-copyable type 'StructWithoutBody'}}
    var begin: StructWithoutBody


@register_passable
struct ExplicitlyCopyableStructWithoutBody(Copyable):
    pass

@fieldwise_init
@register_passable
struct ImplicitCopyableStructWithExplicitBody(ImplicitlyCopyable):
  # expected-error @below {{cannot synthesize __copyinit__ because field 'begin' has non-copyable type 'ExplicitlyCopyableStructWithoutBody'}}
    var begin: ExplicitlyCopyableStructWithoutBody


# MOCO-2186: Initializer syntax should reject incorrect result type
struct StructWithSpecificInit[X: Int]:
    fn __init__(out self: StructWithSpecificInit[4]): # expected-note {{function declared here}}
        pass
def testStructWithSpecificInit():
    # expected-error @+1 {{invalid initialization: return type 'StructWithSpecificInit[4]' parameter 'X' doesn't match expected value '1'}}
    var a = StructWithSpecificInit[1]()  # Infers to A[4]

    # This is ok.
    var b = StructWithSpecificInit[4]()


##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

# MOCO-2391
# expected-error @+1 {{use of unknown declaration 'UnknownTrait'}}
struct StructWithUnknownTrait(UnknownTrait):
    pass


trait EverythingIsWrongTrait:
    var value: Int # expected-error {{fields in traits are not supported yet}}

    fn trait_fn_no_dot_dot_dot(self): # expected-error {{expected body statements; use 'pass' if none is required}}

    trait NestedTrait: # expected-error {{nested trait not supported here}}
        ...

    # expected-note @below {{function declared here}}
    fn parametric[x: Int](self): ...

    struct NestedStruct: # expected-error {{nested struct in a trait not supported here}}
        pass

trait TraitWithParams[T: AnyTrivialRegType]: # expected-error {{TODO: trait declarations do not support parameters yet}}
    ...

fn bad_trait_params[T: EverythingIsWrongTrait](x: T):
  # expected-error @below {{invalid call to 'parametric': failed to infer parameter 'x', it isn't used in any argument}}
  x.parametric()

trait Shape(ImplicitlyCopyable, Movable):
	fn area(self) -> Int:
	    ...

@fieldwise_init
struct ShapeContainer:
    var shape: Shape # expected-error {{dynamic traits not supported yet, please use a compile time generic instead of 'Shape'}}

##===----------------------------------------------------------------------===##
# Struct/Trait conformance check failure
##===----------------------------------------------------------------------===##

trait CFMTrait: # expected-note {{trait 'CFMTrait' declared here}}
    # expected-note @below {{no 'f1' candidates have type 'fn(self: CFMStructFail) -> None'}}
    fn f1(self):
        ...

    @staticmethod
    fn f2(): # expected-note {{required function 'f2' is not implemented}}
        ...

# struct implements CFMTrait but does not have f2().
@register_passable("trivial")
struct CFMStructFail(CFMTrait): # expected-error {{'CFMStructFail' does not implement all requirements for 'CFMTrait'}}
  fn f1(self, x: Int): # expected-note {{candidate declared here with type 'fn(self: CFMStructFail, x: Int) -> None'}}
    pass

@register_passable("trivial")
struct NoTraits:
    pass

fn trait_fn[T: CFMTrait]():
    pass

fn invalid_trait_bind():
    trait_fn[NoTraits]() # expected-error {{cannot bind type 'NoTraits' to trait 'CFMTrait'}}

fn non_copyable_trait[T: CFMTrait](value: T):
    var copy = value # expected-error {{value of type 'T' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}


fn trait_fn_infer[T: CFMTrait](x: T):
    pass

fn dont_crash_pvalue_convert(x: CFMStructFail):
    # This will succeed, the error will be raised when resolving `CFMStructFail`.
    trait_fn_infer(x)

trait GrandFather: # expected-note {{trait 'GrandFather' declared here}}
    fn foo(self): # expected-note {{required function 'foo' is not implemented}}
        ...

trait Father(GrandFather): # expected-note {{inherited through 'Father' here}}
    pass

# expected-error @below {{'MissingInheritedFn' does not implement all requirements for 'GrandFather'}}
struct MissingInheritedFn(Father, GrandFather):
    pass

struct InheritsTwice(Father, Father):
    fn foo(self):
        pass


# https://github.com/modular/mojo/issues/1399
# Parser crash when trait implementation parameters don't match the definition
# expected-note @below {{trait 'TraitWithIntParamOnMethod' declared here}}
trait TraitWithIntParamOnMethod:
  # expected-note @below {{no 'f' candidates have type 'fn[n: Int](self: UseTraitWithIntParamOnMethod) -> None'}}
  fn f[n: Int](self):
    ...
# expected-error @below {{'UseTraitWithIntParamOnMethod' does not implement all requirements for 'TraitWithIntParamOnMethod'}}
struct UseTraitWithIntParamOnMethod(TraitWithIntParamOnMethod):
  # expected-note @below {{candidate declared here with type 'fn[n: Bool](self: UseTraitWithIntParamOnMethod) -> None'}}
  fn f[n: Bool](self):
    pass

##===----------------------------------------------------------------------===##
# Class
##===----------------------------------------------------------------------===##

class SomeClass:  # expected-error {{classes are not supported yet}}
  pass


# Issue #12090
from imported_module import DTypePointer # expected-note {{conflicts with this previous declaration}}
struct DTypePointer: # expected-error {{cannot define a struct here with name 'DTypePointer'}}
    pass

# Issue #13321.
struct copy_init_def:
  var field: Int

  # expected-error @+1 {{cannot define '__copyinit__' as 'def'; 'def' implicitly raises}}
  def __copyinit__(out self, existing: Self):
    self.field = existing.field

struct copy_init_raises:
  # expected-error @+1 {{'__copyinit__' cannot be declared as raising an exception}}
  fn __copyinit__(out self, existing: Self) raises:
     pass


# Order of declaration processing.
# https://github.com/modular/mojo/issues/235
@fieldwise_init
struct Inner:
    pass

@fieldwise_init
@register_passable
struct Outer: # expected-error {{all members of '@register_passable' struct must themselves be '@register_passable'}}
    # expected-error @+1{{cannot synthesize __moveinit__ because field 'inner' has non-copyable and non-movable type 'Inner'}}
    var inner: Inner # expected-note {{'inner' declared with type 'Inner'}}


@fieldwise_init
struct AnyTypeMember[T: AnyType](ImplicitlyCopyable, Movable):
# expected-error @below {{cannot synthesize fieldwise init because field 'value' has non-copyable and non-movable type 'T'}}
# expected-error @below {{cannot synthesize __moveinit__ because field 'value' has non-copyable and non-movable type 'T'}}
# expected-error @below {{cannot synthesize __copyinit__ because field 'value' has non-copyable type 'T'}}
    var value: Self.T


# Issue https://github.com/modular/mojo/issues/1675
# Ensure @fieldwise_init fails gracefully in the presence of duplicate field names.
@fieldwise_init
struct BadStruct:
    var b: Int  # expected-note {{previous definition here}}
    var b: Int  # expected-error {{invalid redefinition of 'b'}}


# Also ensure that @fieldwise_init doesn't fail if a method/alias shadows it.
@fieldwise_init
struct OtherBadStruct:
    # expected-note @below {{previous definition here}}
    # expected-note @below {{cannot overload with this non-function definition}}
    var b: Int
    comptime b = 0  # expected-error {{invalid redefinition of 'b'}}

    fn b(mut self):  # expected-error {{invalid redefinition of 'b'}}
        pass


fn test_bad_struct():
    _ = BadStruct(1)
    _ = OtherBadStruct(2)

##===----------------------------------------------------------------------===##
# Bad implicit conversions.
##===----------------------------------------------------------------------===##


@fieldwise_init
struct HasBoolParam[a: Bool]:
   pass

fn test(arg: HasBoolParam[True]):
  # expected-error @+1 {{cannot implicitly convert 'HasBoolParam[True]' value to 'HasBoolParam[False]'}}
  var bad : HasBoolParam[False] = arg


@fieldwise_init
struct Foo(ImplicitlyCopyable, Movable):
    var val: Int

@fieldwise_init
struct ContainsFoo:
    var foo: Foo

# expected-note @+1 {{function declared here}}
fn take_foo(x: Foo): pass

fn return_foo(x: Int) -> Foo:
    return x # expected-error {{cannot implicitly convert 'Int' value to 'Foo'}}

    return 1.2 # expected-error {{cannot implicitly convert 'FloatLiteral[1.2]' value to 'Foo'}}

# When attempting to do implicit conversions without an @implicit decorator
fn implicit_conversions():
    # assigning to expected type
    var x = 42
    var a: Foo = x # expected-error {{cannot implicitly convert 'Int' value to 'Foo'}}

    # # reassigning
    var b = Foo(42)
    b = 42 # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Foo'}}

    # # assigning to uninitialized
    var c: Foo
    c = 42 # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Foo'}}

    # # assigning to member
    var d = ContainsFoo(Foo(24))
    d.foo = 42 # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Foo'}}

    # # returning conversions
    var e = return_foo(42)

    take_foo(24) # expected-error {{invalid call to 'take_foo': argument #0 cannot be converted from 'IntLiteral[24]' to 'Foo'}}

##===----------------------------------------------------------------------===##
# Top Level Code
##===----------------------------------------------------------------------===##

fn top_level_func() raises -> Int:
   pass

fn use_error(e: Error):
   pass

# expected-error @below {{expressions are not supported at the file scope}}
_ = top_level_func()

# expected-error @below {{'try' must be contained in a function}}
try:
    pass
except e:
    # expected-error @below {{expressions are not supported at the file scope}}
    use_error(e)


fn top_level_func_param[p: Int]():
    pass

comptime a = 100
# expected-error @below {{expressions are not supported at the file scope}}
top_level_func_param[a]()

# expected-error @below {{global vars are not supported}}
var globalVar = 1


struct S[param: Int]: #expected-note {{previous definition here}}
  fn method[param: Int](self): # expected-error {{invalid redefinition of 'param'}}
    pass

struct MyParam[p: Int]:
  pass

#expected-note @below {{previous definition here}}
struct MyStruct[p: Int, m1: MyParam[_], m2: MyParam[_]]:
  fn method[p: Int](self): # expected-error {{invalid redefinition of 'p'}}
    pass

# https://github.com/modular/modular/issues/5479
fn __del__(): # expected-error {{'__del__' must be a method, not a global function}}
  pass

fn raises_int() raises Int:
  raise 1

  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral[4]' value to 'Int'}}
  raise 4.0

  # expected-error @+1 {{cannot implicitly convert 'Error' value to 'Int'}}
  raise Error()

fn bad_raises_fn2() raises:
  # expected-error @+1 {{cannot call function that may raise 'Int' in context that supports an error type of 'Error'}}
  raises_int()

  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral[4]' value to 'Error'}}
  raise 4.0

  try:
    raises_int()

    # expected-error @+1 {{cannot implicitly convert 'Error' value to 'Int'}}
    raise Error()
  except e: # 'e' inferred to Int.
    var x: Int = e
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'Error'}}
    var y: Error = e

  try:
    raise 1 # Should infer error to Int, not IntLiteral
  except e:
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'String'}}
    var x: String = e
