# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics -o /dev/null

@value
struct MemoryType:
    pass

fn test_never_declared_fn():
    # expected-error @+1 {{use of unknown declaration 'never_declared_fn'}}
    never_declared_fn()

fn implicit_var_decl(a: int):
    c = a  # implicit declaration of c

# expected-error @+1 {{'__add__' requires 2 operands}}
fn __add__():
    pass

# expected-error @+1 {{'__sub__' must be a method}}
fn __sub__(self: int, a: int):
    pass

fn missing_colon()  # expected-error {{expected ':' in function definition}}
    # Don't get confused by comments or blank lines!

    var x = `1`

# Missing colon after fn definition complains about function effects
# https://github.com/modularml/modular/issues/23359
# expected-error @+1 {{missing ':' at end of function signature}}
def missing_colon_2()
    test_never_declared_fn()

# expected-error @below {{expected argument name}}
fn missing_argument_name(*: int): pass

# expected-error @below {{expected parameter name}}
fn missing_parameter_name[: int](): pass

# expected-error @+1 {{use of unknown declaration 'InvalidType'}}
fn test_unknown_arg_type(a: InvalidType):
    _ = a.value  # Should not produce a follow-on error.
    return

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_stars(a: int, *, *, b: int):
    pass

# expected-error @+1 {{cannot have two '/' markers in the same argument list}}
fn two_slashes(a: int, /, /, b: int):
    pass

# expected-error @+1 {{cannot specify '/' marker after '*' marker}}
fn slash_after_start(a: int, *, /, b: int):
    pass

# expected-error @+1 {{'/' marker cannot be used at the start of the argument list}}
fn leading_slash(/, a: int):
    pass

# expected-error @+1 {{'*' marker is not allowed at end of argument list}}
fn trailing_star(a: int, *):
    pass

# # expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_variadics(*a: int, *b: int):
    pass

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_variadic_packs[*Ts: AnyTrivialRegType](*a: *Ts, *b: *Ts):
    pass

# expected-error @+1 {{parametric functions may not be used as arguments; consider passing as a parameter instead}}
fn foo(x: fn[a: int] () -> None):
    pass

# expected-error @below {{non-owned variadic keyword arguments are not supported yet}}
fn borrowed_kwargs(borrowed **kwargs: int):
    pass

# expected-error @below {{'//' marker cannot be used at the start of the parameter list}}
fn invalid_inferred[//, x: int]():
    pass

# expected-error @below {{cannot specify '//' marker after '*' marker in parameter list}}
fn invalid_inferred_kw_only[*, x: int, //, y: int]():
    pass

# expected-error @below {{'//' can only be used in parameter lists to denote inferred parameters}}
fn invalid_inferred_argument(x: int, //):
    pass

# expected-error @below {{inferred parameters may not have defaults}}
fn invalid_inferred_default[x: int = `1`, //]():
    pass

struct NonCopyable:
    fn __init__(inout self):
       pass

def test_non_copyable_def_arg(arg: NonCopyable, arg2: int):
    # expected-error @+1 {{'NonCopyable' is not copyable because it has no '__copyinit__'}}
    arg = NonCopyable()

    # OK!
    arg2 = arg2


def defTests() -> None:
  alias abc = 1

  # MOCO-83: [mojo][Bug] def methods can't shadow names via assignment
  # expected-error @+1 {{expression must be mutable in assignment}}
  abc = 4

# expected-error @+1 {{result reference lifetime has unexpected type 'IntLiteral'}}
fn ref_result_invalid() -> ref [4] MemoryType:
    pass

fn ref_result_invalid2(inout a: MemoryType) -> ref [__lifetime_of(a)] Int:
    # expected-error @+1 {{cannot return reference with incompatible lifetime: 'anonymous*' vs 'a'}}
    return 4

fn ref_result_invalid3(inout a: MemoryType, inout b: MemoryType)
     -> ref [__lifetime_of(a)] MemoryType:
    # expected-error @+1 {{cannot return reference with incompatible lifetime: 'b' vs 'a'}}
    return b

fn ref_result_invalid4(inout a: MemoryType, b: Int) -> ref [__lifetime_of(a)] MemoryType:
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'MemoryType'}}
    return b

# expected-error @+1 {{cannot return 'a's lifetime, because it might expand to a @register_passable type}}
fn ref_result_invalid5[T: AnyType](a: T) -> ref [__lifetime_of(a)] T:
    return a

# expected-error @+1 {{cannot return 'b's lifetime, because it has @register_passable type 'Int'}}
fn ref_result_invalid6(inout a: MemoryType, inout b: Int) -> ref [__lifetime_of(b)] Int:
    pass

# expected-error @+1 {{cannot infer lifetime for a function result}}
fn ref_result_invalid7() -> ref [_] MemoryType:
    pass

fn valid_ref_result(x: MemoryType) -> ref [__lifetime_of(x)] MemoryType: return x

fn ref_invalid():
    var a = MemoryType()
    # expected-error @+1 {{expression must be mutable in assignment}}
    valid_ref_result(a) = MemoryType()

fn return_ref_type_error(a: fn (x: MemoryType) -> ref [__lifetime_of(x)] MemoryType):
    # expected-error @+1 {{cannot implicitly convert 'fn(x: MemoryType) -> ref [*[0,0]] MemoryType' value to 'Int'}}
    var b: Int = a

@register_passable
struct SBValue:
    pass

# expected-error @below {{TODO: borrowed non-trivial register-passable arguments are not yet supported in async functions}}
async fn invalid_sb_value(value: SBValue):
    pass

# expected-error @below {{TODO: borrowed non-trivial register-passable arguments are not yet supported in async functions}}
async fn invalid_sb_value_variadic(*value: SBValue):
    pass

async fn borrowed_generic_arg[T: AnyType](value: T):
    pass

fn invalid_sbvalue_borrow(value: SBValue):
    # expected-error @below {{TODO: cannot bind non-trivial register-passable value to borrowed generic argument yet}}
    _ = borrowed_generic_arg(value)
