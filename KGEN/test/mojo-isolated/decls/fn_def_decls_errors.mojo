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

fn implicit_var_decl(a: Index):
    c = a  # implicit declaration of c

# expected-error @+1 {{'__add__' requires 2 operands}}
fn __add__():
    pass

# expected-error @+1 {{'__sub__' must be a method}}
fn __sub__(self: Index, a: Index):
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
fn missing_argument_name(*: Index): pass

# expected-error @below {{expected parameter name}}
fn missing_parameter_name[: Index](): pass

# expected-error @+1 {{use of unknown declaration 'InvalidType'}}
fn test_unknown_arg_type(a: InvalidType):
    _ = a.value  # Should not produce a follow-on error.
    return

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_stars(a: Index, *, *, b: Index):
    pass

# expected-error @+1 {{cannot have two '/' markers in the same argument list}}
fn two_slashes(a: Index, /, /, b: Index):
    pass

# expected-error @+1 {{cannot specify '/' marker after '*' marker}}
fn slash_after_start(a: Index, *, /, b: Index):
    pass

# expected-error @+1 {{'/' marker cannot be used at the start of the argument list}}
fn leading_slash(/, a: Index):
    pass

# expected-error @+1 {{'*' marker is not allowed at end of argument list}}
fn trailing_star(a: Index, *):
    pass

# # expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_variadics(*a: Index, *b: Index):
    pass

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
fn two_variadic_packs[*Ts: AnyTrivialRegType](*a: *Ts, *b: *Ts):
    pass

# expected-error @+1 {{parametric functions may not be used as arguments; consider passing as a parameter instead}}
fn foo(x: fn[a: Index] () -> None):
    pass

# expected-error @below {{non-owned variadic keyword arguments are not supported yet}}
fn borrowed_kwargs(read **kwargs: Index):
    pass

# expected-error @below {{'//' marker cannot be used at the start of the parameter list}}
fn invalid_inferred[//, x: Index]():
    pass

# expected-error @below {{cannot specify '//' marker after '*' marker in parameter list}}
fn invalid_inferred_kw_only[*, x: Index, //, y: Index]():
    pass

# expected-error @below {{'//' can only be used in parameter lists to denote inferred parameters}}
fn invalid_inferred_argument(x: Index, //):
    pass

# expected-error @below {{inferred parameters may not have defaults}}
fn invalid_inferred_default[x: Index = `1`, //]():
    pass

struct NonCopyable:
    fn __init__(out self):
       pass

def defTests() -> None:
  alias abc = 1

  # MOCO-83: [mojo][Bug] def methods can't shadow names via assignment
  # expected-error @+1 {{expression must be mutable in assignment}}
  abc = 4

# expected-error @+1 {{value of type 'IntLiteral[4]' doesn't have a memory origin in origin specifier}}
fn ref_result_invalid1() -> ref [4] MemoryType:
    pass

fn ref_result_invalid2(mut a: MemoryType) -> ref [a] Int:
    # expected-error @+1 {{value of type 'Int' doesn't have a memory origin in return value}}
    return 4

fn ref_result_invalid3(mut a: MemoryType, mut b: MemoryType)
     -> ref [a] MemoryType:
    # expected-error @+1 {{cannot return reference with incompatible origin: 'b' vs 'a'}}
    return b

fn ref_result_invalid4(mut a: MemoryType, b: Int) -> ref [a] MemoryType:
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'MemoryType'}}
    return b

# expected-error @+1 {{cannot return 'a's origin, because it might expand to a @register_passable type}}
fn ref_result_invalid5[T: AnyType](a: T) -> ref [a] T:
    return a

# expected-error @+1 {{cannot return 'b's origin, because it has @register_passable type 'Int'}}
fn ref_result_invalid6(mut a: MemoryType, mut b: Int) -> ref [b] Int:
    pass

# expected-error @+1 {{'ref' result requires an origin specifier}}
fn ref_result_invalid7() -> ref MemoryType:
    pass

# expected-error @+1 {{value of type 'Int' doesn't have a memory origin in origin specifier}}
fn ref_result_invalid8(a: Int) -> ref [a] MemoryType:
    pass

# expected-error @+1 {{cannot infer origin for a function result}}
fn ref_result_invalid9() -> ref [_] MemoryType:
    pass

fn valid_ref_result(x: MemoryType) -> ref [x] MemoryType: return x

fn ref_invalid():
    var a = MemoryType()
    # expected-error @+1 {{expression must be mutable in assignment}}
    valid_ref_result(a) = MemoryType()

fn return_ref_type_error(a: fn (x: MemoryType) -> ref [x] MemoryType):
    # expected-error @+1 {{cannot implicitly convert 'fn(x: MemoryType) -> ref [*[0,0]] MemoryType' value to 'Int'}}
    var b: Int = a

@register_passable
struct SBValue:
    pass

# expected-error @below {{TODO: read-only non-trivial register-passable arguments are not yet supported in async functions}}
async fn invalid_sb_value(value: SBValue):
    pass

# expected-error @below {{TODO: read-only non-trivial register-passable arguments are not yet supported in async functions}}
async fn invalid_sb_value_variadic(*value: SBValue):
    pass

async fn borrowed_generic_arg[T: AnyType](value: T):
    pass

fn valid_sbvalue_borrow(value: SBValue):
    _ = borrowed_generic_arg(value)

# expected-error @+1 {{multiple specification of address space isn't valid}}
fn bad_ref_as[a: AddressSpace](ref [a, a] x: Int):
    pass
