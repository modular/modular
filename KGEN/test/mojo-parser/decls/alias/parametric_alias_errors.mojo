# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# declarations
##===----------------------------------------------------------------------===##

# expected-error @below {{use of unknown declaration 'y'}}
comptime unknownDecl[x: Int] = y

# expected-error @below {{cannot implicitly convert 'Int' value to 'String'}}
comptime wrongType[x: Int]: String = x

comptime myIntAdd[x: Int, y: Int] = x + y

# expected-warning @+2 {{'where' clauses inside parameter lists are deprecated}}
# expected-note @+1 {{use a trailing 'where' clause after the signature instead}}
comptime deprecatedInlineWhere[x: Int where x > 0] = x

# Trailing 'where' clauses are only allowed when the alias is parameterized.
# expected-error @+1 {{trailing 'where' clauses on non-parameterized aliases support TBD}}
comptime typedNonparametricWhere: Int where True = 1

# An empty `[]` parameter list still declares zero parameters, so trailing
# 'where' is rejected just like in the un-bracketed case.
# expected-error @+1 {{trailing 'where' clauses on non-parameterized aliases support TBD}}
comptime emptyParamWhere[] where True = 1

def implicit_generator_constraint_drop_error[cond: Bool]():
    comptime constrained[x: Int]: AnyType where cond = Int
    comptime dropped: __mlir_type[
        `!lit.generator<<"x":`, +Int, `>`, +AnyType, `>`
    # expected-error @below {{cannot implicitly convert '[x: Int] AnyType where cond' value to '[x: Int] AnyType' in comptime initializer}}
    ] = constrained

def implicit_generator_constraint_drop_env_mismatch_error[
    cond: Bool, other: Bool
]() where other:
    comptime constrained[x: Int]: AnyType where cond = Int
    comptime dropped: __mlir_type[
        `!lit.generator<<"x":`, +Int, `>`, +AnyType, `>`
    # expected-error @below {{cannot implicitly convert '[x: Int] AnyType where cond' value to '[x: Int] AnyType' in comptime initializer}}
    ] = constrained

# Discharging a generator body constraint depends on the assumptions in scope,
# so its convertibility result must never be cached: the convertibility cache is
# keyed only on the (value, required) type pair. The two scopes above failed to
# discharge `where cond` for the `[x: Int] AnyType` generator/body pair; this
# scope proves `cond` and must still succeed for the *same* type pair. If a
# scope-dependent result were cached, the failures above would poison this query
# and produce a spurious "cannot implicitly convert" diagnostic here.
def implicit_generator_constraint_drop_cross_scope[cond: Bool]() where cond:
    comptime constrained[x: Int]: AnyType where cond = Int
    comptime dropped: __mlir_type[
        `!lit.generator<<"x":`, +Int, `>`, +AnyType, `>`
    ] = constrained


# Only *fully* dropping body constraints is supported for now. Dropping a strict
# subset (keeping `condB`, dropping `condA`) is rejected even though `condA` is
# provable here: a partial drop would leave `condB` with a mismatched source
# location, requiring a `rebind` that `bind_params` folding cannot yet look
# through. TODO: allow partial dropping once folding handles the rebind.
def implicit_generator_constraint_drop_partial[
    condA: Bool, condB: Bool
]() where condA where condB:
    comptime keepsB[x: Int]: AnyType where condB = Int
    comptime bothConstraints[x: Int]: AnyType where condA where condB = Int
    # expected-error @below {{cannot implicitly convert '[x: Int] AnyType where condA, condB' value to '[x: Int] AnyType where condB' in comptime initializer}}
    comptime r: type_of(keepsB) = bothConstraints


# Adding a body constraint the source lacks is likewise rejected (it is a form of
# partial relaxing -- `expected` retains constraints, so it is not a full drop).
def implicit_generator_constraint_add[condA: Bool, condB: Bool]() where condA where condB:
    comptime onlyA[x: Int]: AnyType where condA = Int
    comptime bothConstraints[x: Int]: AnyType where condA where condB = Int
    # expected-error @below {{cannot implicitly convert '[x: Int] AnyType where condA' value to '[x: Int] AnyType where condA, condB' in comptime initializer}}
    comptime r: type_of(bothConstraints) = onlyA

comptime myCurriedIntAdd[x: Int] = myIntAdd[x, ...]

# expected-error @below {{unknown keyword parameter: 'y'}}
comptime myCurriedIntAdd2 = myCurriedIntAdd[y=2]

comptime myRenamedCurriedIntAdd[a: Int] = myCurriedIntAdd[a]

# expected-error @below {{unknown keyword parameter: 'x'}}
comptime myRenamedCurriedIntAdd2 = myRenamedCurriedIntAdd[x=2]

# expected-error @below {{'Int' is not subscriptable}}
comptime mySix = myCurriedIntAdd[2][4][6]

# expected-error @below {{parametric value expects 2 positional parameters, but 3 were specified}}
comptime myIntAddTooManyParams = myIntAdd[1, 2, 3]


# COM: A type with dependent parameters.
struct Dep[T: AnyType, v: T]:
    pass


comptime MyDep[T: AnyType, v: T] = Dep[T, v]

# expected-error @below {{'T' refers to an unbound parameter in 'MyDep'}}
# expected-note @below {{'MyDep' is aka 'comptime[T: AnyType, v: T] Dep[T, v]'}}
comptime MyDepDotT = MyDep.T

# expected-error @below {{'Dep[_, _]' value has no attribute 'hello'}}
comptime MyDepGetAlias0 = MyDep.hello

# expected-error @below {{'Dep[Int, _]' value has no attribute 'hello'}}
comptime MyDepGetAlias1 = MyDep[Int].hello

# expected-error @below {{'Dep[Int, 2]' value has no attribute 'hello'}}
comptime MyDepGetAlias2 = MyDep[Int, 2].hello


# COM: Using a generator as a struct field type should be rejected (MOCO-3514).
struct FieldWithUnboundAlias:
    # expected-error @below {{'MyDep' is not a concrete type, use '[]' to bind missing parameters}}
    # expected-note @below {{'MyDep' is aka 'comptime[T: AnyType, v: T] Dep[T, v]'}}
    var f: MyDep


def test_variable_type_parameterization():
    # Store an unparameterized struct type in a variable...
    # expected-error @below {{dynamic type values not permitted yet; try creating a 'comptime' instead of a 'var'}}
    var struct_type = Dep

    # .. and try to parameterize it.
    # expected-error @below {{types are not subscriptable}}
    var instance: struct_type[Int]


##===----------------------------------------------------------------------===##
# Trailing 'where' constraints are enforced when the alias's parameters are
# inferred during auto-parameterization, not just for explicit bindings
# (MOCO-4081). This covers every form that auto-parameterizes the alias
# generator: argument types, value-parameter types, and variadic element types.
##===----------------------------------------------------------------------===##


@fieldwise_init
struct Tag[n: Int](Copyable, Movable):
    pass


# A distinct alias is used per form so each violated call's note points at its
# own 'where' clause (identical notes at one location are coalesced by the
# diagnostic verifier).

# Alias used as a function argument type.
# expected-note @+1 {{constraint declared here evaluated to False, expected '(n > 0)'}}
comptime PositiveArg[n: Int, //] where n > 0 = Tag[n]


# expected-note @+1 {{function declared here}}
def take_arg(p: PositiveArg):
    pass


# Alias used as a value-parameter type.
# expected-note @+1 {{constraint declared here evaluated to False, expected '(n > 0)'}}
comptime PositiveParam[n: Int, //] where n > 0 = Tag[n]


# expected-note @+1 {{function declared here}}
def take_param[p: PositiveParam]():
    pass


# Alias used as a variadic argument element type.
# expected-note @+1 {{constraint declared here evaluated to False, expected '(n > 0)'}}
comptime PositiveVar[n: Int, //] where n > 0 = Tag[n]


# expected-note @+1 {{function declared here}}
def take_variadic(*p: PositiveVar):
    pass


def use_ok():
    # Inferred bindings that satisfy the constraint are accepted in every form.
    take_arg(Tag[1]())
    take_param[Tag[1]()]()
    take_variadic(Tag[1](), Tag[1]())


def use_arg_bad():
    # expected-error @+1 {{invalid call to 'take_arg': violated constraint}}
    take_arg(Tag[-1]())


def use_param_bad():
    # expected-error @+1 {{invalid call to 'take_param': violated constraint}}
    take_param[Tag[-1]()]()


def use_variadic_bad():
    # expected-error @+1 {{invalid call to 'take_variadic': violated constraint}}
    take_variadic(Tag[-1]())
