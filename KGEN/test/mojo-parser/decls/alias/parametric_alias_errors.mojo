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
