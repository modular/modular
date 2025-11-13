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
alias unknownDecl[x: Int] = y

# expected-error @below {{cannot implicitly convert 'Int' value to 'String'}}
alias wrongType[x: Int]: String = x

alias myIntAdd[x: Int, y: Int] = x + y

alias myCurriedIntAdd[x: Int] = myIntAdd[x]

# expected-error @below {{unknown keyword parameter: 'y'}}
alias myCurriedIntAdd2 = myCurriedIntAdd[y=2]

alias myRenamedCurriedIntAdd[a: Int] = myCurriedIntAdd[a]

# expected-error @below {{unknown keyword parameter: 'x'}}
alias myRenamedCurriedIntAdd2 = myRenamedCurriedIntAdd[x=2]

# expected-error @below {{'Int' is not subscriptable}}
alias mySix = myCurriedIntAdd[2][4][6]

# expected-error @below {{parametric value expects 2 parameters, but 3 were specified}}
alias myIntAddTooManyParams = myIntAdd[1, 2, 3]

# COM: A type with dependent parameters.
struct Dep[T: AnyType, v: T]:
    pass

alias MyDep[T: AnyType, v: T] = Dep[T, v]

# expected-error @below {{'MyDep' needs more parameters bound before accessing attributes}}
# expected-note @below {{'MyDep' is aka 'alias[T: AnyType, v: T] Dep[T, v]'}}
alias MyDepGetAlias0 = MyDep.hello

# expected-error @below {{'alias[v: Int] Dep[Int, v]' needs more parameters bound before accessing attributes}}
alias MyDepGetAlias1 = MyDep[Int].hello

# expected-error @below {{'Dep[Int, 2]' value has no attribute 'hello'}}
alias MyDepGetAlias2 = MyDep[Int, 2].hello
