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
