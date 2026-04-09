# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


# expected-error @below {{expected 'main' function to have no arguments}}
def main(arg: Int):
    return


# // -----


# expected-error @below {{expected 'main' function to have no arguments}}
def main(arg: Int) raises:
    return


# // -----


# expected-error @below {{expected 'main' function to return 'None'}}
def main() -> Int:
    return 10


# // -----


# expected-error @below {{expected 'main' function to have no parameters}}
def main[input: Int]():
    return


# // -----


# expected-error @below {{'main' can only be exported as 'main'}}
@export("foo")
def main():
    return


# // -----


# expected-error @below {{only 'main' can be exported as 'main'}}
@export("main")
def fooMain():
    return
