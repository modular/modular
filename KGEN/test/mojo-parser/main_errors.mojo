# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


# expected-error @below {{expected 'main' function to have no arguments}}
fn main(arg: Int):
    return

# // -----

# expected-error @below {{expected 'main' function to have no arguments}}
def main(arg: Int):
    return

# // -----

# expected-error @below {{expected 'main' function to return 'None'}}
fn main() -> Int:
    return 10

# // -----

# expected-error @below {{expected 'main' function to have no parameters}}
fn main[input: Int]():
    return

# // -----

# expected-error @below {{'main' can only be exported as 'main'}}
@export("foo")
fn main():
    return

# // -----

# expected-error @below {{only 'main' can be exported as 'main'}}
@export("main")
fn fooMain():
    return
