# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


fn make_worldly(inout *strs: String):
    for i in range(len(strs)):
        strs[i] += " world"


fn main():
    # CHECK: -- Testing inout varargs
    print("-- Testing inout varargs")
    var s1: String = "hello"
    var s2: String = "konnichiwa"
    var s3: String = "bonjour"
    make_worldly(s1, s2, s3)
    print(s1)  # CHECK: hello world
    print(s2)  # CHECK-NEXT: konnichiwa world
    print(s3)  # CHECK-NEXT: bonjour world
