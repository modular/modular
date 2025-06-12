# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: asan

# RUN: %mojo %s 2>&1 | FileCheck %s

# CHECK: warning: Global variables are only partially implemented in Mojo, only the most simple cases work. Using globals is not recommended. To silence this warning, start the variable name with '__'.
var globalVar: Int32 = 1


fn main():
    globalVar += 1
