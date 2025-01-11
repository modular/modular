# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

# The following function has a use of the builtin Error type, make sure that
# gets pulled in.

# CHECK: lit.fn @"use_of_error_type
# CHECK: lit.try %__try_error__ : !lit.ref<!Error,

# CHECK: lit.package @builtin
# CHECK: lit.file_module @error
# CHECK: lit.struct.decl @Error


fn use_of_error_type():
    try:
        return
    except:
        pass
