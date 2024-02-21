# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

# The following function has a use of the builtin Error type, make sure that
# gets pulled in.

# CHECK: lit.func @"use_of_error_type
# CHECK: except (%{{.*}}: !Error)

# CHECK: lit.package @builtin
# CHECK: lit.file_module @error
# CHECK: lit.struct.decl @Error


fn use_of_error_type():
    try:
        return
    except:
        pass
