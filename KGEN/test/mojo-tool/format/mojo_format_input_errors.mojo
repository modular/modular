# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: system-darwin

# RUN: not mojo format 2>&1 | FileCheck %s --check-prefix=NO_INPUTS
# NO_INPUTS: error: no inputs provided

# RUN: not mojo format foo 2>&1 | FileCheck %s --check-prefix=INVALID_INPUT
# INVALID_INPUT: input 'foo' does not exist

# RUN: echo "" > %t.foobar
# RUN: not mojo format %t.foobar 2>&1 | FileCheck %s --check-prefix=INVALID_FILE_TYPE
# INVALID_FILE_TYPE: invalid input '{{.*}}', expected a source .mojo/.🔥 file, or a directory

# RUN: not mojo format %s - 2>&1 | FileCheck %s --check-prefix=MIX_STDIN
# MIX_STDIN: error: cannot mix '-' with other inputs

# RUN: not mojo format %s --line-length foo 2>&1 | FileCheck %s --check-prefix=LINE_LENGTH
# LINE_LENGTH: error: expected integer value for --line-length, but got 'foo'
