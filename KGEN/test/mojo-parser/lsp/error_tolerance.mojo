# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# This shows the defining behavior of the language-server parse path: it is
# error-tolerant. In `-lsp` mode the parser reports the diagnostic AND still
# emits a partial module for the rest of the file, whereas the regular compiler
# path bails out to a null module on the same error.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -lsp %s 2>&1 | FileCheck %s
# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s --check-prefix=NOLSP

struct Good(Movable where False):
    var value: Int


def uses_undefined():
    var x = this_name_does_not_exist

# The error is reported in both modes.
# CHECK: error: use of unknown declaration 'this_name_does_not_exist'
# NOLSP: error: use of unknown declaration 'this_name_does_not_exist'

# But only the LSP path still produces IR for the rest of the file.
# CHECK: lit.struct.decl @Good
# NOLSP-NOT: lit.struct.decl
