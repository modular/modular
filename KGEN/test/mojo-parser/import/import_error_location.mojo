# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %parse-mojo-isolated -split-input-file -I=unknownincludedir -I=%S %s 2>&1 | FileCheck %s

import imported_module.does_not_exist
# CHECK: {{.*}}:24: error: unable to locate module 'does_not_exist'
