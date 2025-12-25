# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"fn() escaping -> None"


fn no_capture():
    fn closure() escaping:
        pass
