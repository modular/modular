# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package %S/inputs/target_dep_package -o %T/target_dep_package.mojopkg
# RUN: mojo package %S/inputs/test_package -o %T/test_package.mojopkg
# RUN: kgen -I %T -gen-lib %s -O0 -S -o %t.0.mlir
# RUN: cat %t.0.mlir | FileCheck %s --check-prefix=GENLIB

from target_dep_package.module import target_dep_function
from test_package.module import exported_func

# COM: This integration test tests every important step for packaging and
# COM: precompilation.


# GENLIB: kgen.generator export @top
# GENLIB: call @exported_func()
# GENLIB: call [[TARGET_FN:@.*target_dep_function.*]]() : ()
@export
fn top() -> Int:
    exported_func()
    return target_dep_function()


# GENLIB: kgen.generator [[TARGET_FN]]

# GENLIB: kgen.generator export @exported_func()
