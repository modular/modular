# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package %S/inputs/target_dep_package -o %T/target_dep_package.mojopkg
# RUN: mojo package %S/inputs/test_package -o %T/test_package.mojopkg
# RUN: kgen -I %T -gen-lib %s -O0 -S -o %t.0.mlir
# RUN: cat %t.0.mlir | FileCheck %s --check-prefix=GENLIB
# RUN: kgen-opt %t.0.mlir -materialize-packages -o %t.1.mlir
# RUN: cat %t.1.mlir | FileCheck %s --check-prefix=MATER
# RUN: kgen-opt %t.1.mlir -elaborate-generators -o %t.2.mlir
# RUN: cat %t.2.mlir | FileCheck %s --check-prefix=ELAB
# RUN: kgen-opt %t.2.mlir -externalize-precompiled-functions -o %t.3.mlir
# RUN: cat %t.3.mlir | FileCheck %s --check-prefix=LAST

from target_dep_package.module import target_dep_function
from test_package.module import exported_func

# COM: This integration test tests every important step for packaging and
# COM: precompilation.

# ELAB: kgen.link dense_resource<{{.*}} as [[PKG:@.*target_dep_package.*]]
# LAST: kgen.link dense_resource<{{.*}} as [[PKG:@.*target_dep_package.*]]


# GENLIB: kgen.generator export @top
# GENLIB: call @exported_func()
# GENLIB: call [[TARGET_FN:@.*target_dep_function.*]]() : ()
# MATER: kgen.generator export @top
# MATER: call @exported_func()
# MATER: call [[TARGET_FN:@.*target_dep_function.*]]() : ()
# ELAB: kgen.func export @top
# ELAB: call @exported_func()
# ELAB: call [[TARGET_FN:@.*target_dep_function.*]]() : ()
# LAST: kgen.func export @top
# LAST: call @exported_func()
# LAST: call [[TARGET_FN:@.*target_dep_function.*]]() : ()
@export
fn top() -> Int:
    exported_func()
    return target_dep_function()


# GENLIB: kgen.extern.generator [[TARGET_FN]]
# GENLIB-SAME: preCompiledModuleRef = [[PKG:@.*target_dep_package.*]]}
# MATER: kgen.generator [[TARGET_FN]]
# MATER-SAME: preCompiledModuleRef = [[PKG:@.*target_dep_package.*]],
# MATER-NEXT: kgen.param.if
# ELAB: kgen.func export package [[TARGET_FN]]
# ELAB-SAME: precompiledBodyRef = [[PKG]]
# LAST: kgen.extern.func export package [[TARGET_FN]]
# LAST-SAME: from [[PKG]]

# GENLIB: kgen.package.link [[PKG]] pre_elaboration(dense_resource<{{.*}}) archives()
# MATER: kgen.package.link [[PKG]] pre_elaboration(dense_resource<{{.*}}) archives()

# GENLIB: kgen.extern.generator export @exported_func()
# MATER: kgen.generator @exported_func()
# ELAB: kgen.func export package @exported_func()
# LAST: kgen.extern.func export package @exported_func()
