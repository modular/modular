# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo package -kgenModule %S/inputs/remove_unused_params -o %T/target.mlirbc
# RUN: kgen-opt %T/target.mlirbc | FileCheck %s

# Test for MOCO-956
# https://linear.app/modularml/issue/MOCO-956/[bug]-segfault-on-struct-recursive-methods

# CHECK: kgen.generator @{{.*}}::FactorialComputer::compute_method{{.*}}_REMOVED_ARG"(%arg0: !pop.scalar<ui8>) -> !pop.scalar<ui8>
# CHECK: kgen.generator @{{.*}}::compute_unusedPost{{.*}}_REMOVED_ARG"(%arg0: !pop.scalar<ui8>) -> !pop.scalar<ui8>
# CHECK: kgen.generator @{{.*}}::compute_unusedPre{{.*}}_REMOVED_ARG"(%arg0: !pop.scalar<ui8>) -> !pop.scalar<ui8>


fn main():
    pass
