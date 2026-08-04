# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Renaming a precompiled package rewrites only the references it roots. A
# dependency may nest a module under the same name as the package's source
# directory, and a reference reaching through that module roots at the
# dependency: the shared name is a component there, and stays. This holds
# for symbol references and for the rendered text inside debug-info source
# names, where a reference's components are substrings.

# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: mojo precompile %S/inputs/nested_name_clash/other -o %t.dir/other.mojoc
# RUN: mojo precompile -I %t.dir %S/inputs/nested_name_clash/pkg \
# RUN:   -o %t.dir/renamed_pkg.mojoc
# RUN: kgen-opt --mlir-print-debuginfo %t.dir/renamed_pkg.mojoc \
# RUN:   | FileCheck %s --check-prefix=IR \
# RUN:     --implicit-check-not '@other::@renamed_pkg'
# RUN: mojo run -I %t.dir %s | FileCheck %s

# IR-DAG: !lit.struct<@other::@pkg::@m::@Thing>
# IR-DAG: trait<@other::@pkg::@m::@Nameable

# CHECK: 42

from renamed_pkg.use import make


def main():
    print(make())
