# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A renamed precompiled package's debug-info source names root at the output
# name: the symbol lineage snapshotted into scoped locations at parse time
# does not keep the source directory's name, and neither do the rendered
# type strings inside source names (where package references are text, not
# symbol references).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "trait Nameable:" > %t.dir/pkg/t.mojo
# RUN: echo "    def name(self) -> Int:" >> %t.dir/pkg/t.mojo
# RUN: echo "        ..." >> %t.dir/pkg/t.mojo
# RUN: echo "" >> %t.dir/pkg/t.mojo
# RUN: echo "struct Holder[T: Nameable]:" >> %t.dir/pkg/t.mojo
# RUN: echo "    def get(self) -> Int:" >> %t.dir/pkg/t.mojo
# RUN: echo "        return 0" >> %t.dir/pkg/t.mojo
# RUN: mojo precompile %t.dir/pkg -o %t.dir/renamed_pkg.mojoc
# RUN: kgen-opt --mlir-print-debuginfo %t.dir/renamed_pkg.mojoc \
# RUN:   | FileCheck %s --implicit-check-not '(pkg)"pkg"' --implicit-check-not '@pkg::'

# CHECK-DAG: (module)"a" from <(pkg)"renamed_pkg">
# CHECK-DAG: (struct)"Holder"[<"trait<@renamed_pkg::@t::@Nameable
