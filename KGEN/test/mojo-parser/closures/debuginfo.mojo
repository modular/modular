# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -debug-level full -import-mojo --mojo-disable-builtins -mlir-print-debuginfo | FileCheck %s

# COM: This tests that code generated to support capturing closures is located and scoped correctly.

# CHECK-DAG: #makes_escaping_closure_name = #debuginfo.source_name<(fn)"makes_escaping_closure"(<"index">, <"index">) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP9:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = #makes_escaping_closure_name, linkageName = "makes_escaping_closure{{.*}}", file = #file, line = [[#LN42:]],

# CHECK-DAG:    lit.func @"makes_escaping_closure
# CHECK-DAG:    debuginfo.value #[[VAR0:.*]] = %m : index loc(#[[LOC26:.*]])
# CHECK-DAG:    debuginfo.value #[[VAR1:.*]] = %z : index
# CHECK-DAG:    %anonymous2A = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !escaping
# CHECK-DAG:    %0 = lit.ref.to_pointer %anonymous2A
# CHECK-DAG:    %1 = lit.call {{.*}}CI{{.*}}__init__{{.*}}"(%0, %m)
# CHECK-DAG:    %anonymous2A_0 = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !wrapper
# CHECK-DAG:    %2 = lit.ref.to_pointer %anonymous2A_0
# CHECK-DAG:    %3 = lit.call {{.*}}CW{{.*}}__init__{{.*}}(%2, %0)
# CHECK-DAG:    %4 = lit.call {{.*}}CW{{.*}}__copyinit__{{.*}}(%__result__, %2) {{.*}}

# CHECK-DAG: #[[LOC26]] = loc(fused<#[[SP9]]>[#

alias int = __mlir_type.index

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

fn makes_escaping_closure(m: int, z: int) -> fn (n: int) escaping -> int:
    fn myclosure(n: int) escaping -> int:
        return m

    return myclosure
