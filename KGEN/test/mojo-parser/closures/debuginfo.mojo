# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -debug-level full -import-mojo --mojo-disable-builtins -mlir-print-debuginfo | FileCheck %s

# COM: This tests that code generated to support capturing closures is located and scoped correctly.

# CHECK-DAG: #makes_escaping_closure_name = #debuginfo.source_name<(fn)"makes_escaping_closure"(<"index">, <"index">) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP9:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = #makes_escaping_closure_name, linkageName = "makes_escaping_closure{{.*}}", file = #file, line = [[#LN42:]],

# CHECK-LABEL:    lit.func @"makes_escaping_closure
# CHECK-NEXT:    debuginfo.value #[[VAR0:.*]] = %m : index loc(#[[LOC26:.*]])
# CHECK-NEXT:    debuginfo.value #[[VAR1:.*]] = %z : index
# CHECK-NEXT:    %anonymous2A = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !escaping
# CHECK-NEXT:    %0 = lit.call {{.*}}CI{{.*}}__init__{{.*}}"[{{.*}}](%anonymous2A, %m)
# CHECK-NEXT:    %anonymous2A_0 = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !index
# CHECK-NEXT:    %1 = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%anonymous2A_0, %anonymous2A)
# CHECK-NEXT:     = lit.call {{.*}}fn{{.*}}__moveinit__{{.*}}(%__result__, %anonymous2A_0)

# CHECK-DAG: #[[LOC26]] = loc(fused<#[[SP9]]>[#

alias int = __mlir_type.index


fn makes_escaping_closure(m: int, z: int) -> fn (n: int) escaping -> int:
    fn myclosure(n: int) escaping -> int:
        return m

    return myclosure
