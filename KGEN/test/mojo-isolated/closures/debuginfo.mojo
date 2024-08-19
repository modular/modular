# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s -debug-level full -mlir-print-debuginfo -split-input-file | FileCheck %s

# COM: This tests that code generated to support capturing closures is located and scoped correctly.

# CHECK-LABEL:    lit.func @"makes_escaping_closure
# CHECK-NEXT:    %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!escaping
# CHECK-NEXT:    %0 = lit.call {{.*}}CI{{.*}}__init__{{.*}}"[{{.*}}](%anonymous2A, %m)
# CHECK-NEXT:    %myclosure = lit.var.decl "myclosure" var : !lit.ref<!index
# CHECK-NEXT:    %1 = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%myclosure, %anonymous2A)
# CHECK-NEXT:    lit.ownership.use %myclosure
# CHECK-NEXT:    lit.call {{.*}}fn{{.*}}__moveinit__{{.*}}(%__result__, %myclosure){{.*}} loc(#[[LOC26:.*]])

# CHECK-DAG: #makes_escaping_closure_name = #debuginfo.source_name<(fn)"makes_escaping_closure"(<"index">, <"index">) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP9:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = #makes_escaping_closure_name, linkageName = "makes_escaping_closure{{.*}}", file = #file, line = [[#LN42:]],
# CHECK-DAG: #[[LOC26]] = loc(fused<#[[SP9]]>[#


fn makes_escaping_closure(m: int, z: int) -> fn (n: int) escaping -> int:
    fn myclosure(n: int) -> int:
        return m

    return myclosure^


# // -----

# COM: This tests that code generated for closures inside lexical blocks have the correct debug scope.

# CHECK-DAG: #Bool_name = #debuginfo.source_name<(struct)"Bool" from {{.*}}>

# CHECK-LABEL: lit.func @"closure_in_block
# CHECK:       hlcf.elif
# CHECK:         %anonymous2A = lit.var.decl "anonymous*" synth : {{.*}} loc(#[[LOC0:.*]])
# CHECK-NEXT:     = lit.call {{.*}}CI{{.*}}__init__{{.*}}"[{{.*}}](%anonymous2A, %m) : {{.*}} loc(#[[LOC0]])
# CHECK-NEXT:    %myclosure = lit.var.decl "myclosure" var : {{.*}} loc(#[[LOC0]])
# CHECK-NEXT:     = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%myclosure, %anonymous2A) : {{.*}} loc(#[[LOC0]])
# CHECK-NEXT:     = lit.ref.immut %myclosure : {{.*}} loc(#[[LOC1:.*]])
# CHECK-NEXT:     = lit.call {{.*}}fn{{.*}}__call__({{.*}}) : {{.*}} loc(#[[LOC1]])

# CHECK-DAG: #closure_in_block_name = #debuginfo.source_name<(fn)"closure_in_block"(<"index">, <"index">, #Bool_name) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<compileUnit = #compile_unit, scope = #file, name = #closure_in_block_name, linkageName = "closure_in_block{{.*}}", file = #file,
# CHECK-DAG: #[[LEXBLOCK:.*]] = #debuginfo.lexical_block<scope = #[[SP]], file = #file,
# CHECK-DAG: #[[LOC0]] = loc(fused<#[[LEXBLOCK]]>[#
# CHECK-DAG: #[[LOC1]] = loc(fused<#[[LEXBLOCK]]>[#


fn closure_in_block(m: int, z: int, b: Bool) -> int:
    if b:

        fn myclosure(n: int) -> int:
            return m

        return myclosure(z)

    return z
