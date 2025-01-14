# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s -debug-level full -mlir-print-debuginfo -split-input-file | FileCheck %s

# COM: This tests that code generated to support capturing closures is located and scoped correctly.

# CHECK-LABEL:    lit.fn @"makes_escaping_closure
# CHECK-NEXT:    %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!escaping
# CHECK-NEXT:    %0 = lit.call {{.*}}CI{{.*}}__init__{{.*}}"[{{.*}}](%m, %anonymous2A)
# CHECK-NEXT:    %myclosure = lit.var.decl "myclosure" var : !lit.ref<!index
# CHECK-NEXT:    %1 = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%anonymous2A, %myclosure)
# CHECK-NEXT:    lit.ownership.use %myclosure
# CHECK-NEXT:    lit.call {{.*}}fn{{.*}}__moveinit__{{.*}}(%myclosure, %__result__){{.*}} loc(#[[LOC26:.*]])

# CHECK-DAG: #makes_escaping_closure_name = #debuginfo.source_name<(fn)"makes_escaping_closure"(<"index">, <"index">) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP9:.*]] = #debuginfo.subprogram<{{.*}}, sourceName = #makes_escaping_closure_name, linkageName = "makes_escaping_closure{{.*}}", {{.*}}, line = [[#LN42:]],
# CHECK-DAG: #[[LOC26]] = loc(fused<#[[SP9]]>[#


fn makes_escaping_closure(m: Index, z: Index) -> fn (n: Index) escaping -> Index:
    fn myclosure(n: Index) -> Index:
        return m

    return myclosure^


# // -----

# COM: This tests that code generated for closures inside lexical blocks have the correct debug scope.

# CHECK-DAG: #Bool_name = #debuginfo.source_name<(struct)"Bool" from {{.*}}>

# CHECK-LABEL: lit.fn @"closure_in_block
# CHECK:       hlcf.elif
# CHECK:         %anonymous2A = lit.var.decl "anonymous*" synth : {{.*}} loc(#[[LOC0:.*]])
# CHECK-NEXT:     = lit.call {{.*}}CI{{.*}}__init__{{.*}}"[{{.*}}](%m, %anonymous2A) : {{.*}} loc(#[[LOC0]])
# CHECK-NEXT:    %myclosure = lit.var.decl "myclosure" var : {{.*}} loc(#[[LOC0]])
# CHECK-NEXT:     = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%anonymous2A, %myclosure) : {{.*}} loc(#[[LOC0]])
# CHECK-NEXT:     = lit.ref.immut %myclosure : {{.*}} loc(#[[LOC1:.*]])
# CHECK-NEXT:     = lit.call {{.*}}fn{{.*}}__call__({{.*}}) : {{.*}} loc(#[[LOC1]])

# CHECK-DAG: #closure_in_block_name = #debuginfo.source_name<(fn)"closure_in_block"(<"index">, <"index">, #Bool_name) from <(module)"debuginfo">>
# CHECK-DAG: #[[SP:.*]] = #debuginfo.subprogram<{{.*}}, sourceName = #closure_in_block_name, linkageName = "closure_in_block{{.*}}",
# CHECK-DAG: #[[LEXBLOCK:.*]] = #debuginfo.lexical_block<scope = #[[SP]],
# CHECK-DAG: #[[LOC0]] = loc(fused<#[[LEXBLOCK]]>[#
# CHECK-DAG: #[[LOC1]] = loc(fused<#[[LEXBLOCK]]>[#


fn closure_in_block(m: Index, z: Index, b: Bool) -> Index:
    if b:

        fn myclosure(n: Index) -> Index:
            return m

        return myclosure(z)

    return z
