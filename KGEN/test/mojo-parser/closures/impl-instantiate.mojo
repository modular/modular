# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: MemType, w: Int):
    # CHECK: %anonymous2A = lit.varlet.decl "anonymous*" var synth
    # CHECK-NEXT: [[ANONPTR:%.*]] = lit.ref.to_pointer %anonymous2A
    # CHECK-NEXT: %anonymous2A_0 = lit.varlet.decl "anonymous*" var synth : !lit.ref<mut !MemType,
    # CHECK-NEXT: [[ANONPTR_0:%.*]] = lit.ref.to_pointer %anonymous2A_0
    # CHECK-NEXT: kgen.call @"{{.*}}@"__copyinit__{{.*}}"([[ANONPTR_0]], %m)
    # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"([[ANONPTR]], [[ANONPTR_0]])
    # CHECK-NEXT: %anonymous2A_1 = lit.varlet.decl "anonymous*" var synth
    # CHECK-NEXT: [[ANONPTR_1:%.*]] = lit.ref.to_pointer %anonymous2A_1
    # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"([[ANONPTR_1]], [[ANONPTR]])
    fn myclosure_with_mem_types(n: MemType) escaping -> MemType:
        return n + m

    # CHECK: %anonymous2A_2 = lit.varlet.decl "anonymous*" var synth
    # CHECK-NEXT: [[ANONPTR:%.*]] = lit.ref.to_pointer %anonymous2A_2
    # CHECK-NEXT: [[A:%.*]] = lit.ref.load %a
    # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"([[ANONPTR]], [[A]], %w)
    # CHECK-NEXT: %anonymous2A_3 = lit.varlet.decl "anonymous*" var synth
    # CHECK-NEXT: [[ANONPTR_0:%.*]] = lit.ref.to_pointer %anonymous2A_3
    # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"([[ANONPTR_0]], [[ANONPTR]])
    var a = w

    fn myclosure_with_reg_types(x: Int) escaping -> Int:
        a = a + 1
        return x + w
