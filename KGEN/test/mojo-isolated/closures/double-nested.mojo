# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | FileCheck %s

# CHECK: lit.file_module @"[[F:.*]]" attributes

# CHECK-COUNT-2: lit.struct.decl @"`_CI_


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK-LABEL: lit.func @"makes_escaping_closure
# CHECK: %anonymous2A = lit.varlet.decl "anonymous*"
# CHECK-NEXT: [[V1:%.*]] = lit.call {{.*}}CI_[[F]]_{{.*}}"::@"__init__{{.*}}(%anonymous2A, %m)
# CHECK-NEXT: %anonymous2A_0 = lit.varlet.decl "anonymous*"
# CHECK-NEXT:  = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%anonymous2A_0, %anonymous2A)
# CHECK-NEXT: [[V3:%.*]] = kgen.param.constant: none
# CHECK-NEXT: lit.return [[V3]]
# CHECK-NEXT: lit.end_func
fn makes_escaping_closure(m: MemType):
    fn myclosure(n: MemType) escaping -> MemType:
        fn nested_nested(k: MemType, l: MemType) escaping -> MemType:
            return n + k

        return n + m
