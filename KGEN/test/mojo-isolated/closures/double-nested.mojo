# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.file_module @"[[F:.*]]"

# CHECK-COUNT-2: lit.struct.decl @"`_CI_


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK-LABEL: lit.func @"makes_escaping_closure
# CHECK: %anonymous2A = lit.var.decl "anonymous*"
# CHECK-NEXT: [[V1:%.*]] = lit.call {{.*}}CI_[[F]]_{{.*}}"::@"__init__{{.*}}(%anonymous2A, %m)
# CHECK-NEXT: %myclosure = lit.var.decl "myclosure"
# CHECK-NEXT:  = lit.call {{.*}}fn{{.*}}__init__{{.*}}(%myclosure, %anonymous2A)
# CHECK-NEXT: [[V3:%.*]] = kgen.param.constant: none
# CHECK-NEXT: lit.return [[V3]]
# CHECK-NEXT: lit.end_func
fn makes_escaping_closure(m: MemType):
    fn myclosure(n: MemType) -> MemType:
        fn nested_nested(k: MemType, l: MemType) -> MemType:
            return n + k

        return n + m
