# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.file_module @[[F:.*]] {

# CHECK-COUNT-2: lit.struct.decl @"`_CI_


@fieldwise_init
struct MemType(ImplicitlyCopyable):
    def __add__(self, rhs: MemType) -> MemType:
        return MemType()


# CHECK-LABEL: lit.fn @"makes_escaping_closure
# CHECK: %__call_result_tmp__ = lit.var.decl "__call_result_tmp__"
# CHECK-NEXT: [[V1:%.*]] = lit.call {{.*}}CI_[[F]]_{{.*}}"::@"__init__{{.*}}(%m, %__call_result_tmp__)
# CHECK-NEXT: %myclosure = lit.var.decl "myclosure"
# CHECK-NEXT:  = lit.call {{.*}}def{{.*}}__init__{{.*}}(%__call_result_tmp__, %myclosure)
# CHECK-NEXT: [[V3:%.*]] = kgen.param.constant: none
# CHECK-NEXT: lit.return [[V3]]
# CHECK-NEXT: lit.end_fn
def makes_escaping_closure(m: MemType):
    def myclosure(n: MemType) -> MemType:
        def nested_nested(k: MemType, l: MemType) -> MemType:
            return n + k

        return n + m
