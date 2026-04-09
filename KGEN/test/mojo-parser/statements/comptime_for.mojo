# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# comptime for
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct IterRange(ImplicitlyCopyable, Iterator):
    comptime Element = Int

    var value: Int

    def __iter__(self) -> Self:
        return self

    def __next__(mut self) raises StopIteration -> Int:
        if self.value <= 0:
            raise StopIteration()
        return self.value


struct MyType:
    pass


def use(value: MyType):
    pass


# CHECK-LABEL: lit.fn @"comptime_for_basic
# CHECK-SAME: <a: !Int>[mut [[LT:.*]]](%value: !lit.ref<!MyType, mut [[LT]]>
def comptime_for_basic[a: Int](var value: MyType):
    # CHECK-NEXT: kgen.param.for [[iter:.*]]:  !IterRange in :!IterRange apply
    # CHECK-NEXT: has_next {{.*}}paramfor_has_next
    # CHECK-NEXT: get_next_iter :{{.*}}paramfor_next_iter{{.*}}<:!Iterator_Copyable !IterRange>
    comptime for i in IterRange(a):
        # CHECK: [[IMM:%.*]] = lit.ref.immut %value
        # CHECK: use{{.*}}[muttoimm [[LT]]]([[IMM]])
        use(value)
        # CHECK: kgen.param.for.continue


# ===----------------------------------------------------------------------=== #
# @parameter for (legacy syntax - same IR as comptime for)
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"parameter_for
# CHECK-SAME: <a: !Int>[mut [[LT:.*]]](%value: !lit.ref<!MyType, mut [[LT]]>
def parameter_for[a: Int](var value: MyType):
    # CHECK-NEXT: kgen.param.for [[iter:.*]]:  !IterRange in :!IterRange apply
    # CHECK-NEXT: has_next {{.*}}paramfor_has_next
    # CHECK-NEXT: get_next_iter :{{.*}}paramfor_next_iter{{.*}}<:!Iterator_Copyable !IterRange>
    @parameter
    for i in IterRange(a):
        # CHECK: [[IMM:%.*]] = lit.ref.immut %value
        # CHECK: use{{.*}}[muttoimm [[LT]]]([[IMM]])
        use(value)
        # CHECK: kgen.param.for.continue
