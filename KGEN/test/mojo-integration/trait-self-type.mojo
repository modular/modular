# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s


trait SelfMethod:
    fn foo(self):
        pass


struct SelfStruct(SelfMethod):
    fn foo(self):
        pass


# CHECK-LABEL: kgen.func @"{{.*}}call_it{{.*}}T=trait-self-type::SelfStruct.foo
fn call_it[T: SelfMethod](x: T):
    # CHECK: call {{.*}}SelfStruct::foo{{.*}}(%arg0)
    x.foo()


@export
fn pass_it(x: SelfStruct):
    call_it(x)
