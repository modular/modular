# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s


trait SelfMethod:
    def foo(self):
        pass


struct SelfStruct(SelfMethod):
    def foo(self):
        pass


# CHECK-LABEL: kgen.func @"{{.*}}call_it{{.*}}T=
def call_it[T: SelfMethod](x: T):
    # CHECK: call {{.*}}SelfStruct::foo{{.*}}(%arg0)
    x.foo()


@export
def pass_it(x: SelfStruct) abi("Mojo"):
    call_it(x)
