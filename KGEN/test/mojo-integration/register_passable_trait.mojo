# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -S -o - | FileCheck %s --check-prefix=ELABORATE
# RUN: %mojo -debug-level full %s 3 | FileCheck %s

from std.sys import argv


trait RGTrivialTrait(TrivialRegisterPassable):
    def doSomething(self):
        ...


@fieldwise_init
struct Conforms(RGTrivialTrait):
    var x: Int

    @no_inline
    def doSomething(self):
        print(self.x)


# ELABORATE: kgen.func @"{{.*}}bar{{.*}}"(%arg0: !kgen.scalar<index>)
# ELABORATE-NEXT: kgen.call @"{{.*}}::Conforms::doSomething{{.*}}"(%arg0) : (!kgen.scalar<index>) -> ()
@no_inline
def bar[x: RGTrivialTrait](y: x):
    y.doSomething()


def main() raises:
    var t = Conforms(atol(argv()[1]))
    # CHECK: 3
    bar(t)
