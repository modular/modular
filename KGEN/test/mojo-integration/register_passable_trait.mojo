# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -S -o - | FileCheck %s --check-prefix=ELABORATE
# RUN: %mojo -debug-level full %s 3 | FileCheck %s

from sys import argv


@register_passable("trivial")
trait RGTrivialTrait:
    fn doSomething(self):
        ...


@fieldwise_init
@register_passable("trivial")
struct Conforms(RGTrivialTrait):
    var x: Int

    @no_inline
    fn doSomething(self):
        print(self.x)


# ELABORATE: kgen.func @"{{.*}}bar{{.*}}"(%arg0: index)
# ELABORATE-NEXT: kgen.call @"{{.*}}::Conforms::doSomething{{.*}}"(%arg0) : (index) -> ()
@no_inline
fn bar[x: RGTrivialTrait](y: x):
    y.doSomething()


def main():
    var t = Conforms(atol(argv()[1]))
    # CHECK: 3
    bar(t)
