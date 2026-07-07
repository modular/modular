# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Regression test for MOCO-4227.
#
# Tests trait inheritance where a parent trait overrides an overloaded method
# from its grandparent with the same signature, and a child trait inherits
# from that parent.

trait GrandParent:
    def foo(self):
        pass

    def foo(self, x: Int):
        pass


trait Parent(GrandParent):
    # Override with same signature as inherited foo(self).
    def foo(self):
        pass


# Use the overload with Int to avoid ambiguity.
def force_resolution[T: Parent](x: T):
    x.foo(42)


# CHECK-LABEL: lit.trait.decl @Child
trait Child(Parent):
    pass
