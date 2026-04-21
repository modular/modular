# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file -debug-level full | FileCheck %s

# COM: Verify that stateless closures are lifted.

# COM: This is how a struct generator would be emitted for the closure.
# CHECK-NOT: kgen.struct.generator @"outer()::stateless"
# CHECK-NOT: lit.struct.decl @"def() -> Int_Mova_Impl_Copy_Impl"

# CHECK-DAG: [[INT:!.*]] = !lit.struct<@{{.*}}::@Int>
# CHECK-DAG: lit.fn @"outer()"() -> [[INT]]
# CHECK-DAG: lit.fn @"stateless()`{{.*}}"() -> [[INT]] attributes {{{.*}}sourceName = "stateless"


def outer() -> Int:
    def stateless() unified {} -> Int:
        return 123

    return stateless()


# // -----

# COM: Verify that stateless closures with the same name in different functions don't collide.

# CHECK: lit.fn @"one()"() -> !Int
# CHECK: %0 = lit.call tail @{{.*}}::@"[[STATELESS1:stateless.*]]"()
# CHECK: lit.fn @"two()"() -> !Int
# CHECK-NOT: %0 = lit.call tail @{{.*}}::@"[[STATELESS1]]"()


def one() -> Int:
    def stateless() unified {} -> Int:
        return 123

    return stateless()


def two() -> Int:
    def stateless() unified {} -> Int:
        return 456

    return stateless()


# // -----

# COM: Verify that the entire function signature is preserved


def outer() -> Int:
    def stateless() raises unified {} -> Int:
        return 123


# CHECK: lit.fn @"stateless()`{{.*}}"[{{.*}}]({{.*}}) throws -> i1

# // -----

# COM: Verify that the lifted function can be wrapped as a closure

# COM: this is the top-level wrapper struct
# CHECK: lit.struct.decl @"def() -> Int_PtrWrapper"
# CHECK: lit.fn @"stateless()`{{.*}}"()


def uses[T: def() unified -> Int](f: T) -> Int:
    return f()


def outer() -> Int:
    def stateless() unified {} -> Int:
        return 123

    return uses(stateless)


# // -----

# COM: Verify that closures with captured parameter references aren't lifted

# CHECK-NOT: lit.fn @"unified_closure()`{{.*}}"()


def iter[v: Int]():
    def unified_closure() unified {} -> Int:
        return v

    unified_closure()
