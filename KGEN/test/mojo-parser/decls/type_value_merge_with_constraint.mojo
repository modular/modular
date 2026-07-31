# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN:  %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# Test that merging two type values through a comptime ternary preserves the
# common (intersection) trait bound *and* the conditional-conformance
# constraint carried by that bound.


# The field's trait bound. Deliberately does NOT require `ImplicitlyDeletable`.
trait Storage(Defaultable):
    pass


def predicate() -> Bool:
    pass


@explicit_destroy("StorageA must be explicitly destroyed.")
struct StorageA[T: AnyType](
    ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable),
    Storage, Movable where False,
):
    def __init__(out self):
        pass

    def __del__(deinit self) where conforms_to(Self.T, ImplicitlyDeletable):
        pass


@explicit_destroy("StorageB must be explicitly destroyed.")
struct StorageB[T: AnyType](
    ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable),
    Storage, Movable where False,
):
    def __init__(out self):
        pass

    def __del__(deinit self) where conforms_to(Self.T, ImplicitlyDeletable):
        pass


# CHECK-LABEL: lit.struct.decl @Container
@explicit_destroy("Container must be explicitly destroyed.")
struct Container[T: AnyType](
    ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable), Movable where False,
):
    # The merged `_Storage` type value must be a constrained trait bound.

    # CHECK: lit.alias.decl {{.*}}_Storage{{.*}}: !constrained_{{.*}}ImplicitlyDeletable{{.*}}Storage
    comptime _Storage = StorageA[Self.T] if predicate() else StorageB[Self.T]

    var _storage: Self._Storage

    def __init__(out self):
        self._storage = Self._Storage()

    def __del__(deinit self) where conforms_to(Self.T, ImplicitlyDeletable):
        self._storage^.__deinit__()
