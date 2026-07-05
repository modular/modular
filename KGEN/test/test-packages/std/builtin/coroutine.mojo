# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

comptime AnyCoroutine = __mlir_type.`!co.routine`


struct Coroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`](
    ImplicitlyDeletable where False, RegisterPassable
):
    var value: __mlir_type.`!co.routine`

    @implicit
    def __init__(out self, handle: AnyCoroutine):
        self.value = handle

    def __await__(deinit self) -> Self.T:
        while True:
            pass


struct RaisingCoroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`](
    ImplicitlyDeletable where False, RegisterPassable
):
    var value: __mlir_type.`!co.routine`

    @implicit
    def __init__(out self, handle: AnyCoroutine):
        self.value = handle

    def __await__(deinit self) raises -> Self.T:
        while True:
            pass
