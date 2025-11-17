# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias AnyCoroutine = __mlir_type.`!co.routine`


@explicit_destroy
@register_passable
struct Coroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`]:
    var value: __mlir_type.`!co.routine`

    @implicit
    fn __init__(out self, handle: AnyCoroutine):
        self.value = handle

    fn __await__(deinit self) -> Self.T:
        while __mlir_attr.true:
            pass


@explicit_destroy
@register_passable
struct RaisingCoroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`]:
    var value: __mlir_type.`!co.routine`

    @implicit
    fn __init__(out self, handle: AnyCoroutine):
        self.value = handle

    fn __await__(deinit self) raises -> Self.T:
        while __mlir_attr.true:
            pass
