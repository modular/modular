# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

alias AnyCoroutine = __mlir_type.`!co.routine`


@value
@register_passable
struct Coroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`]:
    var value: __mlir_type.`!co.routine`

    fn __init__(out self, handle: AnyCoroutine):
        self.value = handle

    fn __await__(owned self) -> T:
        while __mlir_attr.true:
            pass


@value
@register_passable
struct RaisingCoroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`]:
    var value: __mlir_type.`!co.routine`

    fn __await__(owned self) raises -> T:
        while __mlir_attr.true:
            pass
