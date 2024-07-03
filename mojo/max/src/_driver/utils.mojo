# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from .anytensor import AnyTensor


fn _steal_device_memory_impl_ptr(
    inout memory: AnyTensor,
) raises -> UnsafePointer[NoneType]:
    """This takes `memory` as inout and not owned because it is called on
    References owned by a List (returned by List.__getitem__()).
    """
    var taken_memory = memory.take()

    var tmp_device_tensor = taken_memory^.to_device_tensor()
    var taken_device_memory = tmp_device_tensor._storage.take()

    var ptr = taken_device_memory^._steal_impl_ptr()
    return ptr
