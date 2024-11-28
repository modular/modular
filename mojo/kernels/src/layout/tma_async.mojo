# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import sizeof

from gpu.host.memory_v1 import TMADescriptor, create_tma_descriptor
from gpu.memory import AddressSpace, cp_async_bulk_tensor_shared_cluster_global
from gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
)
from layout import IntTuple, LayoutTensor
from memory import UnsafePointer, stack_allocation

from utils.index import Index


# Returns an IntTuple of variadic Int values.
#
fn _to_int_tuple[*vals: Int]() -> IntTuple:
    res = IntTuple()

    @parameter
    fn length() -> Int:
        return __mlir_op.`pop.variadic.size`(vals)

    @parameter
    for i in range(length()):
        res.append(vals[i])
    return res


# A memory barrier with blocking wait.
#
@value
@register_passable("trivial")
struct TMABarrier(CollectionElement):
    var mbar: UnsafePointer[
        Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]

    @always_inline
    fn __init__(inout self):
        self.mbar = stack_allocation[
            1, Int64, address_space = AddressSpace.SHARED
        ]()

        mbarrier_init(self.mbar, 1)

    @always_inline
    fn wait(self):
        mbarrier_try_wait_parity_shared(self.mbar, 0, 10000000)


# TMATensorTile is created on the host with specific memory and tile sizes.
# Each TMATensorTile provides an asynchronous load of a specific tile at specified tile coordinates.
#
@value
struct TMATensorTile[
    dtype: DType,
    layout: Layout,
]:
    var descriptor: TMADescriptor

    fn __init__(inout self, descriptor: TMADescriptor):
        self.descriptor = descriptor

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.descriptor = other.descriptor

    # Schedules an asynchronous load of tile at and returns the memory barrier for the operation.
    @always_inline
    fn async_load(
        self, coord_0: Int, coord_1: Int
    ) -> Tuple[
        LayoutTensor[
            dtype,
            layout,
            address_space = AddressSpace.SHARED,
        ],
        TMABarrier,
    ] as res:
        var sh_mem = LayoutTensor[
            dtype,
            layout,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var barrier = TMABarrier()

        alias size_in_bytes = layout.size() * sizeof[dtype]()

        mbarrier_arrive_expect_tx_shared(barrier.mbar, size_in_bytes)

        cp_async_bulk_tensor_shared_cluster_global(
            sh_mem.ptr,
            UnsafePointer.address_of(self.descriptor).bitcast[NoneType](),
            barrier.mbar,
            Index(coord_0 * sh_mem.shape[0](), coord_1 * sh_mem.shape[1]()),
        )

        return sh_mem, barrier


# Creates a TMATensorTile with specified tile sizes.
#
@always_inline
def create_tma_tile[
    *tile_sizes: Int
](tensor: LayoutTensor) -> TMATensorTile[
    tensor.dtype,
    Layout.row_major(_to_int_tuple[*tile_sizes]()),
]:
    return create_tma_descriptor[tensor.dtype, 2](
        tensor.ptr.bitcast[address_space = AddressSpace.GENERIC](),
        (tensor.dim(0), tensor.dim(1)),
        (tensor.stride[0](), tensor.stride[1]()),
        (tile_sizes[0], tile_sizes[1]),
        (1, 1),
    )
