# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import sizeof

from gpu.host import DeviceContext, DeviceBuffer
from gpu.host.nvidia_cuda import TMADescriptor, create_tma_descriptor
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
    fn __init__(out self):
        # We follow cutlass to adopt 16B alignment instead of 8B suggested by the ptx doc.
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=fence%2520proxy%2520async%2520shared%25203A%25203Acta#size-and-alignment-of-mbarrier-object
        # https://github.com/NVIDIA/cutlass/blob/main/test/unit/cute/hopper/tma_load_testbed.hpp#L50
        self.mbar = stack_allocation[
            1,
            Int64,
            address_space = AddressSpace.SHARED,
            alignment=16,
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

    @always_inline
    @implicit
    fn __init__(out self, descriptor: TMADescriptor):
        self.descriptor = descriptor

    @always_inline
    fn __copyinit__(mut self, other: Self):
        self.descriptor = other.descriptor

    # Schedules an asynchronous copy into the destination tile at the given coordinates.
    #
    @always_inline
    fn async_copy(
        self,
        dst: LayoutTensor[
            dtype,
            layout,
            address_space = AddressSpace.SHARED,
        ],
        mem_barrier: TMABarrier,
        coords: Tuple[UInt, UInt],
    ):
        alias size_in_bytes = layout.size() * sizeof[dtype]()

        mbarrier_arrive_expect_tx_shared(mem_barrier.mbar, size_in_bytes)

        cp_async_bulk_tensor_shared_cluster_global(
            dst.ptr,
            UnsafePointer.address_of(self.descriptor).bitcast[NoneType](),
            mem_barrier.mbar,
            Index(coords[0], coords[1]),
        )


# Creates a TMATensorTile with specified tile sizes.
#
@always_inline
def create_tma_tile[
    *tile_sizes: Int
](ctx: DeviceContext, tensor: LayoutTensor) -> TMATensorTile[
    tensor.dtype,
    Layout.row_major(_to_int_tuple[*tile_sizes]()),
]:
    return create_tma_descriptor[tensor.dtype, 2](
        DeviceBuffer(
            ctx,
            tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        (tensor.dim(0), tensor.dim(1)),
        (tensor.stride[0](), tensor.stride[1]()),
        (tile_sizes[0], tile_sizes[1]),
        (1, 1),
    )
