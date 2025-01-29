# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import sizeof

from gpu.host import DeviceContext, DeviceBuffer
from gpu.host._nvidia_cuda import (
    TMADescriptor,
    TensorMapSwizzle,
    create_tma_descriptor,
)
from gpu.memory import (
    AddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_reduce,
)
from gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
    mbarrier_arrive,
)
from layout import IntTuple, LayoutTensor
from layout.tensor_core_async import _CM_K_BYTES
from memory import UnsafePointer, stack_allocation

from sys._assembly import inlined_assembly
from utils.index import Index, IndexList


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


fn _tma_desc_tile_layout[
    type: DType,
    layout: Layout,
    is_k_major: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> Layout:
    constrained[
        sizeof[type]() >= 1, "Don't support sub-byte type in TMA yet."
    ]()

    alias dim0 = layout.shape[0].value()
    alias dim1 = layout.shape[1].value()

    @parameter
    if is_k_major:
        # TMA copies BM x core_matrix_num_columns each time.
        @parameter
        if swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE:
            return Layout.row_major(dim0, _CM_K_BYTES // sizeof[type]())

        # TMA copies BM x 128B each time.
        elif swizzle_mode == TensorMapSwizzle.SWIZZLE_128B:
            return Layout.row_major(dim0, 128 // sizeof[type]())

    return layout


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

    @always_inline
    fn init(self, num_threads: Int32 = 1):
        mbarrier_init(self.mbar, num_threads)

    @always_inline
    fn expect_bytes(self, bytes: Int32):
        mbarrier_arrive_expect_tx_shared(self.mbar, bytes)

    @always_inline
    fn wait(self, phase: UInt32 = 0):
        # Based on cutlass
        # https://github.com/NVIDIA/cutlass/blob/b78588d1630aa6643bf021613717bafb705df4ef/include/cute/arch/copy_sm90_desc.hpp#L92-L110
        alias asm = """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
        }"""
        inlined_assembly[asm, NoneType, constraints="r,r"](
            Int32(Int(self.mbar)), phase
        )

    @always_inline
    fn arrive(self) -> Int:
        return mbarrier_arrive(self.mbar)


# PipelineState keeps track of the current state of a barrier using circular indexing
#
@value
@register_passable("trivial")
struct PipelineState[num_stages: Int]:
    # The current index of the pipeline.
    var _index: Int
    # The current phase of the pipeline, it switch between 1 and 0
    var _phase: UInt32
    # The current count of the increments.
    var _count: UInt32

    @always_inline
    fn __init__(out self):
        self._index = 0
        self._phase = 0
        self._count = 0

    @always_inline
    fn index(self) -> Int:
        return self._index

    @always_inline
    fn phase(self) -> UInt32:
        return self._phase

    @always_inline
    fn step(mut self):
        """This function increase the index and count. Index will circle back to
        0 when it equals to the num_stage.
        """

        @parameter
        if num_stages > 0:
            self._index += 1
            self._count += 1
            if self._index == num_stages:
                self._index = 0
                self._phase ^= 1


# TMATensorTile is created on the host with specific memory and tile sizes.
# Each TMATensorTile provides an asynchronous load of a specific tile at specified tile coordinates.
#
@value
struct TMATensorTile[
    dtype: DType,
    layout: Layout,
    desc_layout: Layout = layout,
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
            dtype, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        mem_barrier: TMABarrier,
        coords: Tuple[UInt, UInt],
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        constrained[
            __type_of(dst).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()

        # The layout per copy can be smaller than the shared memory tile shape due to
        # WGMMA requirement. We may need multiple copies.
        # TODO: use layout algebra here
        alias copy_dim0 = desc_layout.shape[0].value()
        alias copy_dim1 = desc_layout.shape[1].value()
        alias copy_size = desc_layout.size()
        alias num_copies_dim0 = layout.shape[0].value() // copy_dim0
        alias num_copies_dim1 = layout.shape[1].value() // copy_dim1

        @parameter
        for i in range(num_copies_dim0):

            @parameter
            for j in range(num_copies_dim1):
                alias copy_offset = (i * num_copies_dim1 + j) * copy_size

                cp_async_bulk_tensor_shared_cluster_global(
                    dst.ptr + copy_offset,
                    UnsafePointer.address_of(self.descriptor).bitcast[
                        NoneType
                    ](),
                    mem_barrier.mbar,
                    Index(coords[0] + j * copy_dim1, coords[1] + i * copy_dim0),
                )

    # Schedules an asynchronous store into the global memory
    @always_inline
    fn async_store(
        self,
        src: LayoutTensor[
            dtype, layout, address_space = AddressSpace.SHARED, **_
        ],
        coords: Tuple[UInt, UInt],
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        constrained[
            __type_of(src).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()
        cp_async_bulk_tensor_global_shared_cta(
            src.ptr,
            UnsafePointer.address_of(self.descriptor).bitcast[NoneType](),
            Index(coords[0], coords[1]),
        )

    # Schedules an asynchronous store into the global memory
    @always_inline
    fn async_reduce[
        reduction_kind: StringLiteral
    ](
        self,
        src: LayoutTensor[
            dtype, layout, address_space = AddressSpace.SHARED, **_
        ],
        coords: Tuple[UInt, UInt],
    ):
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        constrained[
            __type_of(src).alignment % 128 == 0,
            "TMA requires 128B alignment in shared memory",
        ]()
        cp_async_bulk_tensor_reduce[reduction_kind=reduction_kind](
            src.ptr,
            UnsafePointer.address_of(self.descriptor).bitcast[NoneType](),
            Index(coords[0], coords[1]),
        )


# Creates a TMATensorTile with specified tile sizes.
#
@always_inline
def create_tma_tile[
    *tile_sizes: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: LayoutTensor) -> TMATensorTile[
    tensor.dtype,
    Layout.row_major(_to_int_tuple[*tile_sizes]()),
]:
    # the last dimension of smem shape has to be smaller or equals to the
    # swizzle bytes.
    alias swizzle_bytes = tile_sizes[tensor.rank - 1] * sizeof[tensor.dtype]()

    @parameter
    if swizzle_mode == TensorMapSwizzle.SWIZZLE_32B:
        constrained[
            swizzle_bytes <= 32,
            "Current swizzle bytes is "
            + String(swizzle_bytes)
            + " which exceeds 32B swizzle requirement.",
        ]()
    elif swizzle_mode == TensorMapSwizzle.SWIZZLE_64B:
        constrained[
            swizzle_bytes <= 64,
            "Current swizzle bytes is "
            + String(swizzle_bytes)
            + " which exceeds 64B swizzle requirement.",
        ]()
    elif swizzle_mode == TensorMapSwizzle.SWIZZLE_128B:
        constrained[
            swizzle_bytes <= 128,
            "Current swizzle bytes is "
            + String(swizzle_bytes)
            + " which exceeds 128B swizzle requirement.",
        ]()

    return create_tma_descriptor[tensor.dtype, 2, swizzle_mode](
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


@always_inline
def create_tma_tile[
    type: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    /,
    is_k_major: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_layout: Layout = Layout.row_major(tile_shape[0], tile_shape[1]),
    __desc_layout: Layout = _tma_desc_tile_layout[
        type, __tile_layout, is_k_major, swizzle_mode
    ](),
](ctx: DeviceContext, tensor: LayoutTensor[type, *_, **_]) -> TMATensorTile[
    type, __tile_layout, __desc_layout
]:
    # Current impl limitations
    constrained[rank == 2, "Only suppot 2D TMA"]()
    constrained[is_k_major, "Only K major layout supported in TMA"]()

    # constrained[
    #     swizzle_mode in (TensorMapSwizzle.SWIZZLE_NONE, TensorMapSwizzle.SWIZZLE_128B),
    #     "Swizzle is not yet supported in TMA",
    # ]()
    @parameter
    if swizzle_mode == TensorMapSwizzle.SWIZZLE_128B:
        constrained[
            (tile_shape[1] * sizeof[type]()) % 128 == 0,
            "128B swizzle requires K dim multiple of 128B",
        ]()
    else:
        constrained[
            swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE,
            "Only support 128B and no swizzle",
        ]()

    return create_tma_descriptor[type, 2, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr.address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        (tensor.dim(0), tensor.dim(1)),
        (tensor.stride[0](), tensor.stride[1]()),
        (__desc_layout.shape[0].value(), __desc_layout.shape[1].value()),
        (1, 1),
    )
