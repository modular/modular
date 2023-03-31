# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from Index import Index, StaticIntTuple
from Int import Int
from Buffer import NDBuffer, Buffer
from SIMD import SIMD
from Image import (
    ImageData,
    Image2DLayout,
    ImageShape,
)
from Range import range
from List import DimList
from Math import min, max, add, div_ceil
from Functional import async_parallelize, vectorize_unroll
from LLCL import Runtime, OutputChainPtr
from Pointer import Pointer
from Pointer import DTypePointer, product as pointer_product
from TargetInfo import dtype_simd_width
from DType import DType
from Numerics import neginf

# Pooling method.
@register_passable("trivial")
struct PoolMethod:
    var value: Int
    alias MAX = PoolMethod(0)  # Max pooling.
    alias AVG = PoolMethod(1)  # Average pooling not counting padded regions.

    @always_inline("nodebug")
    fn __clone__(self&) -> Self:
        return Self {value: self.value}

    @always_inline("nodebug")
    fn __init__(value: Int) -> PoolMethod:
        return PoolMethod {value: value}

    @always_inline("nodebug")
    fn __eq__(self, rhs: PoolMethod) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: PoolMethod) -> Bool:
        return self.value != rhs.value


@register_passable("trivial")
struct Pool2d[
    static_output_shape: DimList[4],
    static_input_shape: DimList[4],
    type: DType,
    static_data_layout: Image2DLayout,
    init_fn: __mlir_type[`!kgen.signature<<>() -> `, SIMD[1, type], `>`],
    update_fn: __mlir_type[
        `!kgen.signature<<>(`,
        SIMD[1, type],
        ` borrow,`,
        SIMD[1, type],
        ` borrow) -> `,
        SIMD[1, type],
        `>`,
    ],
    reduce_fn: __mlir_type[
        `!kgen.signature<<>(`,
        SIMD[1, type],
        ` borrow,`,
        Int,
        ` borrow) -> `,
        SIMD[1, type],
        `>`,
    ],
]:
    """Struct wrapper for pool implementation."""

    # Input params.
    var output: ImageData[static_output_shape, type, static_data_layout]
    var input: ImageData[static_input_shape, type, static_data_layout]
    var pad_h: StaticIntTuple[2]
    var pad_w: StaticIntTuple[2]
    var filter_shape: StaticIntTuple[2]
    var stride: StaticIntTuple[2]
    var dilation: StaticIntTuple[2]
    var num_tasks: Int

    # Derived params.
    var output_shape: ImageShape
    var input_shape: ImageShape

    @staticmethod
    fn run(
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        filter_shape: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        out_chain: OutputChainPtr,
    ):
        """Interface function to run a pooling op on the given input and
        filter tensor and stores the result in the give output tensor.

        Args:
            output: Pre-allocated output tensor space.
            input: Batched image input to the pool2d operator.
            pad_h: Padding on the height dimension with assumed tuple def
              (pad_lower, pad_upper).
            pad_w: Padding on the width dimension with assumed tuple def
              (pad_lower, pad_upper).
            stride: Strides on height and width dimensions with assumed
              tuple def (stride_h, stride_w).
            dilation: Dilations on height and width dimensions with assumed
              tuple def (dilation_h, dilation_w).
            out_chain: OutputChain.
        """
        # TODO: Find a heuristic to replace the magic numbers.
        alias min_task_num_slices = 64
        alias vector_width = dtype_simd_width[type]()
        alias unroll_factor = 8

        let num_threads = out_chain.get_runtime().parallelism_level()
        let num_tasks = min(
            div_ceil(output.num_elements(), min_task_num_slices), num_threads
        )

        let work = output.num_elements()
        let work_block_size = div_ceil(work, num_tasks)

        @always_inline
        fn task_func(task_id: Int):

            # Create an instance of the pooling op.
            let pool2d = Pool2d[
                static_output_shape,
                static_input_shape,
                type,
                static_data_layout,
                init_fn,
                update_fn,
                reduce_fn,
            ](
                output,
                input,
                pad_h,
                pad_w,
                filter_shape,
                stride,
                dilation,
                num_tasks,
            )

            let offset = task_id * work_block_size

            @always_inline
            fn func_wrapper[simd_width: Int](idx: Int):
                var values = SIMD[simd_width, type](0)

                for i in range(simd_width):
                    # Compute the result value at this specific output position.
                    values[i] = pool2d._compute_point(offset + idx + i)

                # Store the computed output at the given output position.
                pool2d.output.data.flatten().simd_store[simd_width](
                    offset + idx,
                    values,
                )

            vectorize_unroll[vector_width, unroll_factor, func_wrapper](
                min(work_block_size, work - offset)
            )

        async_parallelize[task_func](out_chain, num_tasks)

    fn __init__(
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        filter_shape: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        num_tasks: Int,
    ) -> Pool2d[
        static_output_shape,
        static_input_shape,
        type,
        static_data_layout,
        init_fn,
        update_fn,
        reduce_fn,
    ]:
        """Constructor of a pooling op instance on the given input tensor and
        stores the result in the give output tensor.

        Args:
            output: Pre-allocated output tensor space.
            input: Batched image input to the pool2d operator.
            pad_h: Padding on the height dimension with assu-
              med tuple def (pad_lower, pad_lower).
            pad_w: Padding on the width dimension with assum-
              ed tuple def (pad_lower, pad_lower).
            filter_shape: Filter size on height and width
              dimensions with assumed tuple def (filter_h, filter_w).
            stride: Strides on height and width dimensions
              with assumed tuple def (stride_h, stride_w).
            dilation: Dilations on height and width dimensi-
              ons with assumed tuple def (dilation_h, dilation_w).
            num_tasks: Number of tasks to run in parallel.

        Returns:
            An instance of the pooling operator with the input and output buffers
            registered.
        """
        var pool2d: Pool2d[
            static_output_shape,
            static_input_shape,
            type,
            static_data_layout,
            init_fn,
            update_fn,
            reduce_fn,
        ]
        # Register input/output buffers and parameters.
        pool2d.output = output
        pool2d.input = input
        pool2d.pad_h = pad_h
        pool2d.pad_w = pad_w
        pool2d.filter_shape = filter_shape
        pool2d.stride = stride
        pool2d.dilation = dilation
        pool2d.num_tasks = num_tasks

        # Derive layout agnostic shape information.
        pool2d.output_shape = ImageShape.__init__[
            static_output_shape, type, static_data_layout
        ](output)
        pool2d.input_shape = ImageShape.__init__[
            static_input_shape, type, static_data_layout
        ](input)

        return pool2d

    fn _compute_point(self, idx: Int) -> SIMD[1, type]:
        """Implementation of the inner loop computation of a pooling operator
        producing a single scalar value at the given output tensor index.

        Args:
            idx: Flat index specifying which value of the output tensor to
              produce.
        """
        let output_idx = self.output.get_tuple_index(idx)

        # Initialize the result of this point.
        var value: SIMD[1, type] = init_fn()

        # Extract the H and W size of the input image.
        let image_bound = StaticIntTuple[2](
            self.input_shape.H, self.input_shape.W
        )

        # Iterate on filter height dimension.
        for r_idx in range(self.filter_shape[0]):
            # Iterate on filter width dimension.
            for s_idx in range(self.filter_shape[1]):
                # Compute input access index, on the H and W dimension.
                let input_image_index = (
                    # Output HxW with striding.
                    (
                        Index(
                            output_idx[2],
                            output_idx[3],
                        )
                        * self.stride
                    )
                    +
                    # filter RxS with dilation.
                    (Index(r_idx, s_idx) * self.dilation)
                    -
                    # Padding offset, using the left padding only here.
                    Index(self.pad_h[0], self.pad_w[0])
                )

                if (
                    # Check that the current image index is within valid range
                    # on the input image data tensor.
                    Index(0, 0)
                    <= input_image_index
                    < image_bound
                ):
                    # Compute function on input data.
                    value = update_fn(
                        value,
                        self.input[
                            output_idx[0],  # N
                            output_idx[1],  # C
                            input_image_index[0],  # H
                            input_image_index[1],  # W
                        ],
                    )
        return reduce_fn(value, self.filter_shape[0] * self.filter_shape[1])


@always_inline
fn max_pool_init_fn[type: DType]() -> SIMD[1, type]:
    return neginf[type]()


@always_inline
fn max_pool_update_fn[
    type: DType
](a: SIMD[1, type], b: SIMD[1, type]) -> SIMD[1, type]:
    return max(a, b)


@always_inline
fn max_pool_reduce_fn[type: DType](a: SIMD[1, type], s: Int) -> SIMD[1, type]:
    return a


@always_inline
fn avg_pool_init_fn[type: DType]() -> SIMD[1, type]:
    return 0


@always_inline
fn avg_pool_update_fn[
    type: DType
](a: SIMD[1, type], b: SIMD[1, type]) -> SIMD[1, type]:
    return add(a, b)


@always_inline
fn avg_pool_reduce_fn[type: DType](a: SIMD[1, type], s: Int) -> SIMD[1, type]:
    return a / s
