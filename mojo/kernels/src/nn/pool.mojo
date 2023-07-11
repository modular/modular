# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from DType import DType
from Functional import async_parallelize, vectorize_unroll
from Image import ImageData, Image2DLayout, ImageShape
from Index import Index, StaticIntTuple
from LLCL import OutputChainPtr
from List import DimList
from Math import min, max, add, div_ceil, neginf
from Range import range
from SIMD import SIMD
from TargetInfo import simdwidthof

# Pooling method.
@value
@register_passable("trivial")
struct PoolMethod:
    var value: Int
    alias MAX = PoolMethod(0)  # Max pooling.
    alias AVG = PoolMethod(1)  # Average pooling not counting padded regions.

    @always_inline("nodebug")
    fn __eq__(self, rhs: PoolMethod) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: PoolMethod) -> Bool:
        return self.value != rhs.value


@register_passable("trivial")
struct Pool2d[
    static_output_shape: DimList,
    static_input_shape: DimList,
    type: DType,
    static_data_layout: Image2DLayout,
    init_fn: fn () -> SIMD[type, 1],
    update_fn: fn (SIMD[type, 1], SIMD[type, 1]) -> SIMD[type, 1],
    reduce_fn: fn (SIMD[type, 1], Int) -> SIMD[type, 1],
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
        alias simd_width = simdwidthof[type]()
        alias unroll_factor = 8

        let num_threads = out_chain.get_runtime().parallelism_level()
        let num_tasks = min(
            div_ceil(output.num_elements(), min_task_num_slices), num_threads
        )

        let work = output.num_elements()
        let work_block_size = div_ceil(work, num_tasks)

        @always_inline
        @parameter
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
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                var values = SIMD[type, simd_width](0)

                for i in range(simd_width):
                    # Compute the result value at this specific output position.
                    values[i] = pool2d._compute_point(offset + idx + i)

                # Store the computed output at the given output position.
                pool2d.output.data.flatten().simd_store[simd_width](
                    offset + idx,
                    values,
                )

            vectorize_unroll[simd_width, unroll_factor, func_wrapper](
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
            pad_h: Padding on the height dimension with assumed
              tuple def (pad_lower, pad_lower).
            pad_w: Padding on the width dimension with assumed
              tuple def (pad_lower, pad_lower).
            filter_shape: Filter size on height and width
              dimensions with assumed tuple def (filter_h, filter_w).
            stride: Strides on height and width dimensions
              with assumed tuple def (stride_h, stride_w).
            dilation: Dilations on height and width dimensions
              with assumed tuple def (dilation_h, dilation_w).
            num_tasks: Number of tasks to run in parallel.

        Returns:
            An instance of the pooling operator with the input and output buffers
            registered.
        """
        # Register input/output buffers and parameters.
        return Self {
            output: output,
            input: input,
            pad_h: pad_h,
            pad_w: pad_w,
            filter_shape: filter_shape,
            stride: stride,
            dilation: dilation,
            num_tasks: num_tasks,
            # Derive layout agnostic shape information.
            output_shape: ImageShape.__init__[
                static_output_shape, type, static_data_layout
            ](output),
            input_shape: ImageShape.__init__[
                static_input_shape, type, static_data_layout
            ](input),
        }

    fn _compute_point(self, idx: Int) -> SIMD[type, 1]:
        """Implementation of the inner loop computation of a pooling operator
        producing a single scalar value at the given output tensor index.

        Args:
            idx: Flat index specifying which value of the output tensor to
              produce.
        """
        let output_idx = self.output.get_tuple_index(idx)

        # Initialize the result of this point.
        var value: SIMD[type, 1] = init_fn()

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
fn max_pool_init_fn[type: DType]() -> SIMD[type, 1]:
    return neginf[type]()


@always_inline
fn max_pool_update_fn[
    type: DType
](a: SIMD[type, 1], b: SIMD[type, 1]) -> SIMD[type, 1]:
    return max(a, b)


@always_inline
fn max_pool_reduce_fn[type: DType](a: SIMD[type, 1], s: Int) -> SIMD[type, 1]:
    return a


@always_inline
fn avg_pool_init_fn[type: DType]() -> SIMD[type, 1]:
    return 0


@always_inline
fn avg_pool_update_fn[
    type: DType
](a: SIMD[type, 1], b: SIMD[type, 1]) -> SIMD[type, 1]:
    return add(a, b)


@always_inline
fn avg_pool_reduce_fn[type: DType](a: SIMD[type, 1], s: Int) -> SIMD[type, 1]:
    return a / s
