# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from Index import Index, StaticIntTuple
from Int import Int
from Buffer import NDBuffer, Buffer
from SIMD import SIMD
from List import create_kgen_list_unknown, create_kgen_list
from Image import (
    ImageData,
    Image2DLayout,
    ImageShape,
)
from Range import range
from List import _get_kgen_list_item
from Math import min, max, add, div_ceil
from Functional import parallelize, vectorize_unroll
from LLCL import Runtime
from Pointer import Pointer
from Pointer import DTypePointer, product as pointer_product
from TargetInfo import dtype_simd_width
from DType import DType
from Numerics import neginf


@register_passable
struct Pool2d[
    static_output_shape: __mlir_type[`!kgen.list<index[4]>`],
    static_kernel_shape: __mlir_type[`!kgen.list<index[2]>`],
    static_input_shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    static_data_layout: Image2DLayout,
    init_fn: __mlir_type[`!kgen.signature<<>() -> `, SIMD[1, type], `>`],
    update_fn: __mlir_type[
        `!kgen.signature<<>(`,
        SIMD[1, type],
        `,`,
        SIMD[1, type],
        `) -> `,
        SIMD[1, type],
        `>`,
    ],
    reduce_fn: __mlir_type[
        `!kgen.signature<<>(`,
        SIMD[1, type],
        `,`,
        Int,
        `) -> `,
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
    var stride: StaticIntTuple[2]
    var dilation: StaticIntTuple[2]
    var num_tasks: Int

    # Derived params.
    var output_shape: ImageShape
    var input_shape: ImageShape

    fn __clone__(self&) -> Self:
        return Self {
            output: self.output,
            input: self.input,
            pad_h: self.pad_h,
            pad_w: self.pad_w,
            stride: self.stride,
            dilation: self.dilation,
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            num_tasks: self.num_tasks,
        }

    @staticmethod
    fn run(
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        runtime: Runtime.ptr_type,
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
            runtime: Runtime.
        """
        # TODO: Find a heuristic to replace the magic numbers.
        alias min_task_num_slices = 64
        alias vector_width = dtype_simd_width[type]()
        alias unroll_factor = 8

        let num_threads = Runtime(runtime).parallelism_level()
        let num_tasks = min(
            div_ceil(output.num_elements(), min_task_num_slices), num_threads
        )

        # Create an instance of the pooling op.
        var pool2d = Pool2d[
            static_output_shape,
            static_kernel_shape,
            static_input_shape,
            type,
            static_data_layout,
            init_fn,
            update_fn,
            reduce_fn,
        ](output, input, pad_h, pad_w, stride, dilation, num_tasks)

        @always_inline
        fn task_func(task_id: Int):

            let work = pool2d.output.num_elements()
            let work_block_size = div_ceil(work, pool2d.num_tasks)
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

        parallelize[task_func](runtime, num_tasks)

    fn __new__(
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        num_tasks: Int,
    ) -> Pool2d[
        static_output_shape,
        static_kernel_shape,
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
            kernel: Kernel size on height and width
              dimensions with assumed tuple def (kernel_h, kernel_w).
            pad_h: Padding on the height dimension with assu-
              med tuple def (pad_lower, pad_lower).
            pad_w: Padding on the width dimension with assum-
              ed tuple def (pad_lower, pad_lower).
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
            static_kernel_shape,
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
        pool2d.stride = stride
        pool2d.dilation = dilation
        pool2d.num_tasks = num_tasks

        # Derive layout agnostic shape information.
        pool2d.output_shape = ImageShape.__new__[
            static_output_shape, type, static_data_layout
        ](output)
        pool2d.input_shape = ImageShape.__new__[
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

        let r_size = Int(
            _get_kgen_list_item[0, 2, __mlir_type.index](static_kernel_shape)
        )
        let s_size = Int(
            _get_kgen_list_item[1, 2, __mlir_type.index](static_kernel_shape)
        )

        # Iterate on filter height dimension.
        for r_idx in range(r_size):
            # Iterate on filter width dimension.
            for s_idx in range(s_size):
                # Compute input access index, on the H and W dimension.
                let input_image_index = (
                    # Output HxW with striding.
                    (
                        StaticIntTuple[2](
                            output_idx[2],
                            output_idx[3],
                        )
                        * self.stride
                    )
                    +
                    # Kernel RxS with dilation.
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

        return reduce_fn(value, r_size * s_size)


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
