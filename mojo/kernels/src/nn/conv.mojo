# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Index import Index, StaticIntTuple
from Int import Int
from Bool import Bool
from Buffer import NDBuffer
from SIMD import SIMD
from Assert import assert_param

# Data layout encoding.
struct Conv2DLayout:
    alias unknown = -1  # statically unknown layout.
    alias NHWC = 0  # channels last layout.
    alias NCHW = 1  # channels first layout.
    alias RSCF = 2  # TF filter layout for channels last input.


struct ImageData[
    shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    static_layout: __mlir_type.index,
]:
    """Utility class that generalizes conv2d data and filter tensor with a given
    data layout."""

    var data: NDBuffer[4, shape, type]
    var dynamic_layout: Int

    fn __new__(
        data: NDBuffer[4, shape, type], layout: Int
    ) -> ImageData[shape, type, static_layout]:
        """Constructor of an image data instance with dynamic layout param.

        Args:
            data(NDBuffer): An 4d buffer containing the actual data.
            layout(Int): Data layout tag.
        Returns:
            An ImageData instance constructed.
        """
        assert_param[static_layout == Conv2DLayout.unknown]
        var image: ImageData[shape, type, static_layout]
        image.data = data
        image.dynamic_layout = layout
        return image

    fn __new__(
        data: NDBuffer[4, shape, type]
    ) -> ImageData[shape, type, static_layout]:
        assert_param[static_layout != Conv2DLayout.unknown]
        var image: ImageData[shape, type, static_layout]
        image.data = data
        return image

    fn to_static_layout[
        new_static_layout: __mlir_type.index
    ](self) -> ImageData[shape, type, new_static_layout]:
        """Conversion utility from a fully dynamic data structure, e.g. from c
        shim to one with compile-time known data layout.
            Returns:
                The image data with static data layout.
        """
        assert_param[static_layout == Conv2DLayout.unknown]
        return ImageData[shape, type, new_static_layout](self.data)

    fn get_layout(self) -> Int:
        """The getter function of the underlying data layout, resolving from
        either staticall or dynamicall information.
            Returns:
                The resolved data layout tag for this image instance.
        """
        if static_layout == Conv2DLayout.unknown:
            return self.dynamic_layout
        return static_layout

    fn _get_index(self, n: Int, c: Int, h: Int, w: Int) -> StaticIntTuple[4]:
        """Converts the general index to the actual index into the underlying
        data based on the tensor layout.

            Args:
                n(Int): Index on the batch dimension.
                c(Int): Index on the channel dimension.
                h(Int): Index on the height dimension.
                w(Int): Index on the width dimension.

            Returns:
                An StaticIntTuple containing the index based on the underlying
            data layout.
        """
        if self.get_layout() == Conv2DLayout.NCHW:
            return Index(n, c, h, w)
        elif self.get_layout() == Conv2DLayout.RSCF:
            return Index(h, w, c, n)
        return Index(n, h, w, c)

    fn __getitem__(self, n: Int, c: Int, h: Int, w: Int) -> SIMD[1, type]:
        """Reads the underlying data buffer based on the tensor index and under-
        lying data layout.

            Args:
                n(Int): Index on the batch dimension.
                c(Int): Index on the channel dimension.
                h(Int): Index on the height dimension.
                w(Int): Index on the width dimension.

            Returns:
                The value stored at the given index position.
        """
        return self.data.__getitem__(self._get_index(n, c, h, w).as_tuple())

    fn __setitem__(self, n: Int, c: Int, h: Int, w: Int, value: SIMD[1, type]):
        """Writes the underlying data buffer based on the tensor index and under-
        lying data layout.

            Args:
                n(Int): Index on the batch dimension.
                c(Int): Index on the channel dimension.
                h(Int): Index on the height dimension.
                w(Int): Index on the width dimension.
                value(SIMD[1]): The value to store at the given index position.
        """
        self.data.__setitem__(self._get_index(n, c, h, w).as_tuple(), value)


struct ImageShape:
    """A data-layout agnostic representation of tensor shapes used in conv2d"""

    var N: Int
    var C: Int
    var H: Int
    var W: Int

    fn __new__[
        shape: __mlir_type[`!kgen.list<index[4]>`],
        type: __mlir_type.`!kgen.dtype`,
        layout: __mlir_type.index,
    ](image_data: ImageData[shape, type, layout]) -> ImageShape:
        """Constructor of an ImageShape instance from an ImageData.

        Args:
            image_data (ImageData): The image_data instance to extract shape
        info from.

        Returns:
            An ImageShape instance containing the shape info.
        """
        var image_shape: ImageShape

        if image_data.get_layout() == Conv2DLayout.NCHW:
            image_shape.N = image_data.data.dim[0]()
            image_shape.C = image_data.data.dim[1]()
            image_shape.H = image_data.data.dim[2]()
            image_shape.W = image_data.data.dim[3]()

        elif image_data.get_layout() == Conv2DLayout.NHWC:
            image_shape.N = image_data.data.dim[0]()
            image_shape.C = image_data.data.dim[3]()
            image_shape.H = image_data.data.dim[1]()
            image_shape.W = image_data.data.dim[2]()

        elif image_data.get_layout() == Conv2DLayout.RSCF:
            image_shape.N = image_data.data.dim[3]()
            image_shape.C = image_data.data.dim[2]()
            image_shape.H = image_data.data.dim[0]()
            image_shape.W = image_data.data.dim[1]()

        return image_shape


struct Naive2dConvolution[
    static_output_shape: __mlir_type[`!kgen.list<index[4]>`],
    static_filter_shape: __mlir_type[`!kgen.list<index[4]>`],
    static_input_shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    static_data_layout: __mlir_type.index,
    static_filter_layout: __mlir_type.index,
]:
    """Struct wrapper for naive 2d convolution implementation."""

    # Input params.
    var output: ImageData[static_output_shape, type, static_data_layout]
    var input: ImageData[static_input_shape, type, static_data_layout]
    var filter: ImageData[static_filter_shape, type, static_filter_layout]
    var pad_h: StaticIntTuple[2]
    var pad_w: StaticIntTuple[2]
    var stride: StaticIntTuple[2]
    var dilation: StaticIntTuple[2]

    # Derived params.
    var output_shape: ImageShape
    var input_shape: ImageShape
    var filter_shape: ImageShape

    @staticmethod
    fn run(
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        filter: ImageData[static_filter_shape, type, static_filter_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
    ):
        """Interface function to run a convolution op on the given input and
        filter tensor and stores the result in the give output tensor.

            Args:
                output(ImageData): Pre-allocated output tensor space.
                input(ImageData): Batched image input to the conv2d operator.
                filter(ImageData): Filters to apply in the conv2d operator.
                pad_h(StaticIntTuple): Padding on the height dimension with assu-
                    med tuple def (PadOnLowerIdx, PadOnHigherIdx).
                pad_w(StaticIntTuple): Padding on the width dimension with assum-
                    ed tuple def (PadOnLowerIdx, PadOnHigherIdx).
                stride(StaticIntTuple): Strides on height and width dimensions
                    with assumed tuple def (StrideH, StrideW).
                dilation(StaticIntTuple): Dilations on height and width dimensi-
                    ons with assumed tuple def (dilation_h, dilation_w).
        """
        # Create an instance of the convolution op.
        var naive2d_convolution = Naive2dConvolution[
            static_output_shape,
            static_filter_shape,
            static_input_shape,
            type,
            static_data_layout,
            static_filter_layout,
        ](output, input, filter, pad_h, pad_w, stride, dilation)

        # Run the actual loops and computations.
        naive2d_convolution._outer_loop()

    fn __new__(
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        filter: ImageData[static_filter_shape, type, static_filter_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
    ) -> Naive2dConvolution[
        static_output_shape,
        static_filter_shape,
        static_input_shape,
        type,
        static_data_layout,
        static_filter_layout,
    ]:
        """Constructor of a convolution op instance on the given input and
        filter tensor and stores the result in the give output tensor.

            Args:
                output(ImageData): Pre-allocated output tensor space.
                input(ImageData): Batched image input to the conv2d operator.
                filter(ImageData): Filters to apply in the conv2d operator.
                pad_h(StaticIntTuple): Padding on the height dimension with assu-
                    med tuple def (PadOnLowerIdx, PadOnHigherIdx).
                pad_w(StaticIntTuple): Padding on the width dimension with assum-
                    ed tuple def (PadOnLowerIdx, PadOnHigherIdx).
                stride(StaticIntTuple): Strides on height and width dimensions
                    with assumed tuple def (StrideH, StrideW).
                dilation(StaticIntTuple): Dilations on height and width dimensi-
                    ons with assumed tuple def (dilation_h, dilation_w).
            Returns:
                An instance of the convolution operator with the input and outp-
                    ut buffers registered.
        """
        var naive2d_convolution: Naive2dConvolution[
            static_output_shape,
            static_filter_shape,
            static_input_shape,
            type,
            static_data_layout,
            static_filter_layout,
        ]
        # Register input/output buffers and parameters.
        naive2d_convolution.output = output
        naive2d_convolution.input = input
        naive2d_convolution.filter = filter
        naive2d_convolution.pad_h = pad_h
        naive2d_convolution.pad_w = pad_w
        naive2d_convolution.stride = stride
        naive2d_convolution.dilation = dilation

        # Derive layout agnostic shape information.
        naive2d_convolution.output_shape = ImageShape.__new__[
            static_output_shape, type, static_data_layout
        ](output)
        naive2d_convolution.input_shape = ImageShape.__new__[
            static_input_shape, type, static_data_layout
        ](input)
        naive2d_convolution.filter_shape = ImageShape.__new__[
            static_filter_shape, type, static_filter_layout
        ](filter)
        return naive2d_convolution

    fn _outer_loop(self):
        """Implementation of the outermost loop of a convolution operator with
        loops covering the iteration space of batch, filter count, height and wi-
        dth dimensions.
        """
        var no_idx: Int = 0  # Iterate on output batch dimension.
        while no_idx < self.output_shape.N:
            var f_idx: Int = 0  # Iterate on filter dimension.
            while f_idx < self.output_shape.C:
                var ho_idx: Int = 0  # Iterate on output H dimension.
                while ho_idx < self.output_shape.H:
                    var wo_idx: Int = 0  # Iterate on output W dimension.
                    while wo_idx < self.output_shape.W:
                        # Compute the result value at this specific output posit-
                        #  ion.
                        self._compute_point(
                            Index(no_idx, f_idx, ho_idx, wo_idx)
                        )
                        wo_idx += 1
                    ho_idx += 1
                f_idx += 1
            no_idx += 1

    fn _compute_point(
        self,
        # Output index [N,C,H,W]
        output_idx: StaticIntTuple[4],
    ):
        """Implementation of the inner loop computation of a conv2d operator
        producing a single scalar value at the given output tensor index.
            Args:
                output_index(StaticIntTuple): Index vector specifying which
            value of the output tensor to produce.
        """
        # Initialize the result of this point.
        var value: SIMD[1, type] = 0

        # Extract the H and W size of the input image.
        let image_bound = Index(self.input_shape.H, self.input_shape.W)

        var r_idx: Int = 0
        while (
            r_idx < self.filter_shape.H
        ):  # Iterate on filter height dimension.
            var s_idx: Int = 0
            while (
                s_idx < self.filter_shape.W
            ):  # Iterate on filter width dimension.
                # Compute input access index, on the H and W dimension.
                let input_image_index = (
                    # Output HxW with striding.
                    (
                        Index(
                            output_idx.__getitem__[2](),
                            output_idx.__getitem__[3](),
                        )
                        * self.stride
                    )
                    +
                    # Filter RxS with dilation.
                    (Index(r_idx, s_idx) * self.dilation)
                    -
                    # Padding offset, using the left padding only here.
                    Index(
                        self.pad_h.__getitem__[0](), self.pad_w.__getitem__[0]()
                    )
                )

                if (
                    # Check that the current image index is within valid range
                    #  on the input image data tensor.
                    Index(0, 0) <= input_image_index
                    and input_image_index < image_bound
                ):
                    var c_idx: Int = 0
                    # Iterate on channels dimension.
                    while c_idx < self.input_shape.C:
                        # Accumulate product of input data filter data.
                        value += self.input.__getitem__(
                            output_idx.__getitem__[0](),  # N
                            c_idx,  # C
                            input_image_index.__getitem__[0](),  # H
                            input_image_index.__getitem__[1](),  # W
                        ) * self.filter.__getitem__(
                            output_idx.__getitem__[1](),
                            c_idx,
                            r_idx,
                            s_idx,  # F  # C  # R  # S
                        )
                        c_idx += 1
                s_idx += 1
            r_idx += 1

        # Store the computed output at the given output position..
        self.output.__setitem__(
            output_idx.__getitem__[0](),
            output_idx.__getitem__[1](),
            output_idx.__getitem__[2](),
            output_idx.__getitem__[3](),
            value,
        )
