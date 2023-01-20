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
alias Layout_unknown = -1  # statically unknown layout.
alias Layout_NHWC = 0  # channels last layout.
alias Layout_NCHW = 1  # channels first layout.


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
        assert_param[static_layout == Layout_unknown]
        var image: ImageData[shape, type, static_layout]
        image.data = data
        image.dynamic_layout = layout
        return image

    fn __new__(
        data: NDBuffer[4, shape, type]
    ) -> ImageData[shape, type, static_layout]:
        assert_param[static_layout != Layout_unknown]
        var image: ImageData[shape, type, static_layout]
        image.data = data
        return image

    fn getLayout(self) -> Int:
        """The getter function of the underlying data layout, resolving from
        either staticall or dynamicall information.
            Returns:
                The resolved data layout tag for this image instance.
        """
        if static_layout == Layout_unknown:
            return self.dynamic_layout
        return static_layout

    fn _getIndex(self, n: Int, c: Int, h: Int, w: Int) -> StaticIntTuple[4]:
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
        if self.getLayout() == Layout_NCHW:
            return Index(n, c, h, w)
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
        return self.data.__getitem__(self._getIndex(n, c, h, w).as_tuple())

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
        self.data.__setitem__(self._getIndex(n, c, h, w).as_tuple(), value)


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
    ](imageData: ImageData[shape, type, layout]) -> ImageShape:
        """Constructor of an ImageShape instance from an ImageData.

        Args:
            imageData (ImageData): The imageData instance to extract shape
        info from.

        Returns:
            An ImageShape instance containing the shape info.
        """
        var imageShape: ImageShape
        imageShape.N = imageData.data.dim[0]()

        if imageData.getLayout() == Layout_NCHW:
            imageShape.C = imageData.data.dim[1]()
            imageShape.H = imageData.data.dim[2]()
            imageShape.W = imageData.data.dim[3]()

        elif imageData.getLayout() == Layout_NHWC:
            imageShape.C = imageData.data.dim[3]()
            imageShape.H = imageData.data.dim[1]()
            imageShape.W = imageData.data.dim[2]()
        return imageShape


struct Naive2dConvolution[
    ShapeOutput: __mlir_type[`!kgen.list<index[4]>`],
    ShapeFilter: __mlir_type[`!kgen.list<index[4]>`],
    ShapeInput: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    LayoutData: __mlir_type.index,
    LayoutFilter: __mlir_type.index,
]:
    """Struct wrapper for naive 2d convolution implementation."""

    # Input params.
    var output: ImageData[ShapeOutput, type, LayoutData]
    var input: ImageData[ShapeInput, type, LayoutData]
    var filter: ImageData[ShapeFilter, type, LayoutFilter]
    var padH: StaticIntTuple[2]
    var padW: StaticIntTuple[2]
    var stride: StaticIntTuple[2]
    var dilation: StaticIntTuple[2]

    # Derived params.
    var outputShape: ImageShape
    var inputShape: ImageShape
    var filterShape: ImageShape

    @staticmethod
    fn run(
        output: ImageData[ShapeOutput, type, LayoutData],
        input: ImageData[ShapeInput, type, LayoutData],
        filter: ImageData[ShapeFilter, type, LayoutFilter],
        padH: StaticIntTuple[2],
        padW: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
    ):
        """Interface function to run a convolution op on the given input and
        filter tensor and stores the result in the give output tensor.

            Args:
                output(ImageData): Pre-allocated output tensor space.
                input(ImageData): Batched image input to the conv2d operator.
                filter(ImageData): Filters to apply in the conv2d operator.
                padH(StaticIntTuple): Padding on the height dimension with assu-
                    med tuple def (PadOnLowerIdx, PadOnHigherIdx).
                padW(StaticIntTuple): Padding on the width dimension with assum-
                    ed tuple def (PadOnLowerIdx, PadOnHigherIdx).
                stride(StaticIntTuple): Strides on height and width dimensions
                    with assumed tuple def (StrideH, StrideW).
                dilation(StaticIntTuple): Dilations on height and width dimensi-
                    ons with assumed tuple def (dilationH, dilationW).
        """
        # Create an instance of the convolution op.
        var naive2dConvolution = Naive2dConvolution[
            ShapeOutput, ShapeFilter, ShapeInput, type, LayoutData, LayoutFilter
        ](output, input, filter, padH, padW, stride, dilation)

        # Run the actual loops and computations.
        naive2dConvolution._outerLoop()

    fn __new__(
        output: ImageData[ShapeOutput, type, LayoutData],
        input: ImageData[ShapeInput, type, LayoutData],
        filter: ImageData[ShapeFilter, type, LayoutFilter],
        padH: StaticIntTuple[2],
        padW: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
    ) -> Naive2dConvolution[
        ShapeOutput, ShapeFilter, ShapeInput, type, LayoutData, LayoutFilter
    ]:
        """Constructor of a convolution op instance on the given input and
        filter tensor and stores the result in the give output tensor.

            Args:
                output(ImageData): Pre-allocated output tensor space.
                input(ImageData): Batched image input to the conv2d operator.
                filter(ImageData): Filters to apply in the conv2d operator.
                padH(StaticIntTuple): Padding on the height dimension with assu-
                    med tuple def (PadOnLowerIdx, PadOnHigherIdx).
                padW(StaticIntTuple): Padding on the width dimension with assum-
                    ed tuple def (PadOnLowerIdx, PadOnHigherIdx).
                stride(StaticIntTuple): Strides on height and width dimensions
                    with assumed tuple def (StrideH, StrideW).
                dilation(StaticIntTuple): Dilations on height and width dimensi-
                    ons with assumed tuple def (dilationH, dilationW).
            Returns:
                An instance of the convolution operator with the input and outp-
                    ut buffers registered.
        """
        var naive2dConvolution: Naive2dConvolution[
            ShapeOutput, ShapeFilter, ShapeInput, type, LayoutData, LayoutFilter
        ]
        # Register input/output buffers and parameters.
        naive2dConvolution.output = output
        naive2dConvolution.input = input
        naive2dConvolution.filter = filter
        naive2dConvolution.padH = padH
        naive2dConvolution.padW = padW
        naive2dConvolution.stride = stride
        naive2dConvolution.dilation = dilation

        # Derive layout agnostic shape information.
        naive2dConvolution.outputShape = ImageShape.__new__[
            ShapeOutput, type, LayoutData
        ](output)
        naive2dConvolution.inputShape = ImageShape.__new__[
            ShapeInput, type, LayoutData
        ](input)
        naive2dConvolution.filterShape = ImageShape.__new__[
            ShapeFilter, type, LayoutFilter
        ](filter)
        return naive2dConvolution

    fn _outerLoop(self):
        """Implementation of the outermost loop of a convolution operator with
        loops covering the iteration space of batch, filter count, height and wi-
        dth dimensions.
        """
        var nOIdx: Int = 0  # Iterate on output batch dimension.
        while nOIdx < self.outputShape.N:
            var fIdx: Int = 0  # Iterate on filter dimension.
            while fIdx < self.outputShape.C:
                var hoIdx: Int = 0  # Iterate on output H dimension.
                while hoIdx < self.outputShape.H:
                    var woIdx: Int = 0  # Iterate on output W dimension.
                    while woIdx < self.outputShape.W:
                        # Compute the result value at this specific output posit-
                        #  ion.
                        self._computePoint(Index(nOIdx, fIdx, hoIdx, woIdx))
                        woIdx += 1
                    hoIdx += 1
                fIdx += 1
            nOIdx += 1

    fn _computePoint(
        self,
        # Output index [N,C,H,W]
        outputIdx: StaticIntTuple[4],
    ):
        """Implementation of the inner loop computation of a conv2d operator
        producing a single scalar value at the given output tensor index.
            Args:
                outputIndex(StaticIntTuple): Index vector specifying which
            value of the output tensor to produce.
        """
        # Initialize the result of this point.
        var value: SIMD[1, type] = 0

        # Extract the H and W size of the input image.
        let imageBound = Index(self.inputShape.H, self.inputShape.W)

        var rIdx: Int = 0
        while rIdx < self.filterShape.H:  # Iterate on filter height dimension.
            var sIdx: Int = 0
            while (
                sIdx < self.filterShape.W
            ):  # Iterate on filter width dimension.
                # Compute input access index, on the H and W dimension.
                let inputImageIndex = (
                    # Output HxW with striding.
                    (
                        Index(
                            outputIdx.__getitem__[2](),
                            outputIdx.__getitem__[3](),
                        )
                        * self.stride
                    )
                    +
                    # Filter RxS with dilation.
                    (Index(rIdx, sIdx) * self.dilation)
                    -
                    # Padding offset, using the left padding only here.
                    Index(
                        self.padH.__getitem__[0](), self.padW.__getitem__[0]()
                    )
                )

                if (
                    # Check that the current image index is within valid range
                    #  on the input image data tensor.
                    Index(0, 0) <= inputImageIndex
                    and inputImageIndex < imageBound
                ):
                    var cIdx: Int = 0
                    # Iterate on channels dimension.
                    while cIdx < self.inputShape.C:
                        # Accumulate product of input data filter data.
                        value += self.input.__getitem__(
                            outputIdx.__getitem__[0](),  # N
                            cIdx,  # C
                            inputImageIndex.__getitem__[0](),  # H
                            inputImageIndex.__getitem__[1](),  # W
                        ) * self.filter.__getitem__(
                            outputIdx.__getitem__[1](),
                            cIdx,
                            rIdx,
                            sIdx,  # F  # C  # R  # S
                        )
                        cIdx += 1
                sIdx += 1
            rIdx += 1

        # Store the computed output at the given output position..
        self.output.__setitem__(
            outputIdx.__getitem__[0](),
            outputIdx.__getitem__[1](),
            outputIdx.__getitem__[2](),
            outputIdx.__getitem__[3](),
            value,
        )
