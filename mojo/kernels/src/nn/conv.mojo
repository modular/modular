# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Index import Index, StaticIntTuple
from Int import Int
from Bool import Bool
from Buffer import (
    NDBuffer,
    Buffer,
    _raw_stack_allocation,
    partial_simd_load,
    partial_simd_store,
    _compute_ndbuffer_offset,
)
from SIMD import SIMD
from Assert import assert_param, debug_assert
from Matmul import (
    GemmShape,
    MatmulInnerLoopBPacked,
    calculate_tile_n_k,
    PackMatrixCols,
    PackMatrixRows,
)
from List import create_kgen_list_unknown, create_kgen_list
from Range import range
from TargetInfo import simd_byte_width
from Pointer import DTypePointer
from DType import DType

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
        assert_param[static_layout == Conv2DLayout.unknown]()
        var image: ImageData[shape, type, static_layout]
        image.data = data
        image.dynamic_layout = layout
        return image

    fn __new__(
        data: NDBuffer[4, shape, type]
    ) -> ImageData[shape, type, static_layout]:
        assert_param[static_layout != Conv2DLayout.unknown]()
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
        assert_param[static_layout == Conv2DLayout.unknown]()
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


struct ConvShape:
    """A shape struct describing the convolution dimensions"""

    # Input dimensions.
    var n: Int  # Input batch size.
    var h: Int  # Input height.
    var w: Int  # Input width.
    var c: Int  # Input channel count.
    var out_h: Int  # Output height.
    var out_w: Int  # Output width.

    # Filter dimensions.
    var f: Int  # Filter count.
    var r: Int  # Filter height.
    var s: Int  # Filter width.
    # Convolution parameters.
    var stride: StaticIntTuple[2]  # Stride on [H, W]
    var dilation: StaticIntTuple[2]  # Dilation on [H, W]
    var pad_h: StaticIntTuple[2]  # Padding on H dimension in (Low, High)
    var pad_w: StaticIntTuple[2]  # Padding on W dimension in (Low, High)


@interface
fn get_conv2d_shape[
    output_shape: __mlir_type[`!kgen.list<index[4]>`],
    input_shape: __mlir_type[`!kgen.list<index[4]>`],
    filter_shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    data_layout: __mlir_type.index,
    filter_layout: __mlir_type.index,
](
    # TODO: support static layout too.
    output: ImageData[output_shape, type, Conv2DLayout.unknown],
    input: ImageData[input_shape, type, Conv2DLayout.unknown],
    filter: ImageData[filter_shape, type, Conv2DLayout.unknown],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    ...


@implements(get_conv2d_shape)
fn get_conv2d_shape_nchw[
    output_shape: __mlir_type[`!kgen.list<index[4]>`],
    input_shape: __mlir_type[`!kgen.list<index[4]>`],
    filter_shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    data_layout: __mlir_type.index,
    filter_layout: __mlir_type.index,
](
    output: ImageData[output_shape, type, Conv2DLayout.unknown],
    input: ImageData[input_shape, type, Conv2DLayout.unknown],
    filter: ImageData[filter_shape, type, Conv2DLayout.unknown],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Conv2DLayout.NCHW]()
    assert_param[filter_layout == Conv2DLayout.NCHW]()

    return ConvShape {
        n: input.data.dim[0](),
        h: input.data.dim[2](),
        w: input.data.dim[3](),
        c: input.data.dim[1](),
        out_h: output.data.dim[2](),
        out_w: output.data.dim[3](),
        f: filter.data.dim[0](),
        r: filter.data.dim[2](),
        s: filter.data.dim[3](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


@implements(get_conv2d_shape)
fn get_conv2d_shape_nhwc[
    output_shape: __mlir_type[`!kgen.list<index[4]>`],
    input_shape: __mlir_type[`!kgen.list<index[4]>`],
    filter_shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    data_layout: __mlir_type.index,
    filter_layout: __mlir_type.index,
](
    output: ImageData[output_shape, type, Conv2DLayout.unknown],
    input: ImageData[input_shape, type, Conv2DLayout.unknown],
    filter: ImageData[filter_shape, type, Conv2DLayout.unknown],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Conv2DLayout.NHWC]()
    assert_param[filter_layout == Conv2DLayout.NHWC]()

    return ConvShape {
        n: input.data.dim[0](),
        h: input.data.dim[1](),
        w: input.data.dim[2](),
        c: input.data.dim[3](),
        out_h: output.data.dim[1](),
        out_w: output.data.dim[2](),
        f: filter.data.dim[0](),
        r: filter.data.dim[1](),
        s: filter.data.dim[2](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


@implements(get_conv2d_shape)
fn get_conv2d_shape_nhwc_rscf[
    output_shape: __mlir_type[`!kgen.list<index[4]>`],
    input_shape: __mlir_type[`!kgen.list<index[4]>`],
    filter_shape: __mlir_type[`!kgen.list<index[4]>`],
    type: __mlir_type.`!kgen.dtype`,
    data_layout: __mlir_type.index,
    filter_layout: __mlir_type.index,
](
    output: ImageData[output_shape, type, Conv2DLayout.unknown],
    input: ImageData[input_shape, type, Conv2DLayout.unknown],
    filter: ImageData[filter_shape, type, Conv2DLayout.unknown],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Conv2DLayout.NHWC]()
    assert_param[filter_layout == Conv2DLayout.RSCF]()

    return ConvShape {
        n: input.data.dim[0](),
        h: input.data.dim[1](),
        w: input.data.dim[2](),
        c: input.data.dim[3](),
        out_h: output.data.dim[1](),
        out_w: output.data.dim[2](),
        f: filter.data.dim[3](),
        r: filter.data.dim[0](),
        s: filter.data.dim[1](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


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
        # Iterate on output batch dimension.
        for no_idx in range(self.output_shape.N):
            # Iterate on filter dimension.
            for f_idx in range(self.output_shape.C):
                # Iterate on output H dimension.
                for ho_idx in range(self.output_shape.H):
                    # Iterate on output W dimension.
                    for wo_idx in range(self.output_shape.W):
                        # Compute the result value at this specific output posit-
                        #  ion.
                        self._compute_point(
                            Index(no_idx, f_idx, ho_idx, wo_idx)
                        )

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

        # Iterate on filter height dimension.
        for r_idx in range(self.filter_shape.H):
            # Iterate on filter width dimension.
            for s_idx in range(self.filter_shape.W):
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
                    # Iterate on channels dimension.
                    for c_idx in range(self.input_shape.C):
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

        # Store the computed output at the given output position..
        self.output.__setitem__(
            output_idx.__getitem__[0](),
            output_idx.__getitem__[1](),
            output_idx.__getitem__[2](),
            output_idx.__getitem__[3](),
            value,
        )


# ===----------------------------------------------------------------------=== #
# Im2Col convolution:
# ===----------------------------------------------------------------------=== #


fn conv_one_dimensional_padding(
    filter_shape: Int, is_same_padding: Bool
) -> StaticIntTuple[2]:
    """Computes the amount of padding needed on one dimension based on given
    convolution parameters.
        Args:
            filter_shape (Int): size of filter on the given dimension.
            is_same_padding(Int): apply same padding if true.

        TODO: Supporting heavy right only, need to extend padding scheme
                support.
        TODO: Unify the convolution struct interface.
        TODO: Add stride and dilation.
    """
    if is_same_padding:
        let pad_light = (filter_shape - 1) // 2
        let pad_heavy = filter_shape - 1 - pad_light
        # Same-heavy-right padding.
        return Index(pad_light, pad_heavy)
    # Valid padding.
    return Index(0, 0)


struct PackIm2ColNCHW[
    # original matrix shape list
    static_original_shape: __mlir_type[`!kgen.list<index[4]>`],
    # packed matrix shape list
    static_packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    col_inner_size: __mlir_type.index,
]:
    """
    Im2Col transform outputing mlas's packed data layout, with indexing
     based on output indices.
    i.e. [CRS][Yout] to [Yout_o][CRS][Yout_i]
    In NCHW layout, the Im2Col mapping is K = CRS, M = F, N = HoWo, this is in
     fact a bmm.
    """

    var packed_matrix: NDBuffer[3, static_packed_shape, type]
    var origin_image: NDBuffer[4, static_original_shape, type]
    # Convolution parameters
    var conv_shape: ConvShape
    # offset based on [CRS, HoWo] or [K, N] coordinates.
    var global_offset: StaticIntTuple[2]
    # The matmul packing tile dimension. [TileK, TileN] for Im2Col.
    var pack_tile_kn_dim: StaticIntTuple[2]
    # Batch index, currently not wrapping anyting across batch boundary.
    var batch_idx: Int

    # Im2col output shape [Ho, Wo].
    var image_output_shape: StaticIntTuple[2]
    # global im2col matrix size [CRS, HoWo]
    var im2col_output_shape: StaticIntTuple[2]
    # Padding parameters for (H, W) dimensions.
    var pad_low: StaticIntTuple[2]
    var pad_high: StaticIntTuple[2]

    @staticmethod
    fn run(
        origin_image: NDBuffer[4, static_original_shape, type],
        packed_matrix: NDBuffer[3, static_packed_shape, type],
        conv_shape: ConvShape,
        global_offset: StaticIntTuple[2],
        pack_tile_kn_dim: StaticIntTuple[2],
        batch_idx: Int,
    ):
        let im2col = PackIm2ColNCHW[
            static_original_shape,
            static_packed_shape,
            type,
            simd_size,
            col_inner_size,
        ](
            origin_image,
            packed_matrix,
            conv_shape,
            global_offset,
            pack_tile_kn_dim,
            batch_idx,
        )

        im2col._pack()

    fn _pack(self):
        """Pack the whole k,n tile."""
        # Process the output tile row by row.
        for k_idx in range(self.pack_tile_kn_dim.__getitem__[0]()):
            # Check that the current output row corresponds to valid data.
            if k_idx < (
                self.im2col_output_shape.__getitem__[0]()
                - self.global_offset.__getitem__[0]()
            ):
                # Pack data input of current row if data is valid.
                self._pack_output_row(k_idx)
            else:
                # Fill all zeros for out of bound rows.
                self._pack_zeros_for_k(k_idx)

    fn __new__(
        origin_image: NDBuffer[4, static_original_shape, type],
        packed_matrix: NDBuffer[3, static_packed_shape, type],
        conv_shape: ConvShape,
        global_offset: StaticIntTuple[2],
        pack_tile_kn_dim: StaticIntTuple[2],
        batch_idx: Int,
    ) -> PackIm2ColNCHW[
        static_original_shape,
        static_packed_shape,
        type,
        simd_size,
        col_inner_size,
    ]:
        """Constructor of a Im2Col instance.
        Args:
            origin_image (NDBuffer): The tensor to be transformed.
            packed_matrix (NDBuffer): The transformed tensor in packed
                layout.
            conv_shape (ConvShape): The convolution shape struct.
            global_offset (StaticIntTuple): The (Row, Column) offset for
                indexing the original image tensor.
            pack_tile_kn_dim (StaticIntTuple): The packed tile size on the
                K and N dimension for packed layout.
            batch_idx (Int): The batch index in the original tensor.
        """
        var pack: PackIm2ColNCHW[
            static_original_shape,
            static_packed_shape,
            type,
            simd_size,
            col_inner_size,
        ]

        pack.origin_image = origin_image
        pack.packed_matrix = packed_matrix
        pack.conv_shape = conv_shape
        pack.global_offset = global_offset
        pack.pack_tile_kn_dim = pack_tile_kn_dim
        pack.batch_idx = batch_idx

        # TODO:
        #  Assuming same-padding and stride-1, so output shape and input shape
        # are the same. Will extend this.
        let image_output_shape = Index(conv_shape.h, conv_shape.w)
        pack.image_output_shape = image_output_shape
        pack.im2col_output_shape = Index(
            conv_shape.c * conv_shape.r * conv_shape.s,
            image_output_shape.__getitem__[0]()
            * image_output_shape.__getitem__[1](),
        )

        # TODO:
        # Assuming all heavy right padding  Will extend this.
        let pad_h = conv_one_dimensional_padding(
            conv_shape.r,
            # is_same_padding
            True,
        )
        let pad_w = conv_one_dimensional_padding(
            conv_shape.s,
            # is_same_padding
            True,
        )
        pack.pad_low = Index(pad_h.__getitem__[0](), pad_w.__getitem__[0]())
        pack.pad_high = Index(pad_h.__getitem__[1](), pad_h.__getitem__[1]())

        return pack

    fn _output_to_input(
        self, out_image_idx: StaticIntTuple[2], rs_idx: StaticIntTuple[2]
    ) -> StaticIntTuple[2]:
        """Utility function to convert from convolution output (Ho, Wo) index to
        the corresponding input index (Hi, Wi).
            Args:
                out_image_idx (StaticIntTuple): The output index.
                rs_idx (StaticIntTuple): The filter access index in (r,s).
            Returns:
                The input image index in (H, W).
        """
        # [Ho, Wo] -> [Hi, Wi]
        return out_image_idx - self.pad_low + rs_idx

    # Pack straight block_size of data from original image without checking.
    fn _process_contiguous_blocks[
        block_size: __mlir_type.index,
        fill_zero: Bool,
    ](
        self,
        c_idx: Int,
        rs_idx: StaticIntTuple[2],
        local_tile_nk_offset: StaticIntTuple[2],
        global_out_image_offset: StaticIntTuple[2],
    ):
        """Packing utility: pack block_size of data from original tensor to the
        packed output without any boundary check.
            Args:
                block_size (index): Amount of data to proceed.
                fill_zero: Will fill zero to target position if true, i.e. instead
                    of reading value from original tensor.
                c_idx (Int): The input channel dimension index.
                rs_index(StaticIntTuple): The filter access index, in (r,s).
                local_tile_nk_offset(StaticIntTuple): The output tile access
                    index in (n, k).
                global_out_image_offset(StaticIntTuple): The global offset to
                    access the original tensor in (Ho, Wo).
        """
        # Makes sure the block size is vector compatible.

        # TODO: add this.
        # assert_param[is_disivible_by[block_size, simd_size]()]

        # Convert output index to input index.
        var global_in_image_offset = self._output_to_input(
            global_out_image_offset, rs_idx
        )

        for col_idx in range(0, block_size, simd_size):
            # calculate input index
            let global_in_image_idx = global_in_image_offset + Index(0, col_idx)

            # Load a vector of image data or fill zero.
            var image_data: SIMD[simd_size, type]

            if fill_zero:
                image_data = SIMD[simd_size, type](0)
            else:
                image_data = self.origin_image.simd_load[simd_size](
                    # Indexing for nchw layout.
                    Index(
                        self.batch_idx,
                        c_idx,
                        global_in_image_idx.__getitem__[0](),
                        global_in_image_idx.__getitem__[1](),
                    ).as_tuple()
                )

            # Store image data to the corresponding output index.
            self._pack_vector(
                local_tile_nk_offset + Index(col_idx, 0), image_data
            )

    # Write a simd vector into packed layout buffer.
    @always_inline
    fn _pack_vector(
        self, nk_idx: StaticIntTuple[2], vec_data: SIMD[simd_size, type]
    ):
        """Utility to write a simd vector into the corresponding position in
        packed layout.
            Args:
                nk_idx (StaticIntTuple): The output tile index in (n, k).
                vec_data (SIMD): The data vector to store.
        """
        # Calculate index in packed layout.
        let out_n_idx = nk_idx.__getitem__[0]()
        let out_n_outerIdx = out_n_idx // col_inner_size
        let out_n_innerIdx = Int.remu(out_n_idx, col_inner_size)
        let out_k_idx = nk_idx.__getitem__[1]()

        # Store the simd vector.
        self.packed_matrix.simd_store[simd_size](
            Index(out_n_outerIdx, out_k_idx, out_n_innerIdx).as_tuple(),
            vec_data,
        )

    fn _process_single_simd_vector(
        self,
        # Input channel index
        c_idx: Int,
        rs_idx: StaticIntTuple[2],
        # Offset within tile. [N, K]
        local_tile_nk_offset: StaticIntTuple[2],
        # Offset on whole image [Ho, Wo].
        global_out_image_offset: StaticIntTuple[2],
    ):
        """Fills a simd vector of image data, handling padding and wrap around.
        Args:
            c_idx (Int): Input channel index.
            rs_idx(StaticIntTuple): Filter access indices in (r,s).
            local_tile_nk_offset(StaticIntTuple): Output offset in (n, k).
            global_out_image_offset(StaticIntTuple): Global offset in
                convolution output in (Ho, Wo).

        Returns:
            The next output index position in (ho, wo).
        """
        let vector = Buffer[
            simd_size,
            type,
        ].stack_allocation()

        # Initialize data with zero
        vector.simd_store[simd_size](0, SIMD[simd_size, type](0))

        # calculate h and w output indices.
        var h_o_idx = global_out_image_offset.__getitem__[0]()
        var w_o_idx = global_out_image_offset.__getitem__[1]()

        # Vector index for filling the simd elements.
        for vec_idx in range(simd_size):
            # Calculate the current output and input indices.
            let o_image_idx = Index(h_o_idx, w_o_idx)
            let i_image_idx = self._output_to_input(o_image_idx, rs_idx)

            var element = SIMD[1, type](0)
            if self._is_valid_input_imageIdx(i_image_idx):
                # within valid bound, load data.
                element = self.origin_image.__getitem__(
                    # [N,C,H,W]
                    Index(
                        self.batch_idx,
                        c_idx,
                        i_image_idx.__getitem__[0](),
                        i_image_idx.__getitem__[1](),
                    ).as_tuple()
                )

            vector.__setitem__(vec_idx, element)

            # Increment row index
            w_o_idx += 1

            # Increment h if w reaches bound and wrap around w.
            if w_o_idx == self.image_output_shape.__getitem__[1]():
                w_o_idx = 0
                h_o_idx += 1

        # Load the prepared data into simd vector.
        let vec_data = vector.simd_load[simd_size](0)

        # Write the simd vector into the packed matrix.
        self._pack_vector(local_tile_nk_offset, vec_data)

    fn _pack_zeros_for_k(self, k_idx: Int):
        """Fills zero for the given k index on the packed output.
        Args:
            k_idx (Int): The k index to fill zero at.
        """
        for n_idx in range(
            0, self.pack_tile_kn_dim.__getitem__[1](), col_inner_size
        ):
            self._process_contiguous_blocks[
                # block_size:
                col_inner_size,
                # fill_zero
                True,
            ](
                # c_idx (ignored)
                Int(0),
                # rs_idx (ignored)
                Index(0, 0),
                # local_tile_offset
                Index(n_idx, k_idx),
                # global read offset (ignored)
                Index(0, 0),
            )

    fn _n_to_ho_wo(self, n_idx: Int) -> StaticIntTuple[2]:
        """Map output n index to conv output index in (Ho, Wo).

        Args:
            n_idx (Int): The n index in packed output.
        Returns:
            The output index in (Ho, Wo).
        """
        return Index(
            n_idx // self.image_output_shape.__getitem__[1](),
            Int.remu(n_idx, self.image_output_shape.__getitem__[1]()),
        )

    fn _k_to_c_r_s(self, k_idx: Int) -> StaticIntTuple[3]:
        """Map the packed k index to conv input index in (c,r,s).
        Args:
            k_idx (Int): The transformed k index.
        Returns:
            The pre-transformed index in (c,r,s).
        """
        let shape_rs = self.conv_shape.r * self.conv_shape.s
        let c = k_idx // shape_rs
        let rs = Int.remu(k_idx, shape_rs)
        let r = rs // self.conv_shape.s
        let s = Int.remu(rs, self.conv_shape.s)
        return Index(c, r, s)

    @always_inline
    fn _is_valid_input_imageIdx(self, hw: StaticIntTuple[2]) -> Bool:
        """Checks if the input index is within valid bound.
        Args:
            hw (StaticIntTuple): input index in (h, w).
        Returns:
            True if the given index corresponds to valid data.
        """
        return hw >= Index(0, 0) and hw < Index(
            self.conv_shape.h, self.conv_shape.w
        )

    fn _process_output_row_helper[
        block_size: __mlir_type.index
    ](
        self,
        n_idx: Int,
        k_idx: Int,
        crs: StaticIntTuple[3],
        o_image_ho_wo: StaticIntTuple[2],
    ) -> Bool:
        """Helper to try to progress by block_size in a single Row.
        Args:
            block_size: The contiguous block size to process with.
            n_idx (Int): The local index on n dimension within output tile.
            k_idx (Int): The local index on k dimension within output tile.
            crs (StaticIntTuple): The filter access index in (c, r, s).
            o_image_ho_wo (StaticIntTuple): The output image indices in
                (Ho, Wo).
        Returns:
            True if the block was processed.
        """
        # Unpack the tile index.
        let tile_n = self.pack_tile_kn_dim.__getitem__[1]()

        # Unpack the filter access indices
        let rs_idx = Index(crs.__getitem__[1](), crs.__getitem__[2]())

        # Convert the output index to input index.
        let i_image_h_w = self._output_to_input(o_image_ho_wo, rs_idx)

        # Check that:
        #  1. The output tile index is within valid bound.
        #  2. The starting point of the contiguous block is within input valid
        #   bound.
        #  3. The end point of the contiguous block is within output valid
        #   bound.
        if (
            n_idx <= (tile_n - block_size)
            and self._is_valid_input_imageIdx(i_image_h_w)
            and self._is_valid_input_imageIdx(
                i_image_h_w + Index(0, block_size - 1)
            )
            and (
                (o_image_ho_wo.__getitem__[1]() + block_size - 1)
                < self.conv_shape.out_w
            )
        ):
            self._process_contiguous_blocks[
                block_size,
                # fill_zero
                False,
            ](
                # c_idx,
                crs.__getitem__[0](),
                # rs_idx
                rs_idx,
                # local_tile_nk_offset
                Index(n_idx, k_idx),
                # global_out_image_offset
                o_image_ho_wo,
            )
            return True
        return False

    # Output layout is [K, N] or [CRS, HoWo]
    fn _pack_output_row(self, k_idx: Int):
        """Process a row of the output tile. The output tile is a [K, N] tile of
        The im2col tensor with original global layout in [CRS, HoWo].
            Args:
                k_idx (Int): The index of the row to process.
        """
        # Total number of data needed in the packed layout.
        #  assumed to be multiple of col_inner_size.
        let tile_n = self.pack_tile_kn_dim.__getitem__[1]()
        # unpack crs coordinates.
        let crs = self._k_to_c_r_s(k_idx + self.global_offset.__getitem__[0]())

        var n_idx: Int = 0
        while n_idx < tile_n:
            # Map local tile index to global output matrix index.
            var global_k_n_idx = self.global_offset + Index(k_idx, n_idx)
            # Map output matrix index to output convolution image index.
            var o_image_ho_wo = self._n_to_ho_wo(
                global_k_n_idx.__getitem__[1]()
            )
            # Map output convolution image index to input convolution image
            #  index.
            var i_image_h_w = self._output_to_input(
                o_image_ho_wo, Index(crs.__getitem__[1](), crs.__getitem__[2]())
            )

            # starting from a valid data point:
            # TODO: Also try to speed up zero padding.
            # TODO: handle dilation.
            if self._is_valid_input_imageIdx(i_image_h_w):
                if self._process_output_row_helper[4 * col_inner_size](
                    n_idx, k_idx, crs, o_image_ho_wo
                ):
                    n_idx += 4 * col_inner_size
                    continue
                elif self._process_output_row_helper[2 * col_inner_size](
                    n_idx, k_idx, crs, o_image_ho_wo
                ):
                    n_idx += 2 * col_inner_size
                    continue
                elif self._process_output_row_helper[1 * col_inner_size](
                    n_idx, k_idx, crs, o_image_ho_wo
                ):
                    n_idx += 1 * col_inner_size
                    continue

            # Fall back path, try to proceed by 1 SIMD vector at a time.
            self._process_single_simd_vector(
                # c_idx
                crs.__getitem__[0](),
                # rs_idx
                Index(crs.__getitem__[1](), crs.__getitem__[2]()),
                # local_tile_nk_offset
                Index(n_idx, k_idx),
                # global_out_image_offset
                o_image_ho_wo,
            )
            n_idx += simd_size


# TODO (Fixel): This class has massive code duplication with matmul kernels.
#  Could drastically clean up when non-inlined closure is supported or without
#   language support the conv op and matmul op should share a "gemm skeleton"
#   library to de-duplicate.
struct ConvIm2ColNCHW[
    shape_input: __mlir_type[`!kgen.list<index[4]>`],
    shape_filter: __mlir_type[`!kgen.list<index[4]>`],
    shape_output: __mlir_type[`!kgen.list<index[4]>`],
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    a_row_size: __mlir_type.index,
    pack_inner_size: __mlir_type.index,
    pack_cache_size: __mlir_type.index,
]:
    var out: NDBuffer[4, shape_output, type]
    var input: NDBuffer[4, shape_input, type]
    var filter: NDBuffer[4, shape_filter, type]

    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]
    var gemm_shape: GemmShape
    var conv_shape: ConvShape

    # The current active batch index.
    var batch_idx: Int

    # Temporary buffer for the implicit matmul calls.
    var c: NDBuffer[2, create_kgen_list_unknown[2](), type]

    # 2D view of the filter as implicit matmul input.
    var a: NDBuffer[2, create_kgen_list_unknown[2](), type]

    # Interface method
    @staticmethod
    fn run(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
    ):
        """Interface function to run im2col convolution on the given images and
        filters.
        Args:
            out(NDBuffer): Pre-allocated output space.
            input(NDBuffer): The input to the convolution op.
            filter(NDBuffer): The filter to convolve the input with.
            conv_shape: Struct describing the convolution dimensions.
        """
        # TODO: Support stride and dilation.
        # debug_assertions on the runtime shapes to guard against
        #  unsupported cases yet.
        # Assert same padding.
        debug_assert(
            conv_shape.out_h == conv_shape.h,
            "output height must be the same as the input height",
        )
        debug_assert(
            conv_shape.out_w == conv_shape.w,
            "output width must be the same as the input width",
        )
        # Assert unit stride and padding.
        debug_assert(conv_shape.stride == Index(1, 1), "stride must be [1,1]")
        debug_assert(
            conv_shape.dilation == Index(1, 1), "dilation must be [1,1]"
        )

        var conv = ConvIm2ColNCHW[
            shape_input,
            shape_filter,
            shape_output,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            pack_cache_size,
        ](out, input, filter, conv_shape)

        conv._run_implicit_matmul()

    fn __new__(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
    ) -> ConvIm2ColNCHW[
        shape_input,
        shape_filter,
        shape_output,
        packed_shape,
        type,
        simd_size,
        a_row_size,
        pack_inner_size,
        pack_cache_size,
    ]:
        """Constructor of an instance of the im2col conv2d operator.

        Args:
            out(NDBuffer): Pre-allocated output space.
            input(NDBuffer): The input to the convolution op.
            filter(NDBuffer): The filter to convolve the input with.
            conv_shape: Struct describing the convolution dimensions.
        """
        var conv: ConvIm2ColNCHW[
            shape_input,
            shape_filter,
            shape_output,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            pack_cache_size,
        ]

        conv.out = out
        conv.input = input
        conv.filter = filter
        conv.conv_shape = conv_shape

        # Translate conv shape to gemm shape for computation mapping.
        let gemm_shape = GemmShape {
            M: conv_shape.f,
            N: (conv_shape.out_h * conv_shape.out_w),
            K: (conv_shape.r * conv_shape.s * conv_shape.c),
        }

        conv.gemm_shape = gemm_shape
        conv.tile_n_k = calculate_tile_n_k[pack_cache_size, pack_inner_size](
            gemm_shape
        )

        return conv

    fn _run_implicit_matmul(self&):
        """Wrapper utility funciton: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        # Allocate buffer to pack transformed image.
        var _bpacked_data = _raw_stack_allocation[
            pack_cache_size,  # Count.
            type,  # Data type.
            simd_byte_width().__as_mlir_index(),  # Alignment.
        ]()

        var b_packed = NDBuffer[3, packed_shape, type](_bpacked_data.address)
        # Manually set the shape of packed B buffer:
        let mapped_bpacked = self._view_buffer_as(
            b_packed,
            self.tile_n_k.__getitem__[0](),
            self.tile_n_k.__getitem__[1](),
            pack_inner_size,
        )

        self.batch_idx = 0
        while self.batch_idx < self.conv_shape.n:
            # Generate buffer view of the output matrix.
            self._initialize_buffer_view()
            self._outer_k_loop(mapped_bpacked)
            self.batch_idx += 1

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(self&, b_packed: NDBuffer[3, packed_shape, type]):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        let tile_k = self.tile_n_k.__getitem__[1]()
        let valid_k_count = self.gemm_shape.K
        var k_idx: Int = 0

        # Iterate on batch dimension:

        # Proceed with the largest K tile until crossing
        #  valid boundary.
        while k_idx <= (valid_k_count - tile_k):
            self._outer_n_loop(b_packed, GemmShape(0, 0, k_idx), tile_k)
            k_idx += tile_k

        # Launch another k tile to clean up the residue:
        let remaining_k = valid_k_count - k_idx

        # Do a residue tile if original gemm shape K is not
        #  a multiple of tile K.
        if remaining_k > 0:
            # TODO: possibly need to re-adjust N tile here, if the
            #  residue K is small then could use L2 cache better by
            #  having a wider N.
            self._outer_n_loop(b_packed, GemmShape(0, 0, k_idx), remaining_k)

    fn _outer_n_loop(
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_k: Int,
    ):
        """Iterate on the N dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        let valid_col_count: Int = self.gemm_shape.N - global_offset.N
        let tile_n: Int = self.tile_n_k.__getitem__[0]()

        # Remap buffer indices for current tile.
        var remapped_bpacked = self._view_buffer_as(
            b_packed, tile_n, sub_tile_k, Int(pack_inner_size)
        )

        var col_idx: Int = 0
        # Proceed with the large tile:
        col_idx = self._outer_n_loop_helper[pack_inner_size](
            remapped_bpacked,
            global_offset,
            tile_n,
            sub_tile_k,
            col_idx,
            valid_col_count,
        )

        # Cover residual tiles.
        if col_idx < valid_col_count:
            remapped_bpacked = self._view_buffer_as(
                b_packed, simd_size, sub_tile_k, simd_size
            )
            col_idx = self._outer_n_loop_helper[simd_size](
                remapped_bpacked,
                global_offset,
                simd_size,
                sub_tile_k,
                col_idx,
                valid_col_count,
            )

        # Cover the last sub simdsize tile:
        # This call will handle the sub-simd size boundary.
        if col_idx < valid_col_count:
            self._outer_m_loop[simd_size](
                remapped_bpacked,
                global_offset + GemmShape(0, col_idx, 0),
                simd_size,
                sub_tile_k,
            )

    fn _outer_n_loop_helper[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
        start_idx: Int,
        valid_col_count: Int,
    ) -> Int:
        """Helper function: Iterate on the N dimension by steps of size
            sub_tile_n without crossing valid boundary.

        Args:
            m_loop_pack_inner_size(index): Inner dimension of the packed data
                layout.
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_n(Int): Dynamic tile size to use on N dimension.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
            start_idx(Int): Starting index on N dimension.
            valid_col_count(Int): Number of valid columns remaining on the
                current processing tile.
        """
        var col_idx = start_idx
        while col_idx <= (valid_col_count - sub_tile_n):
            self._outer_m_loop[m_loop_pack_inner_size](
                b_packed,
                global_offset + GemmShape(0, col_idx, 0),
                sub_tile_n,
                sub_tile_k,
            )
            col_idx += sub_tile_n
        return col_idx

    fn _outer_m_loop[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
    ):
        """Pack a subtile of B and iterate through all the rows
        of C.

        Args:
            m_loop_pack_inner_size(index): Inner dimension of the packed data
                layout.
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_n(Int): Dynamic tile size to use on N dimension.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        let valid_col_count = self.c.dim[1]() - global_offset.N

        # The whole subtile is within valid columns,
        #  no need to check boundary when loading C
        #  on this tile.
        # TODO: this could be in finer granularity.
        if valid_col_count >= sub_tile_n:
            self._outer_m_loop_helper[
                # skip_col_bound
                True,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)
        else:
            self._outer_m_loop_helper[
                # skip_col_bound
                False,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)

    fn _outer_m_loop_helper[
        skip_col_bound: Bool, m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
    ):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

            Args:
                skip_col_bound(i1): Column dimension boundary check will be
                    statically skipped if true.
                m_loop_pack_inner_size(index): Inner dimension of the packed data
                    layout.
                b_packed(NDBuffer): B matrix in packed layout.
                global_offset(GemmShape): 3D global offset within the whole
                    matmul problem space.
                sub_tile_n(Int): Dynamic tile size to use on N dimension.
                sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        # pack B:
        PackIm2ColNCHW[
            shape_input,
            packed_shape,
            type,
            simd_size,
            pack_inner_size,
        ].run(
            # original_image
            self.input,
            # packed_matrix
            b_packed,
            self.conv_shape,
            # global offset in (K, N)
            Index(global_offset.K, global_offset.N),
            # packed tile size.
            Index(sub_tile_k, sub_tile_n),
            self.batch_idx,
        )

        # Launch the MLoop
        let sub_tile_n_k = Index(sub_tile_n, sub_tile_k)
        let valid_row_count = self.c.dim[0]() - global_offset.M

        # Launch largest row blocks possible and
        #  then reduce row size to maximizing unrolled tiles.
        var row_idx: Int = 0
        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, a_row_size
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 4
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 3
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 2
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 1
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

    fn _outer_m_loop_row_helper[
        skip_col_bound: Bool,
        m_loop_pack_inner_size: __mlir_type.index,
        RowSize: __mlir_type.index,
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n_k: StaticIntTuple[2],
        start_idx: Int,
        valid_row_count: Int,
    ) -> Int:
        """
        Helper function: Process blocks of rows of the gemm space with the given
          RowBlock size until the given row block does not completely fit in
          valid operand bound.

            Args:
                skip_col_bound(i1): Column dimension boundary check will be
                    statically skipped if true.
                m_loop_pack_inner_size(index): Inner dimension of the packed data
                    layout.
                RowSize(index): Size of row blocks to proceed with on the tile.
                b_packed(NDBuffer): B matrix in packed layout.
                global_offset(GemmShape): 3D global offset within the whole
                    matmul problem space.
                sub_tile_n_k(StaticTuple): Dynamic tile size to use, in
                    (TileN, TileK).
                start_idx(Int): row idx to start from.
                valid_row_count(Int): number of valid rows to process from the
                    start_idx.
        """
        var row_idx = start_idx
        while row_idx <= (valid_row_count - RowSize):
            MatmulInnerLoopBPacked[
                create_kgen_list_unknown[2](),  # shape_a
                create_kgen_list_unknown[2](),  # shape c
                packed_shape,  # packed_shape
                type,  # accum_type
                type,  # value_type
                simd_size,
                RowSize,
                m_loop_pack_inner_size,
                skip_col_bound,
            ].run(
                self.c,
                self.a,
                b_packed,
                global_offset + GemmShape(row_idx, 0, 0),
                sub_tile_n_k,
            )
            row_idx += RowSize
        return row_idx

    fn _initialize_buffer_view(self&):
        """Initializes the internal gemm operand tensors with the translated
        dynamic gemm shapes from convolution shapes.
        """
        # Ouput shape [N, F, Ho, Wo]
        let c_pointer = self.out._offset(
            Index(self.batch_idx, 0, 0, 0).as_tuple()
        )
        self.c = NDBuffer[2, create_kgen_list_unknown[2](), type](
            c_pointer.address
        )
        self.c.dynamic_shape.__setitem__[0](self.conv_shape.f.__as_mlir_index())
        self.c.dynamic_shape.__setitem__[1](
            (self.conv_shape.out_h * self.conv_shape.out_w).__as_mlir_index()
        )

        # Create 2D view for filter.
        self.a = NDBuffer[2, create_kgen_list_unknown[2](), type](
            self.filter.data.address
        )
        self.a.dynamic_shape.__setitem__[0](self.gemm_shape.M.__as_mlir_index())
        self.a.dynamic_shape.__setitem__[1](self.gemm_shape.K.__as_mlir_index())

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[3, packed_shape, type]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

            Args:
                b_packed(NDBuffer): B matrix in packed layout.
                tile_n(Int): Dynamic tile size to use on N dimension.
                tile_k(Int): Dynamic tile size to use on K dimension.
                n_inner_size(Int): Inner dimension size to use for the packed
                    data layout.
        """
        var new_b_packed = b_packed
        let n_outer = tile_n // n_inner_size
        new_b_packed.dynamic_shape.__setitem__[0](n_outer.__as_mlir_index())
        new_b_packed.dynamic_shape.__setitem__[1](tile_k.__as_mlir_index())
        new_b_packed.dynamic_shape.__setitem__[2](
            n_inner_size.__as_mlir_index()
        )
        return new_b_packed


# TODO (Fixel): This class has massive code duplication with matmul kernels.
#  Could drastically clean up when non-inlined closure is supported or without
#   language support the conv op and matmul op should share a "gemm skeleton"
#   library to de-duplicate.
struct ConvNHWCInnerLoopFilterPacked[
    shape_input: __mlir_type[`!kgen.list<index[4]>`],
    shape_c: __mlir_type[`!kgen.list<index[2]>`],
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    accum_type: __mlir_type.`!kgen.dtype`,
    value_type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    a_row_size: __mlir_type.index,
    pack_inner_size: __mlir_type.index,
    skip_boundary_check: Bool,
    same_channel_index: Bool,
]:
    """Inner loop implementation for mlas-style tiled inner loops for a NHWC
    im2col convolution. Accumulates a gemm tile of input defined by (M, N, K) of
    (a_row_size, TileN, TileK).
    """

    # Parameters for global reference.
    var c: NDBuffer[2, shape_c, accum_type]
    var input: NDBuffer[4, shape_input, value_type]
    var b_packed: NDBuffer[3, packed_shape, value_type]
    # 3D global offset within the whole matmul problem space.
    var global_offset: GemmShape
    # Dynamic tiling parameter for this inner loop
    #  in (TileN, TileK).
    var tile_n_k: StaticIntTuple[2]
    # Boundary of valid output space within the
    #  local tile, in (a_row_size, TileN).
    var c_bound: StaticIntTuple[2]

    # Convolution shape parameters.
    var conv_shape: ConvShape

    # Table of pointers for all the rows.
    var offset_table: Buffer[a_row_size, DType.index.value]

    var input_base_pointer: DTypePointer[value_type]

    @staticmethod
    fn run(
        c: NDBuffer[2, shape_c, accum_type],
        input: NDBuffer[4, shape_input, value_type],
        b_packed: NDBuffer[3, packed_shape, value_type],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
        conv_shape: ConvShape,
    ):
        """Interface function to run the packing routine.
        Args:
            c(NDBuffer): pre-allocated buffer space for packed result.
            a(NDBuffer): data buffer operand A.
            b(NDBuffer): data buffer operand B in packed layout.
            global_offset(StaticIntTuple): offset to use when indexing the
                original matrix.
            tile_n_k(StaticIntTuple): 2D dimension tuple describing the
                size of the packed tile of B.
        """
        let offset_table = Buffer[
            a_row_size, DType.index.value
        ].stack_allocation()
        var instance = ConvNHWCInnerLoopFilterPacked[
            shape_input,
            shape_c,
            packed_shape,
            accum_type,
            value_type,
            simd_size,
            a_row_size,
            pack_inner_size,
            skip_boundary_check,
            same_channel_index,
        ] {
            c: c,
            input: input,
            b_packed: b_packed,
            global_offset: global_offset,
            tile_n_k: tile_n_k,
            c_bound: (
                Index(c.dim[0](), c.dim[1]())
                - Index(global_offset.M, global_offset.N)
            ),
            conv_shape: conv_shape,
            offset_table: offset_table,
            input_base_pointer: input.data,
        }

        instance._run_inner_loop()

    fn _initialize_offset_table(self):
        let k_offset = self.global_offset.K
        let r_s_c = _k_to_r_s_c_nhwc(k_offset, self.conv_shape)
        let r_s = Index(r_s_c.__getitem__[0](), r_s_c.__getitem__[1]())

        for row_idx in range(a_row_size):
            let m_offset = self.global_offset.M + row_idx
            let n_ho_wo = _m_to_n_ho_wo_nhwc(m_offset, self.conv_shape)
            let ho_wo = Index(
                n_ho_wo.__getitem__[1](), n_ho_wo.__getitem__[2]()
            )
            let hi_wi = _ho_wo_to_hi_wi(ho_wo, r_s, self.conv_shape)

            let linear_offset = _compute_ndbuffer_offset(
                self.input,
                Index(
                    n_ho_wo.__getitem__[0](),
                    hi_wi.__getitem__[0](),
                    hi_wi.__getitem__[1](),
                    r_s_c.__getitem__[2](),
                ).as_tuple(),
            )

            if hi_wi >= Index(0, 0) and hi_wi < Index(
                self.conv_shape.h, self.conv_shape.w
            ):
                self.offset_table.__setitem__(
                    row_idx, linear_offset.__as_mlir_index()
                )
            else:
                self.offset_table.__setitem__(row_idx, -1)

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            accum_type,
        ],
    ):
        """Utility funcion on the inner loop. Initializes a local c buffer with
        all zeros.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
        """
        for row_idx in range(a_row_size):
            for col_idx in range(0, pack_inner_size, simd_size):
                c_local.simd_store[simd_size](
                    Index(row_idx, col_idx).as_tuple(),
                    SIMD[simd_size, accum_type](0),
                )

    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            accum_type,
        ],
        # indexing within tile, in (m,n)
        tile_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Loads a local c_buffer with the
        value stored in the output buffer space, given the indices within the
        tile being processed.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
                tile_idx(StaticIntTuple): index tuple with (m,n) coordinates
                    within the current processing tile.
        """
        for row_idx in range(a_row_size):
            for col_idx in range(0, pack_inner_size, simd_size):
                let global_idx_pair = (
                    Index(self.global_offset.M, self.global_offset.N)
                    + tile_idx
                    + Index(row_idx, col_idx)
                )
                let global_idx = Index(
                    global_idx_pair.__getitem__[0](),
                    global_idx_pair.__getitem__[1](),
                ).as_tuple()
                let local_idx = Index(row_idx, col_idx).as_tuple()

                # Load data from original matrix C.
                var c_data: SIMD[simd_size, accum_type] = 0
                if skip_boundary_check or (
                    Index(row_idx, col_idx + simd_size)
                    <= (self.c_bound - tile_idx)
                ):
                    # Use simd load if all within bound
                    c_data = self.c.simd_load[simd_size](global_idx)
                elif (
                    row_idx + tile_idx.__getitem__[0]()
                ) < self.c_bound.__getitem__[0]():
                    # Use partial load if row inbound but col not
                    #  in simd bound.
                    c_data = partial_simd_load[simd_size, accum_type](
                        self.c._offset(global_idx),
                        0,
                        self.c_bound.__getitem__[1]()
                        - tile_idx.__getitem__[1]()
                        - col_idx,
                        0,
                    )
                else:
                    # Fill zero if row out of bound
                    c_data = SIMD[simd_size, accum_type](0)

                # Store data to local buffer.
                c_local.simd_store[simd_size](local_idx, c_data)

    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            accum_type,
        ],
        tile_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Stores the value of a local c
        buffer to the corresponding position in the output buffer space.

            Args:
                c_local(NDBuffer): pre-allocated local buffer for c partial
                    sums.
                tile_idx(StaticIntTuple): index tuple with (m,n) coordinates
                    within the current processing tile.
        """
        for row_idx in range(a_row_size):
            for col_idx in range(0, pack_inner_size, simd_size):
                let global_idx_pair = (
                    Index(self.global_offset.M, self.global_offset.N)
                    + tile_idx
                    + Index(row_idx, col_idx)
                )
                let global_idx = Index(
                    global_idx_pair.__getitem__[0](),
                    global_idx_pair.__getitem__[1](),
                ).as_tuple()
                let local_idx = Index(row_idx, col_idx).as_tuple()

                # Load data from original matrix C.
                var c_data = c_local.simd_load[simd_size](local_idx)

                if skip_boundary_check or (
                    Index(row_idx, col_idx + simd_size)
                    <= (self.c_bound - tile_idx)
                ):
                    # Use simd store if all within bound
                    self.c.simd_store[simd_size](global_idx, c_data)
                elif row_idx < (
                    self.c_bound.__getitem__[0]() - tile_idx.__getitem__[0]()
                ):
                    # Use partial store if row in bound but col not
                    #  in simd bound.
                    partial_simd_store[simd_size, accum_type](
                        self.c._offset(global_idx),
                        0,
                        self.c_bound.__getitem__[1]()
                        - tile_idx.__getitem__[1]()
                        - col_idx,
                        c_data,
                    )

    # TODO: This can be lifted to common utility.

    @always_inline
    fn _load_a(
        self, index_m_k: StaticIntTuple[2], row_idx: Int
    ) -> SIMD[1, value_type]:
        """Utility to load one value of Im2col transformed matrix from the
        pre-transformed image.
            Args:
                index_m_k (StaticIntTuple): Index into the post im2col operandA
                    in (M, K) format.
            Returns (SIMD):
                Value loaded from the translated address of image input.
        """
        if same_channel_index:
            let linear_offset: Int = Int(
                self.offset_table.__getitem__(row_idx).value
            )
            if linear_offset == -1:
                return SIMD[1, value_type](0)
            else:
                self.offset_table.__setitem__(
                    row_idx, (linear_offset + 1).__as_mlir_index()
                )
                return self.input_base_pointer.load(linear_offset)
        else:
            let n_ho_wo = _m_to_n_ho_wo_nhwc(
                index_m_k.__getitem__[0](), self.conv_shape
            )
            let rsc = _k_to_r_s_c_nhwc(
                index_m_k.__getitem__[1](), self.conv_shape
            )
            let hi_wi = _ho_wo_to_hi_wi(
                Index(n_ho_wo.__getitem__[1](), n_ho_wo.__getitem__[2]()),
                Index(rsc.__getitem__[0](), rsc.__getitem__[1]()),
                self.conv_shape,
            )

            if hi_wi < Index(
                self.conv_shape.h, self.conv_shape.w
            ) and hi_wi >= Index(0, 0):
                return self.input.simd_load[1](
                    Index(
                        # N
                        n_ho_wo.__getitem__[0](),
                        # H
                        hi_wi.__getitem__[0](),
                        # W
                        hi_wi.__getitem__[1](),
                        # C
                        rsc.__getitem__[2](),
                    ).as_tuple()
                )

        return SIMD[1, value_type](0)

    fn _accumulate(
        self,
        c_local: NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            accum_type,
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility funcion on the inner loop. Launch one tile of fma on the
        local accumulation buffer.

        Args:
            c_local(NDBuffer): pre-allocated local buffer for c partial
                sums.
            tile_n_k_idx(StaticIntTuple): index tuple with (n, k)
                coordinates within the current processing tile to index the
                packed B matrix.
        """
        # Seek outer indices in packed layout.
        let n_outer_idx = tile_n_k_idx.__getitem__[0]() // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx.__getitem__[1]()

        let local_a = Buffer[
            simd_size * a_row_size,
            value_type,
        ].stack_allocation()

        for fill_a_idx in range(a_row_size):
            var global_m = self.global_offset.M + fill_a_idx
            let a_val_scalar = self._load_a(
                Index(global_m, global_k).as_tuple(), fill_a_idx
            )
            let a_fill_val = SIMD[simd_size, value_type](a_val_scalar)
            local_a.simd_store[simd_size](fill_a_idx * simd_size, a_fill_val)

        # Loop over local accumulator tiles.
        for col_idx in range(0, pack_inner_size, simd_size):
            let b_val = self.b_packed.simd_load[simd_size](
                Index(
                    n_outer_idx, tile_n_k_idx.__getitem__[1](), col_idx
                ).as_tuple()
            ).cast[accum_type]()
            for row_idx in range(a_row_size):
                let a_val = local_a.simd_load[simd_size](
                    row_idx * simd_size
                ).cast[accum_type]()
                var c_idx = Index(row_idx, col_idx).as_tuple()
                var c_val = c_local.simd_load[simd_size](c_idx)

                c_val = a_val.fma(b_val, c_val)
                c_local.simd_store[simd_size](c_idx, c_val)

    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            2,
            create_kgen_list[__mlir_type.index](
                a_row_size, pack_inner_size * simd_size
            ),
            accum_type,
        ].stack_allocation()

        for idx_n in range(0, self.tile_n_k.__getitem__[0](), pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, Index(0, idx_n))

            # Iterate on tile K dimension.
            # Allocate offset table.
            if same_channel_index:
                self._initialize_offset_table()

            # Not unrolled on K path.
            for idx_k in range(self.tile_n_k.__getitem__[1]()):
                # accumulate data for this (n, k) index
                self._accumulate(c_local, Index(idx_n, idx_k))

            self._store_c_tile(c_local, Index(0, idx_n))


@always_inline
fn _m_to_n_ho_wo_nhwc(m: Int, conv_shape: ConvShape) -> StaticIntTuple[3]:
    """Converts post-im2col m dimension index to pre-im2col coordinates on
    (N, Hout, Wout) dimensions.
        Args:
            m (Int): Index on M dimension.
            conv_shape (ConvShape): convolution dimension description.

        Returns (StaticIntTuple):
            The translated 3d indices in (N, Hout, Wout) format.
    TODO(Fixel): This utilty should be generalized into a im2col util
    class with some additional layout agnostic logic.
    """
    let n = m // (conv_shape.out_h * conv_shape.out_w)
    let ho = (
        Int.remu(m, conv_shape.out_h * conv_shape.out_w) // conv_shape.out_w
    )
    let wo = Int.remu(m, conv_shape.out_w)
    return Index(n, ho, wo)


fn _k_to_r_s_c_nhwc(k: Int, conv_shape: ConvShape) -> StaticIntTuple[3]:
    """Converts post-im2col k dimension index to pre-im2col coordinates on
    (R, S, C) dimensions.
        Args:
            m (Int): Index on M dimension.
            conv_shape (ConvShape): convolution dimension description.

        Returns (StaticIntTuple):
            The translated 3d indices in (R, S, C) format.
    TODO(Fixel): This utilty should be generalized into a im2col util
    class with some additional layout agnostic logic.
    """
    let r = k // (conv_shape.s * conv_shape.c)
    let s = Int.remu(k // conv_shape.c, conv_shape.s)
    let c = Int.remu(k, conv_shape.c)
    return Index(r, s, c)


fn _ho_wo_to_hi_wi(
    ho_wo: StaticIntTuple[2],
    r_s: StaticIntTuple[2],
    conv_shape: ConvShape,
) -> StaticIntTuple[2]:
    """Converts index on the output images to index on the input images.
        Args:
            ho_wo (StaticIntTuple): Index on output image in (Hout, Wout).
            r_s (StaticIntTuple): Index on filter dimensions in (R, S).
            conv_shape (ConvShape): convolution dimension description.

        Returns (StaticIntTuple):
            Index on input image in (H, W).

        Returns (StaticIntTuple):
            The translated 3d indices in (R, S, C) format.
    TODO(Fixel): This utilty should be generalized into a conv util
    class with some additional layout agnostic logic.
    """
    let pad_left = Index(
        conv_shape.pad_h.__getitem__[0](),
        conv_shape.pad_w.__getitem__[0](),
    )
    return ho_wo * conv_shape.stride - pad_left + r_s * conv_shape.dilation


# TODO (Fixel): This class has massive code duplication with matmul kernels.
#  Could drastically clean up when non-inlined closure is supported or without
#   language support the conv op and matmul op should share a "gemm skeleton"
#   library to de-duplicate.
struct ConvIm2ColNHWC[
    shape_input: __mlir_type[`!kgen.list<index[4]>`],
    shape_filter: __mlir_type[`!kgen.list<index[4]>`],
    shape_output: __mlir_type[`!kgen.list<index[4]>`],
    packed_shape: __mlir_type[`!kgen.list<index[3]>`],
    type: __mlir_type.`!kgen.dtype`,
    simd_size: __mlir_type.index,
    a_row_size: __mlir_type.index,
    pack_inner_size: __mlir_type.index,
    pack_cache_size: __mlir_type.index,
    filter_layout: __mlir_type.index,
]:
    var out: NDBuffer[4, shape_output, type]
    var input: NDBuffer[4, shape_input, type]
    var filter: NDBuffer[4, shape_filter, type]

    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]
    var gemm_shape: GemmShape
    var conv_shape: ConvShape

    # The current active batch index.
    var batch_idx: Int

    # 2D view of the tensors as implicit matmul input.
    var c: NDBuffer[2, create_kgen_list_unknown[2](), type]
    var a: NDBuffer[2, create_kgen_list_unknown[2](), type]
    var b: NDBuffer[2, create_kgen_list_unknown[2](), type]

    # Interface method
    @staticmethod
    fn run(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
    ):
        """Interface function to run im2col convolution on the given images and
        filters.
        Args:
            out(NDBuffer): Pre-allocated output space.
            input(NDBuffer): The input to the convolution op.
            filter(NDBuffer): The filter to convolve the input with.
            conv_shape: Struct describing the convolution dimensions.
        """
        # Assert on supported convolution dimensions.

        var conv = ConvIm2ColNHWC[
            shape_input,
            shape_filter,
            shape_output,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            pack_cache_size,
            filter_layout,
        ](out, input, filter, conv_shape)

        conv._run_implicit_matmul()

    fn __new__(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
    ) -> ConvIm2ColNHWC[
        shape_input,
        shape_filter,
        shape_output,
        packed_shape,
        type,
        simd_size,
        a_row_size,
        pack_inner_size,
        pack_cache_size,
        filter_layout,
    ]:
        """Constructor of an instance of the im2col conv2d operator.

        Args:
            out(NDBuffer): Pre-allocated output space.
            input(NDBuffer): The input to the convolution op.
            filter(NDBuffer): The filter to convolve the input with.
            conv_shape: Struct describing the convolution dimensions.
        """
        var conv: ConvIm2ColNHWC[
            shape_input,
            shape_filter,
            shape_output,
            packed_shape,
            type,
            simd_size,
            a_row_size,
            pack_inner_size,
            pack_cache_size,
            filter_layout,
        ]

        conv.out = out
        conv.input = input
        conv.filter = filter
        conv.conv_shape = conv_shape

        # Translate conv shape to gemm shape for computation mapping.
        #  For NHWC data layout.
        let gemm_shape = GemmShape {
            M: (conv_shape.n * conv_shape.out_h * conv_shape.out_w),
            N: (conv_shape.f),
            K: (conv_shape.r * conv_shape.s * conv_shape.c),
        }

        conv.gemm_shape = gemm_shape
        conv.tile_n_k = calculate_tile_n_k[pack_cache_size, pack_inner_size](
            GemmShape(
                gemm_shape.M,
                gemm_shape.N,
                conv_shape.c,  # Do not yet pack more than C.
            )
        )

        return conv

    fn _run_implicit_matmul(self&):
        """Wrapper utility funciton: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        # Allocate buffer to pack transformed image.
        var _bpacked_data = _raw_stack_allocation[
            pack_cache_size,  # Count.
            type,  # Data type.
            simd_byte_width().__as_mlir_index(),  # Alignment.
        ]()

        var b_packed = NDBuffer[3, packed_shape, type](_bpacked_data.address)
        # Manually set the shape of packed B buffer:
        let mapped_bpacked = self._view_buffer_as(
            b_packed,
            self.tile_n_k.__getitem__[0](),
            self.tile_n_k.__getitem__[1](),
            pack_inner_size,
        )

        self._initialize_buffer_view()
        self._outer_k_loop(mapped_bpacked)

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(self&, b_packed: NDBuffer[3, packed_shape, type]):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        let tile_k = self.tile_n_k.__getitem__[1]()
        let valid_k_count = self.gemm_shape.K
        var k_idx: Int = 0

        # Assume starting at multiple of channel count always.
        # TODO: relax this assumption for perf coverage.
        while k_idx < valid_k_count:
            var processed_k: Int = 0
            # Proceed with the largest K tile until crossing
            #  valid boundary.
            while processed_k <= (self.conv_shape.c - tile_k):
                self._outer_n_loop(b_packed, GemmShape(0, 0, k_idx), tile_k)
                k_idx += tile_k
                processed_k += tile_k

            # Launch another k tile to clean up the residue to the c boundary.
            let remaining_k = self.conv_shape.c - processed_k

            # Do a residue tile if original gemm shape K is not
            #  a multiple of tile K.
            if remaining_k > 0:
                # TODO: possibly need to re-adjust N tile here, if the
                #  residue K is small then could use L2 cache better by
                #  having a wider N.
                self._outer_n_loop(
                    b_packed, GemmShape(0, 0, k_idx), remaining_k
                )
                k_idx += remaining_k

    fn _outer_n_loop(
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_k: Int,
    ):
        """Iterate on the N dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        let valid_col_count: Int = self.gemm_shape.N - global_offset.N
        let tile_n: Int = self.tile_n_k.__getitem__[0]()

        # Remap buffer indices for current tile.
        var remapped_bpacked = self._view_buffer_as(
            b_packed, tile_n, sub_tile_k, Int(pack_inner_size)
        )

        var col_idx: Int = 0
        # Proceed with the large tile:
        col_idx = self._outer_n_loop_helper[pack_inner_size](
            remapped_bpacked,
            global_offset,
            tile_n,
            sub_tile_k,
            col_idx,
            valid_col_count,
        )

        # Cover residual tiles.
        if col_idx < valid_col_count:
            remapped_bpacked = self._view_buffer_as(
                b_packed, simd_size, sub_tile_k, simd_size
            )
            col_idx = self._outer_n_loop_helper[simd_size](
                remapped_bpacked,
                global_offset,
                simd_size,
                sub_tile_k,
                col_idx,
                valid_col_count,
            )

        # Cover the last sub simdsize tile:
        # This call will handle the sub-simd size boundary.
        if col_idx < valid_col_count:
            self._outer_m_loop[simd_size](
                remapped_bpacked,
                global_offset + GemmShape(0, col_idx, 0),
                simd_size,
                sub_tile_k,
            )

    fn _outer_n_loop_helper[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
        start_idx: Int,
        valid_col_count: Int,
    ) -> Int:
        """Helper function: Iterate on the N dimension by steps of size
            sub_tile_n without crossing valid boundary.

        Args:
            m_loop_pack_inner_size(index): Inner dimension of the packed data
                layout.
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_n(Int): Dynamic tile size to use on N dimension.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
            start_idx(Int): Starting index on N dimension.
            valid_col_count(Int): Number of valid columns remaining on the
                current processing tile.
        """
        var col_idx = start_idx
        while col_idx <= (valid_col_count - sub_tile_n):
            self._outer_m_loop[m_loop_pack_inner_size](
                b_packed,
                global_offset + GemmShape(0, col_idx, 0),
                sub_tile_n,
                sub_tile_k,
            )
            col_idx += sub_tile_n
        return col_idx

    fn _outer_m_loop[
        m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
    ):
        """Pack a subtile of B and iterate through all the rows
        of C.

        Args:
            m_loop_pack_inner_size(index): Inner dimension of the packed data
                layout.
            b_packed(NDBuffer): B matrix in packed layout.
            global_offset(GemmShape): 3D global offset within the whole
                matmul problem space.
            sub_tile_n(Int): Dynamic tile size to use on N dimension.
            sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        let valid_col_count = self.c.dim[1]() - global_offset.N

        # The whole subtile is within valid columns,
        #  no need to check boundary when loading C
        #  on this tile.
        # TODO: this could be in finer granularity.
        if valid_col_count >= sub_tile_n:
            self._outer_m_loop_helper[
                # skip_col_bound
                True,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)
        else:
            self._outer_m_loop_helper[
                # skip_col_bound
                False,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)

    fn _outer_m_loop_helper[
        skip_col_bound: Bool, m_loop_pack_inner_size: __mlir_type.index
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
    ):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

            Args:
                skip_col_bound(i1): Column dimension boundary check will be
                    statically skipped if true.
                m_loop_pack_inner_size(index): Inner dimension of the packed data
                    layout.
                b_packed(NDBuffer): B matrix in packed layout.
                global_offset(GemmShape): 3D global offset within the whole
                    matmul problem space.
                sub_tile_n(Int): Dynamic tile size to use on N dimension.
                sub_tile_k(Int): Dynamic tile size to use on K dimension.
        """
        # pack B:
        if filter_layout == Conv2DLayout.NHWC:
            PackMatrixRows[
                create_kgen_list_unknown[2](),
                packed_shape,
                type,
                simd_size,
                m_loop_pack_inner_size,
            ].run(
                b_packed,
                self.b,
                # Input is [N, K]:
                Index(global_offset.N, global_offset.K),
                Index(sub_tile_n, sub_tile_k),
            )
        else:  # TODO: add assert, filter layout should be RSCF.
            PackMatrixCols[
                create_kgen_list_unknown[2](),
                packed_shape,
                type,
                simd_size,
                m_loop_pack_inner_size,
            ].run(
                b_packed,
                self.b,
                # Input is [K, N]:
                Index(global_offset.K, global_offset.N),
                Index(sub_tile_k, sub_tile_n),
            )

        # Launch the MLoop
        let sub_tile_n_k = Index(sub_tile_n, sub_tile_k)
        let valid_row_count = self.c.dim[0]() - global_offset.M

        # Launch largest row blocks possible and
        #  then reduce row size to maximizing unrolled tiles.
        var row_idx: Int = 0
        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, a_row_size
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 4
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 3
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 2
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 1
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_count)

    fn _can_skip_hw_check(
        self, row_start_index: Int, number_of_rows: Int
    ) -> Bool:
        """Checks that the current block of rows are perfectly within image
        boundary so the check on H and W dimensions can be skipped.
        """
        for row_index in range(
            row_start_index, row_start_index + number_of_rows
        ):
            let n_ho_wo = _m_to_n_ho_wo_nhwc(row_index, self.conv_shape)
            let ho_wo = Index(
                n_ho_wo.__getitem__[1](), n_ho_wo.__getitem__[2]()
            )

            let corner_lower_left = _ho_wo_to_hi_wi(
                # Output index.
                ho_wo,
                # Filter offset r,s.
                Index(0, 0),
                self.conv_shape,
            )
            let corner_upper_right = _ho_wo_to_hi_wi(
                # Output index.
                ho_wo,
                # Filter offset r,s.
                Index(self.conv_shape.r - 1, self.conv_shape.s - 1),
                self.conv_shape,
            )

            if corner_lower_left >= Index(0, 0) and corner_upper_right < Index(
                self.conv_shape.h, self.conv_shape.w
            ):
                # Continue only if this is within boundary.
                continue
            return False
        # All rows are checked in the group.
        return True

    fn _outer_m_loop_row_helper[
        skip_col_bound: Bool,
        m_loop_pack_inner_size: __mlir_type.index,
        RowSize: __mlir_type.index,
    ](
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        global_offset: GemmShape,
        sub_tile_n_k: StaticIntTuple[2],
        start_idx: Int,
        valid_row_count: Int,
    ) -> Int:
        """
        Helper function: Process blocks of rows of the gemm space with the given
          RowBlock size until the given row block does not completely fit in
          valid operand bound.

            Args:
                skip_col_bound(i1): Column dimension boundary check will be
                    statically skipped if true.
                m_loop_pack_inner_size(index): Inner dimension of the packed data
                    layout.
                RowSize(index): Size of row blocks to proceed with on the tile.
                b_packed(NDBuffer): B matrix in packed layout.
                global_offset(GemmShape): 3D global offset within the whole
                    matmul problem space.
                sub_tile_n_k(StaticTuple): Dynamic tile size to use, in
                    (TileN, TileK).
                start_idx(Int): row idx to start from.
                valid_row_count(Int): number of valid rows to process from the
                    start_idx.
        """
        var row_idx = start_idx
        while row_idx <= (valid_row_count - RowSize):
            let current_offset = global_offset + GemmShape(row_idx, 0, 0)
            ConvNHWCInnerLoopFilterPacked[
                shape_input,
                create_kgen_list_unknown[2](),
                packed_shape,
                type,
                type,
                simd_size,
                RowSize,
                m_loop_pack_inner_size,
                skip_col_bound,
                True,  # same_channel_index
            ].run(
                self.c,
                self.input,
                b_packed,
                current_offset,
                sub_tile_n_k,
                self.conv_shape,
            )
            row_idx += RowSize
        return row_idx

    fn _initialize_buffer_view(self&):
        """Initializes the internal gemm operand tensors with the translated
        dynamic gemm shapes from convolution shapes.
        """
        # Ouput shape [N, F, Ho, Wo]
        let c_pointer = self.out._offset(Index(0, 0, 0, 0).as_tuple())
        self.c = NDBuffer[2, create_kgen_list_unknown[2](), type](
            c_pointer.address
        )
        self.c.dynamic_shape.__setitem__[0](self.gemm_shape.M.__as_mlir_index())
        self.c.dynamic_shape.__setitem__[1](self.gemm_shape.N.__as_mlir_index())

        # Create 2D view for input.
        self.a = NDBuffer[2, create_kgen_list_unknown[2](), type](
            self.input.data.address
        )
        self.a.dynamic_shape.__setitem__[0](self.gemm_shape.M.__as_mlir_index())
        self.a.dynamic_shape.__setitem__[1](self.gemm_shape.K.__as_mlir_index())

        # Create 2D view for filter.
        self.b = NDBuffer[2, create_kgen_list_unknown[2](), type](
            self.filter.data.address
        )
        if filter_layout == Conv2DLayout.NHWC:  # FRSC layout
            self.b.dynamic_shape.__setitem__[0](
                self.gemm_shape.N.__as_mlir_index()
            )
            self.b.dynamic_shape.__setitem__[1](
                self.gemm_shape.K.__as_mlir_index()
            )
        elif filter_layout == Conv2DLayout.RSCF:  # RSCF layout
            self.b.dynamic_shape.__setitem__[0](
                self.gemm_shape.K.__as_mlir_index()
            )
            self.b.dynamic_shape.__setitem__[1](
                self.gemm_shape.N.__as_mlir_index()
            )

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed: NDBuffer[3, packed_shape, type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[3, packed_shape, type]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

            Args:
                b_packed(NDBuffer): B matrix in packed layout.
                tile_n(Int): Dynamic tile size to use on N dimension.
                tile_k(Int): Dynamic tile size to use on K dimension.
                n_inner_size(Int): Inner dimension size to use for the packed
                    data layout.
        """
        var new_b_packed = b_packed
        let n_outer = tile_n // n_inner_size
        new_b_packed.dynamic_shape.__setitem__[0](n_outer.__as_mlir_index())
        new_b_packed.dynamic_shape.__setitem__[1](tile_k.__as_mlir_index())
        new_b_packed.dynamic_shape.__setitem__[2](
            n_inner_size.__as_mlir_index()
        )
        return new_b_packed
