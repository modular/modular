# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Index import Index, StaticIntTuple
from Int import Int
from Buffer import (
    NDBuffer,
    Buffer,
    _raw_stack_allocation,
    partial_simd_load,
    partial_simd_store,
    _compute_ndbuffer_offset,
)
from SIMD import SIMD
from Assert import assert_param, assert_param_bool, debug_assert
from Matmul import (
    GemmShape,
    MatmulInnerLoopBPacked,
    calculate_tile_n_k,
    PackMatrixCols,
    PackMatrixRows,
)
from Math import div_ceil, min, max
from List import DimList, create_dim_list
from Range import range
from TargetInfo import simd_byte_width
from Pointer import DTypePointer, Pointer
from DType import DType
from LLCL import Runtime, OutputChainPtr
from Functional import unroll, unroll2, async_parallelize
from Range import range
from Image import (
    ImageData,
    Image2DLayout,
    ImageShape,
)
from IO import _printf


@register_passable("trivial")
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


@adaptive
fn get_conv2d_shape[
    output_shape: DimList[4],
    input_shape: DimList[4],
    filter_shape: DimList[4],
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[4, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param_bool[data_layout == Image2DLayout.NCHW]()
    assert_param_bool[filter_layout == Image2DLayout.NCHW]()

    return ConvShape {
        n: input.dim[0](),
        h: input.dim[2](),
        w: input.dim[3](),
        c: input.dim[1](),
        out_h: output.dim[2](),
        out_w: output.dim[3](),
        f: filter.dim[0](),
        r: filter.dim[2](),
        s: filter.dim[3](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


@adaptive
fn get_conv2d_shape[
    output_shape: DimList[4],
    input_shape: DimList[4],
    filter_shape: DimList[4],
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[4, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param_bool[data_layout == Image2DLayout.NHWC]()
    assert_param_bool[filter_layout == Image2DLayout.NHWC]()

    return ConvShape {
        n: input.dim[0](),
        h: input.dim[1](),
        w: input.dim[2](),
        c: input.dim[3](),
        out_h: output.dim[1](),
        out_w: output.dim[2](),
        f: filter.dim[0](),
        r: filter.dim[1](),
        s: filter.dim[2](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


@adaptive
fn get_conv2d_shape[
    output_shape: DimList[4],
    input_shape: DimList[4],
    filter_shape: DimList[4],
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[4, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param_bool[data_layout == Image2DLayout.NHWC]()
    assert_param_bool[filter_layout == Image2DLayout.RSCF]()

    return ConvShape {
        n: input.dim[0](),
        h: input.dim[1](),
        w: input.dim[2](),
        c: input.dim[3](),
        out_h: output.dim[1](),
        out_w: output.dim[2](),
        f: filter.dim[3](),
        r: filter.dim[0](),
        s: filter.dim[1](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


struct Naive2dConvolution[
    static_output_shape: DimList[4],
    static_filter_shape: DimList[4],
    static_input_shape: DimList[4],
    type: DType,
    static_data_layout: Image2DLayout,
    static_filter_layout: Image2DLayout,
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

    fn __copy__(self) -> Self:
        return Self {
            output: self.output,
            input: self.input,
            filter: self.filter,
            pad_h: self.pad_h,
            pad_w: self.pad_w,
            stride: self.stride,
            dilation: self.dilation,
            output_shape: self.output_shape,
            input_shape: self.input_shape,
            filter_shape: self.filter_shape,
        }

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

    fn __init__(
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
        naive2d_convolution.output_shape = ImageShape.__init__[
            static_output_shape, type, static_data_layout
        ](output)
        naive2d_convolution.input_shape = ImageShape.__init__[
            static_input_shape, type, static_data_layout
        ](input)
        naive2d_convolution.filter_shape = ImageShape.__init__[
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
                            StaticIntTuple[4](no_idx, f_idx, ho_idx, wo_idx)
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
        let image_bound = StaticIntTuple[2](
            self.input_shape.H, self.input_shape.W
        )

        # Iterate on filter height dimension.
        for r_idx in range(self.filter_shape.H):
            # Iterate on filter width dimension.
            for s_idx in range(self.filter_shape.W):
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
                    # Filter RxS with dilation.
                    (Index(r_idx, s_idx) * self.dilation)
                    -
                    # Padding offset, using the left padding only here.
                    Index(self.pad_h[0], self.pad_w[0])
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
                        value += (
                            self.input[
                                output_idx[0],  # N
                                c_idx,  # C
                                input_image_index[0],  # H
                                input_image_index[1],  # W
                            ]
                            * self.filter[
                                output_idx[1],
                                c_idx,
                                r_idx,
                                s_idx,  # F  # C  # R  # S
                            ]
                        )

        # Store the computed output at the given output position..
        self.output[
            output_idx[0], output_idx[1], output_idx[2], output_idx[3]
        ] = value


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
    static_original_shape: DimList[4],
    # packed matrix shape list
    static_packed_shape: DimList[3],
    type: DType,
    simd_size: Int,
    col_inner_size: Int,
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

    fn __copy__(self) -> Self:
        return Self {
            packed_matrix: self.packed_matrix,
            origin_image: self.origin_image,
            conv_shape: self.conv_shape,
            global_offset: self.global_offset,
            pack_tile_kn_dim: self.pack_tile_kn_dim,
            batch_idx: self.batch_idx,
            image_output_shape: self.image_output_shape,
            im2col_output_shape: self.im2col_output_shape,
            pad_low: self.pad_low,
            pad_high: self.pad_high,
        }

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
        for k_idx in range(self.pack_tile_kn_dim[0]):
            # Check that the current output row corresponds to valid data.
            if k_idx < (self.im2col_output_shape[0] - self.global_offset[0]):
                # Pack data input of current row if data is valid.
                self._pack_output_row(k_idx)
            else:
                # Fill all zeros for out of bound rows.
                self._pack_zeros_for_k(k_idx)

    fn __init__(
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
            image_output_shape[0] * image_output_shape[1],
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
        pack.pad_low = Index(pad_h[0], pad_w[0])
        pack.pad_high = Index(pad_h[1], pad_h[1])

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
        block_size: Int,
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

        alias loop_iters = block_size // simd_size

        @always_inline
        fn body[idx: Int]():
            alias col_idx = idx * simd_size
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
                        global_in_image_idx[0],
                        global_in_image_idx[1],
                    )
                )

            # Store image data to the corresponding output index.
            self._pack_vector(
                local_tile_nk_offset + Index(col_idx, 0), image_data
            )

        unroll[loop_iters, body]()

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
        let out_n_idx = nk_idx[0]
        let out_n_outerIdx = out_n_idx // col_inner_size
        let out_n_innerIdx = out_n_idx % col_inner_size
        let out_k_idx = nk_idx[1]

        # Store the simd vector.
        self.packed_matrix.simd_store[simd_size](
            Index(out_n_outerIdx, out_k_idx, out_n_innerIdx),
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
            simd_size.__as_mlir_index(),
            type,
        ].stack_allocation()

        # Initialize data with zero
        vector.simd_store[simd_size](0, SIMD[simd_size, type](0))

        # calculate h and w output indices.
        var h_o_idx = global_out_image_offset[0]
        var w_o_idx = global_out_image_offset[1]

        # Vector index for filling the simd elements.
        @always_inline
        fn body[idx: Int]():
            alias vec_idx = idx
            # Calculate the current output and input indices.
            let o_image_idx = Index(h_o_idx, w_o_idx)
            let i_image_idx = self._output_to_input(o_image_idx, rs_idx)

            var element = SIMD[1, type](0)
            if self._is_valid_input_imageIdx(i_image_idx):
                # within valid bound, load data.
                element = self.origin_image[
                    # [N,C,H,W]
                    Index(
                        self.batch_idx,
                        c_idx,
                        i_image_idx[0],
                        i_image_idx[1],
                    )
                ]

            vector[vec_idx] = element

            # Increment row index
            w_o_idx += 1

            # Increment h if w reaches bound and wrap around w.
            if w_o_idx == self.image_output_shape[1]:
                w_o_idx = 0
                h_o_idx += 1

        unroll[simd_size, body]()

        # Load the prepared data into simd vector.
        let vec_data = vector.simd_load[simd_size](0)

        # Write the simd vector into the packed matrix.
        self._pack_vector(local_tile_nk_offset, vec_data)

    fn _pack_zeros_for_k(self, k_idx: Int):
        """Fills zero for the given k index on the packed output.
        Args:
            k_idx (Int): The k index to fill zero at.
        """
        for n_idx in range(0, self.pack_tile_kn_dim[1], col_inner_size):
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
            n_idx // self.image_output_shape[1],
            n_idx % self.image_output_shape[1],
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
        let rs = k_idx % shape_rs
        let r = rs // self.conv_shape.s
        let s = rs % self.conv_shape.s
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
        block_size: Int
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
        let tile_n = self.pack_tile_kn_dim[1]

        # Unpack the filter access indices
        let rs_idx = Index(crs[1], crs[2])

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
            and ((o_image_ho_wo[1] + block_size - 1) < self.conv_shape.out_w)
        ):
            self._process_contiguous_blocks[
                block_size,
                # fill_zero
                False,
            ](
                # c_idx,
                crs[0],
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
        let tile_n = self.pack_tile_kn_dim[1]
        # unpack crs coordinates.
        let crs = self._k_to_c_r_s(k_idx + self.global_offset[0])

        var n_idx: Int = 0
        while n_idx < tile_n:
            # Map local tile index to global output matrix index.
            var global_k_n_idx = self.global_offset + Index(k_idx, n_idx)
            # Map output matrix index to output convolution image index.
            var o_image_ho_wo = self._n_to_ho_wo(global_k_n_idx[1])
            # Map output convolution image index to input convolution image
            #  index.
            var i_image_h_w = self._output_to_input(
                o_image_ho_wo, Index(crs[1], crs[2])
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
                crs[0],
                # rs_idx
                Index(crs[1], crs[2]),
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
@register_passable("trivial")
struct ConvIm2ColNCHW[
    shape_input: DimList[4],
    shape_filter: DimList[4],
    shape_output: DimList[4],
    packed_shape: DimList[3],
    type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    pack_cache_size: Int,
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
    var c: NDBuffer[2, DimList[2].create_unknown(), type]

    # 2D view of the filter as implicit matmul input.
    var a: NDBuffer[2, DimList[2].create_unknown(), type]

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

    fn __init__(
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
            simd_byte_width(),  # Alignment.
        ]()

        # Manually set the shape of packed B buffer:
        let mapped_bpacked = self._view_buffer_as(
            _bpacked_data.address,
            self.tile_n_k[0],
            self.tile_n_k[1],
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
        let tile_k = self.tile_n_k[1]
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
        let tile_n: Int = self.tile_n_k[0]

        # Remap buffer indices for current tile.
        var remapped_bpacked = self._view_buffer_as(
            b_packed.data, tile_n, sub_tile_k, pack_inner_size
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
                b_packed.data, simd_size, sub_tile_k, simd_size
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
        m_loop_pack_inner_size: Int
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
        m_loop_pack_inner_size: Int,
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
        skip_col_bound: Bool, m_loop_pack_inner_size: Int
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
            skip_col_bound,
            m_loop_pack_inner_size,
            a_row_size.__as_mlir_index(),
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
        m_loop_pack_inner_size: Int,
        RowSize: Int,
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
        alias prefetch_b_distance_k = 4
        var row_idx = start_idx
        while row_idx <= (valid_row_count - RowSize):
            MatmulInnerLoopBPacked[
                DimList[2].create_unknown(),  # shape_a
                DimList[2].create_unknown(),  # shape c
                packed_shape,  # packed_shape
                type,  # accum_type
                type,  # value_type
                simd_size,
                RowSize,
                m_loop_pack_inner_size,
                skip_col_bound,
                prefetch_b_distance_k,  # prefetch distance
            ].run(
                self.c,
                self.a,
                b_packed,
                global_offset + GemmShape(row_idx, 0, 0),
                GemmShape(self.c.dim[0](), self.c.dim[1](), self.a.dim[1]()),
                sub_tile_n_k,
            )
            row_idx += RowSize
        return row_idx

    fn _initialize_buffer_view(self&):
        """Initializes the internal gemm operand tensors with the translated
        dynamic gemm shapes from convolution shapes.
        """
        # Ouput shape [N, F, Ho, Wo]
        let c_pointer = self.out._offset(Index(self.batch_idx, 0, 0, 0))
        self.c = NDBuffer[2, DimList[2].create_unknown(), type](
            c_pointer.address,
            create_dim_list(
                self.conv_shape.f,
                self.conv_shape.out_h * self.conv_shape.out_w,
            ),
            type,
        )

        # Create 2D view for filter.
        self.a = NDBuffer[2, DimList[2].create_unknown(), type](
            self.filter.data.address,
            create_dim_list(
                self.gemm_shape.M,
                self.gemm_shape.K,
            ),
            type,
        )

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed: DTypePointer[type],
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
        return NDBuffer[3, packed_shape, type](
            b_packed.address,
            create_dim_list(
                tile_n // n_inner_size,
                tile_k,
                n_inner_size,
            ),
            type,
        )


# TODO (Fixel): This class has massive code duplication with matmul kernels.
#  Could drastically clean up when non-inlined closure is supported or without
#   language support the conv op and matmul op should share a "gemm skeleton"
#   library to de-duplicate.
struct ConvNHWCInnerLoopFilterPacked[
    shape_input: DimList[4],
    shape_c: DimList[2],
    packed_shape: DimList[3],
    accum_type: DType,
    value_type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
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
    var offset_table: Buffer[a_row_size.__as_mlir_index(), DType.index]

    var input_base_pointer: DTypePointer[value_type]

    fn __copy__(self) -> Self:
        return Self {
            c: self.c,
            input: self.input,
            b_packed: self.b_packed,
            global_offset: self.global_offset,
            tile_n_k: self.tile_n_k,
            c_bound: self.c_bound,
            conv_shape: self.conv_shape,
            offset_table: self.offset_table,
            input_base_pointer: self.input_base_pointer,
        }

    @staticmethod
    fn run(
        c: NDBuffer[2, shape_c, accum_type],
        input: NDBuffer[4, shape_input, value_type],
        b_packed: NDBuffer[3, packed_shape, value_type],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
        conv_shape: ConvShape,
        col_start_idx: Int,
        total_col_count: Int,
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
            a_row_size.__as_mlir_index(), DType.index
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
                Index(c.dim[0](), col_start_idx + total_col_count)
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
        let r_s = Index(r_s_c[0], r_s_c[1])

        @always_inline
        fn body[idx: Int]():
            alias row_idx = idx
            let m_offset = self.global_offset.M + row_idx
            let n_ho_wo = _m_to_n_ho_wo_nhwc(m_offset, self.conv_shape)
            let ho_wo = Index(n_ho_wo[1], n_ho_wo[2])
            let hi_wi = _ho_wo_to_hi_wi(ho_wo, r_s, self.conv_shape)

            let linear_offset = _compute_ndbuffer_offset(
                self.input,
                Index(
                    n_ho_wo[0],
                    hi_wi[0],
                    hi_wi[1],
                    r_s_c[2],
                ),
            )

            if hi_wi >= Index(0, 0) and hi_wi < Index(
                self.conv_shape.h, self.conv_shape.w
            ):
                self.offset_table[row_idx] = linear_offset.__as_mlir_index()
            else:
                self.offset_table[row_idx] = -1

        unroll[a_row_size, body]()

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(
                a_row_size,
                pack_inner_size * simd_size,
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

        @always_inline
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.simd_store[simd_size](
                Index(idx0, idx1 * simd_size),
                SIMD[simd_size, accum_type](0),
            )

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(
                a_row_size,
                pack_inner_size * simd_size,
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

        @always_inline
        fn outer_body[idx0: Int, idx1: Int]():
            alias col_idx = idx1 * simd_size

            let global_idx_pair = (
                Index(self.global_offset.M, self.global_offset.N)
                + tile_idx
                + Index(idx0, col_idx)
            )
            let global_idx = Index(
                global_idx_pair[0],
                global_idx_pair[1],
            )
            let local_idx = Index(idx0, col_idx)

            # Load data from original matrix C.
            var c_data: SIMD[simd_size, accum_type] = 0
            if skip_boundary_check or (
                Index(idx0, col_idx + simd_size) <= (self.c_bound - tile_idx)
            ):
                # Use simd load if all within bound
                c_data = self.c.simd_load[simd_size](global_idx)
            elif (idx0 + tile_idx[0]) < self.c_bound[0]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[simd_size, accum_type](
                    self.c._offset(global_idx),
                    0,
                    self.c_bound[1] - tile_idx[1] - col_idx,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = SIMD[simd_size, accum_type](0)

            # Store data to local buffer.
            c_local.simd_store[simd_size](local_idx, c_data)

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(
                a_row_size,
                pack_inner_size * simd_size,
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

        @always_inline
        fn outer_body[idx0: Int, idx1: Int]():
            alias col_idx = idx1 * simd_size
            let global_idx_pair = (
                Index(self.global_offset.M, self.global_offset.N)
                + tile_idx
                + Index(idx0, col_idx)
            )
            let global_idx = Index(
                global_idx_pair[0],
                global_idx_pair[1],
            )
            let local_idx = Index(idx0, col_idx)

            # Load data from original matrix C.
            var c_data = c_local.simd_load[simd_size](local_idx)

            if skip_boundary_check or (
                Index(idx0, col_idx + simd_size) <= (self.c_bound - tile_idx)
            ):
                # Use simd store if all within bound
                self.c.simd_store[simd_size](global_idx, c_data)
            elif idx0 < (self.c_bound[0] - tile_idx[0]):
                # Use partial store if row in bound but col not
                #  in simd bound.
                partial_simd_store[simd_size, accum_type](
                    self.c._offset(global_idx),
                    0,
                    self.c_bound[1] - tile_idx[1] - col_idx,
                    c_data,
                )

        unroll2[a_row_size, pack_inner_size // simd_size, outer_body]()

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
            let linear_offset: Int = Int(self.offset_table[row_idx].value)
            if linear_offset == -1:
                return SIMD[1, value_type](0)
            else:
                self.offset_table[row_idx] = (
                    linear_offset + 1
                ).__as_mlir_index()
                return self.input_base_pointer.load(linear_offset)
        else:
            let n_ho_wo = _m_to_n_ho_wo_nhwc(index_m_k[0], self.conv_shape)
            let rsc = _k_to_r_s_c_nhwc(index_m_k[1], self.conv_shape)
            let hi_wi = _ho_wo_to_hi_wi(
                Index(n_ho_wo[1], n_ho_wo[2]),
                Index(rsc[0], rsc[1]),
                self.conv_shape,
            )

            if hi_wi < Index(
                self.conv_shape.h, self.conv_shape.w
            ) and hi_wi >= Index(0, 0):
                return self.input.simd_load[1](
                    Index(
                        # N
                        n_ho_wo[0],
                        # H
                        hi_wi[0],
                        # W
                        hi_wi[1],
                        # C
                        rsc[2],
                    )
                )

        return SIMD[1, value_type](0)

    fn _accumulate(
        self,
        c_local: NDBuffer[
            2,
            create_dim_list(
                a_row_size,
                pack_inner_size * simd_size,
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
        let n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx[1]

        let local_a = Buffer[
            (simd_size * a_row_size).__as_mlir_index(),
            value_type,
        ].stack_allocation()

        @always_inline
        fn body[idx: Int]():
            alias fill_a_idx = idx
            var global_m = self.global_offset.M + fill_a_idx
            let a_val_scalar = self._load_a(
                Index(global_m, global_k), fill_a_idx
            )
            let a_fill_val = SIMD[simd_size, value_type](a_val_scalar)
            local_a.simd_store[simd_size](fill_a_idx * simd_size, a_fill_val)

        unroll[a_row_size, body]()

        @always_inline
        fn outer_body[idx0: Int, idx1: Int]():
            alias col_idx = idx0 * simd_size

            # Loop over local accumulator tiles.
            let b_val = self.b_packed.simd_load[simd_size](
                Index(n_outer_idx, tile_n_k_idx[1], col_idx)
            ).cast[accum_type]()

            let a_val = local_a.simd_load[simd_size](idx1 * simd_size).cast[
                accum_type
            ]()
            var c_idx = Index(idx1, col_idx)
            var c_val = c_local.simd_load[simd_size](c_idx)

            c_val = a_val.fma(b_val, c_val)
            c_local.simd_store[simd_size](c_idx, c_val)

        unroll2[pack_inner_size // simd_size, a_row_size, outer_body]()

    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            2,
            create_dim_list(
                a_row_size,
                pack_inner_size * simd_size,
            ),
            accum_type,
        ].stack_allocation()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
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
            for idx_k in range(self.tile_n_k[1]):
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
    let ho = (m % (conv_shape.out_h * conv_shape.out_w)) // conv_shape.out_w
    let wo = m % conv_shape.out_w
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
    let s = (k // conv_shape.c) % conv_shape.s
    let c = k % conv_shape.c
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
        conv_shape.pad_h[0],
        conv_shape.pad_w[0],
    )
    return ho_wo * conv_shape.stride - pad_left + r_s * conv_shape.dilation


fn get_partitioned_workload(
    task_idx: Int, number_of_tasks: Int, total_load: Int
) -> StaticIntTuple[2]:
    """Naive balanced implementation of load partition.
    Args:
        task_idx: Index of the assigned task.
        number_of_tasks: Total number of task to partition.
    Returns:
        Partition result in (load_start_idx, load_amount), meaning the partition
        is in [start_idx, start_idx+load_amount)
    """
    var divided_load = total_load // number_of_tasks
    let residue_load = total_load % number_of_tasks
    var start_idx: Int
    if task_idx < residue_load:
        start_idx = task_idx * (divided_load + 1)
        divided_load += 1
    else:
        start_idx = (
            residue_load * (divided_load + 1)
            + (task_idx - residue_load) * divided_load
        )
    return Index(start_idx, divided_load)


# TODO (Fixel): This class has massive code duplication with matmul kernels.
#  Could drastically clean up when non-inlined closure is supported or without
#   language support the conv op and matmul op should share a "gemm skeleton"
#   library to de-duplicate.
@register_passable("trivial")
struct ConvIm2ColNHWC[
    shape_input: DimList[4],
    shape_filter: DimList[4],
    shape_output: DimList[4],
    packed_shape: DimList[3],
    type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    pack_cache_size: Int,
    filter_layout: Image2DLayout,
]:
    var out: NDBuffer[4, shape_output, type]
    var input: NDBuffer[4, shape_input, type]
    var filter: NDBuffer[4, shape_filter, type]

    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]
    var gemm_shape: GemmShape
    var conv_shape: ConvShape

    # 2D view of the tensors as implicit matmul input.
    var c: NDBuffer[2, DimList[2].create_unknown(), type]
    var a: NDBuffer[2, DimList[2].create_unknown(), type]
    var b: NDBuffer[2, DimList[2].create_unknown(), type]

    var num_tasks_m: Int
    var num_tasks_n: Int

    # Partitioned row index for the current thread.
    var row_start_idx: Int
    var total_row_count: Int

    # Partitioned col index for the current thread.
    var col_start_idx: Int
    var total_col_count: Int

    # Interface method
    @staticmethod
    fn run(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
        out_chain: OutputChainPtr,
    ):
        """Interface function to run im2col convolution on the given images and
        filters.
        Args:
            out(NDBuffer): Pre-allocated output space.
            input(NDBuffer): The input to the convolution op.
            filter(NDBuffer): The filter to convolve the input with.
            conv_shape: Struct describing the convolution dimensions.
        """

        # Translate conv shape to gemm shape for computation mapping.
        # For NHWC data layout.
        let gemm_shape = GemmShape {
            M: (conv_shape.n * conv_shape.out_h * conv_shape.out_w),
            N: (conv_shape.f),
            K: (conv_shape.r * conv_shape.s * conv_shape.c),
        }

        let num_threads = out_chain.get_runtime().parallelism_level()
        let complexity = gemm_shape.M

        # minimum number of rows each thread should take.
        let row_block_unit: Int = 32
        let col_block_unit: Int = pack_inner_size

        # TODO: add a proper partition heuristic.
        let num_tasks_m = max(min(complexity // row_block_unit, num_threads), 1)
        let num_tasks_n = max(
            min(num_threads // num_tasks_m, gemm_shape.N // col_block_unit),
            1,
        )

        @always_inline
        fn task_func(task_id: Int):
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
            ](
                out,
                input,
                filter,
                conv_shape,
                gemm_shape,
                num_tasks_m,
                num_tasks_n,
                task_id,
            )
            conv._run_implicit_matmul()

        async_parallelize[task_func](out_chain, num_tasks_m * num_tasks_n)

    fn __init__(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
        gemm_shape: GemmShape,
        num_tasks_m: Int,
        num_tasks_n: Int,
        task_id: Int,
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
        conv.tile_n_k = calculate_tile_n_k[pack_cache_size, pack_inner_size](
            GemmShape(
                gemm_shape.M,
                gemm_shape.N,
                conv_shape.c,  # Do not yet pack more than C.
            )
        )
        conv.gemm_shape = gemm_shape
        conv.conv_shape = conv_shape

        # Output shape [N, F, Ho, Wo]
        let c_pointer = out._offset(Index(0, 0, 0, 0))
        conv.c = NDBuffer[2, DimList[2].create_unknown(), type](
            c_pointer.address,
            create_dim_list(
                gemm_shape.M,
                gemm_shape.N,
            ),
            type,
        )

        # Create 2D view for input.
        conv.a = NDBuffer[2, DimList[2].create_unknown(), type](
            input.data.address,
            create_dim_list(
                gemm_shape.M,
                gemm_shape.K,
            ),
            type,
        )

        # Create 2D view for filter.
        if filter_layout == Image2DLayout.NHWC:  # FRSC layout
            conv.b = NDBuffer[2, DimList[2].create_unknown(), type](
                filter.data.address,
                create_dim_list(
                    gemm_shape.N,
                    gemm_shape.K,
                ),
                type,
            )
        elif filter_layout == Image2DLayout.RSCF:  # RSCF layout
            conv.b = NDBuffer[2, DimList[2].create_unknown(), type](
                filter.data.address,
                create_dim_list(
                    gemm_shape.K,
                    gemm_shape.N,
                ),
                type,
            )

        conv.num_tasks_m = num_tasks_m
        conv.num_tasks_n = num_tasks_n

        let task_id_m = task_id // num_tasks_n
        let task_id_n = task_id % num_tasks_n
        let partition_m = get_partitioned_workload(
            task_id_m, num_tasks_m, gemm_shape.M
        )
        let partition_n = get_partitioned_workload(
            task_id_n, num_tasks_n, gemm_shape.N
        )
        conv.row_start_idx = partition_m[0]
        conv.total_row_count = partition_m[1]
        conv.col_start_idx = partition_n[0]
        conv.total_col_count = partition_n[1]

        return conv

    fn _run_implicit_matmul(self):
        """Wrapper utility funciton: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        # Allocate buffer to pack transformed image.
        var _bpacked_data = _raw_stack_allocation[
            pack_cache_size,  # Count.
            type,  # Data type.
            simd_byte_width(),  # Alignment.
        ]()

        # Manually set the shape of packed B buffer:
        let mapped_bpacked = self._view_buffer_as(
            _bpacked_data.address,
            self.tile_n_k[0],
            self.tile_n_k[1],
            pack_inner_size,
        )

        self._outer_k_loop(mapped_bpacked)

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(self, b_packed: NDBuffer[3, packed_shape, type]):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        let tile_k = self.tile_n_k[1]
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
        let valid_col_end: Int = self.col_start_idx + self.total_col_count
        let tile_n: Int = self.tile_n_k[0]

        # Remap buffer indices for current tile.
        var remapped_bpacked = self._view_buffer_as(
            b_packed.data, tile_n, sub_tile_k, pack_inner_size
        )

        var col_idx: Int = self.col_start_idx
        # Proceed with the large tile:
        col_idx = self._outer_n_loop_helper[pack_inner_size](
            remapped_bpacked,
            global_offset,
            tile_n,
            sub_tile_k,
            col_idx,
            valid_col_end,
        )

        # Cover residual tiles.
        if col_idx < valid_col_end:
            remapped_bpacked = self._view_buffer_as(
                b_packed.data, simd_size, sub_tile_k, simd_size
            )
            col_idx = self._outer_n_loop_helper[simd_size](
                remapped_bpacked,
                global_offset,
                simd_size,
                sub_tile_k,
                col_idx,
                valid_col_end,
            )

        # Cover the last sub simdsize tile:
        # This call will handle the sub-simd size boundary.
        if col_idx < valid_col_end:
            self._outer_m_loop[simd_size](
                remapped_bpacked,
                global_offset + GemmShape(0, col_idx, 0),
                simd_size,
                sub_tile_k,
            )

    fn _outer_n_loop_helper[
        m_loop_pack_inner_size: Int
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
        m_loop_pack_inner_size: Int
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
        let valid_col_count = (
            self.col_start_idx + self.total_col_count - global_offset.N
        )

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
        skip_col_bound: Bool, m_loop_pack_inner_size: Int
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
        if filter_layout == Image2DLayout.NHWC:
            PackMatrixRows[
                DimList[2].create_unknown(),
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
                Index(
                    self.b.dim[0]() - global_offset.N,
                    self.b.dim[1]() - global_offset.K,
                ),
            )
        else:  # TODO: add assert, filter layout should be RSCF.
            PackMatrixCols[
                DimList[2].create_unknown(),
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
                Index(
                    self.b.dim[0]() - global_offset.K,
                    self.b.dim[1]() - global_offset.N,
                ),
            )

        # Launch the MLoop
        let sub_tile_n_k = Index(sub_tile_n, sub_tile_k)
        let valid_row_end = self.total_row_count + self.row_start_idx

        # Launch largest row blocks possible and
        #  then reduce row size to maximizing unrolled tiles.
        var row_idx: Int = self.row_start_idx
        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, a_row_size
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 4
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 3
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 2
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            skip_col_bound, m_loop_pack_inner_size, 1
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

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
            let ho_wo = Index(n_ho_wo[1], n_ho_wo[2])

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
        m_loop_pack_inner_size: Int,
        RowSize: Int,
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
                DimList[2].create_unknown(),
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
                self.col_start_idx,
                self.total_col_count,
            )
            row_idx += RowSize
        return row_idx

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed: DTypePointer[type],
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
        return NDBuffer[3, packed_shape, type](
            b_packed.address,
            create_dim_list(
                tile_n // n_inner_size,
                tile_k,
                n_inner_size,
            ),
            type,
        )
