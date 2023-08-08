# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, debug_assert
from Buffer import (
    NDBuffer,
    Buffer,
    DynamicRankBuffer,
    partial_simd_load,
    partial_simd_store,
    _compute_ndbuffer_offset,
)
from ConvUtils import (
    ConvShape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_conv_num_tasks,
    get_conv_num_partitions,
    get_conv2d_shape,
)
from DType import DType
from Functional import (
    unroll,
    async_parallelize,
    sync_parallelize,
    tile,
    unswitch,
    vectorize_unroll,
)
from Image import ImageData, Image2DLayout, ImageShape
from Index import Index, StaticIntTuple
from Intrinsics import PrefetchOptions, external_call
from LLCL import OutputChainPtr, OwningOutputChainPtr
from List import Dim, DimList, VariadicList
from Math import min, max, fma, div_ceil
from Matmul import (
    GemmShape,
    MatmulInnerLoopBPacked,
    calculate_tile_n_k,
    PackMatrixCols,
    PackMatrixRows,
    null_elementwise_epilogue,
)
from MatmulUtils import (
    get_matmul_prefetch_b_distance_k,
    get_partitioned_matmul,
    get_partitioned_matmul_im2col,
    get_min_task_size,
    PartitionHeuristic,
    partition_work,
)
from Memory import memset_zero, stack_allocation
from Pointer import DTypePointer
from Range import range
from ShapeFuncUtils import get_sliding_window_out_dim
from SIMD import SIMD
from TargetInfo import simd_byte_width, simdwidthof, alignof
from TypeUtilities import rebind
from OptionalParam import OptionalParamInts


alias MAX_NUM_CHANNELS_TILE = 384


@value
struct Naive2dConvolution[
    static_output_shape: DimList,
    static_filter_shape: DimList,
    static_input_shape: DimList,
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
                pad_h(StaticIntTuple): Padding on the height dimension with
                    assumed tuple def (PadOnLowerIdx, PadOnHigherIdx).
                pad_w(StaticIntTuple): Padding on the width dimension with
                    assumed tuple def (PadOnLowerIdx, PadOnHigherIdx).
                stride(StaticIntTuple): Strides on height and width dimensions
                    with assumed tuple def (StrideH, StrideW).
                dilation(StaticIntTuple): Dilations on height and width
                    dimensions with assumed tuple def (dilation_h, dilation_w).
        """
        # Create an instance of the convolution op.
        let naive2d_convolution = Naive2dConvolution[
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
        inout self,
        output: ImageData[static_output_shape, type, static_data_layout],
        input: ImageData[static_input_shape, type, static_data_layout],
        filter: ImageData[static_filter_shape, type, static_filter_layout],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
    ):
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
        # Register input/output buffers and parameters.
        self.output = output
        self.input = input
        self.filter = filter
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride = stride
        self.dilation = dilation

        # Derive layout agnostic shape information.
        self.output_shape = ImageShape.__init__[
            static_output_shape, type, static_data_layout
        ](output)
        self.input_shape = ImageShape.__init__[
            static_input_shape, type, static_data_layout
        ](input)
        self.filter_shape = ImageShape.__init__[
            static_filter_shape, type, static_filter_layout
        ](filter)

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
        var value: SIMD[type, 1] = 0

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
                        let input_val = self.input[
                            output_idx[0],  # N
                            c_idx,  # C
                            input_image_index[0],  # H
                            input_image_index[1],  # W
                        ]
                        let filter_val = self.filter[
                            output_idx[1],
                            c_idx,
                            r_idx,
                            s_idx,  # F  # C  # R  # S
                        ]
                        value += input_val * filter_val

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


@value
struct PackIm2ColNCHW[
    # original matrix shape list
    static_original_shape: DimList,
    # packed matrix shape list
    static_packed_shape: DimList,
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
        inout self,
        origin_image: NDBuffer[4, static_original_shape, type],
        packed_matrix: NDBuffer[3, static_packed_shape, type],
        conv_shape: ConvShape,
        global_offset: StaticIntTuple[2],
        pack_tile_kn_dim: StaticIntTuple[2],
        batch_idx: Int,
    ):
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
        self.origin_image = origin_image
        self.packed_matrix = packed_matrix
        self.conv_shape = conv_shape
        self.global_offset = global_offset
        self.pack_tile_kn_dim = pack_tile_kn_dim
        self.batch_idx = batch_idx

        # TODO:
        #  Assuming same-padding and stride-1, so output shape and input shape
        # are the same. Will extend this.
        let image_output_shape = Index(conv_shape.h, conv_shape.w)
        self.image_output_shape = image_output_shape
        self.im2col_output_shape = Index(
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
        self.pad_low = Index(pad_h[0], pad_w[0])
        self.pad_high = Index(pad_h[1], pad_h[1])

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
        # assert_param[is_divisible_by[block_size, simd_size]()]

        # Convert output index to input index.
        let global_in_image_offset = self._output_to_input(
            global_out_image_offset, rs_idx
        )

        alias loop_iters = block_size // simd_size

        @always_inline
        @parameter
        fn body[idx: Int]():
            alias col_idx = idx * simd_size
            # calculate input index
            let global_in_image_idx = global_in_image_offset + Index(0, col_idx)

            # Load a vector of image data or fill zero.
            let image_data: SIMD[type, simd_size]

            if fill_zero:
                image_data = SIMD[type, simd_size](0)
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
        self, nk_idx: StaticIntTuple[2], vec_data: SIMD[type, simd_size]
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
            simd_size,
            type,
        ].stack_allocation()

        # Initialize data with zero
        vector.simd_store[simd_size](0, SIMD[type, simd_size](0))

        # calculate h and w output indices.
        var h_o_idx = global_out_image_offset[0]
        var w_o_idx = global_out_image_offset[1]

        # Vector index for filling the simd elements.
        @always_inline
        @parameter
        fn body[idx: Int]():
            alias vec_idx = idx
            # Calculate the current output and input indices.
            let o_image_idx = Index(h_o_idx, w_o_idx)
            let i_image_idx = self._output_to_input(o_image_idx, rs_idx)

            var element = SIMD[type, 1](0)
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
                0,
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
            let global_k_n_idx = self.global_offset + Index(k_idx, n_idx)
            # Map output matrix index to output convolution image index.
            let o_image_ho_wo = self._n_to_ho_wo(global_k_n_idx[1])
            # Map output convolution image index to input convolution image
            #  index.
            let i_image_h_w = self._output_to_input(
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
@value
@register_passable("trivial")
struct ConvIm2ColNCHW[
    shape_input: DimList,
    shape_filter: DimList,
    shape_output: DimList,
    packed_shape: DimList,
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
    var c: NDBuffer[2, DimList.create_unknown[2](), type]

    # 2D view of the filter as implicit matmul input.
    var a: NDBuffer[2, DimList.create_unknown[2](), type]

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

        # Translate conv shape to gemm shape for computation mapping.
        let gemm_shape = GemmShape {
            M: conv_shape.f,
            N: (conv_shape.out_h * conv_shape.out_w),
            K: (conv_shape.r * conv_shape.s * conv_shape.c),
        }

        let tile_n_k = calculate_tile_n_k[pack_cache_size, pack_inner_size](
            gemm_shape
        )

        return Self {
            out: out,
            input: input,
            filter: filter,
            conv_shape: conv_shape,
            gemm_shape: gemm_shape,
            tile_n_k: tile_n_k,
            batch_idx: 0,
            c: NDBuffer[2, DimList.create_unknown[2](), type](),
            a: NDBuffer[2, DimList.create_unknown[2](), type](),
        }

    fn _run_implicit_matmul(inout self):
        """Wrapper utility function: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        # Allocate buffer to pack transformed image.
        let _bpacked_data = stack_allocation[
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
    fn _outer_k_loop(inout self, b_packed: NDBuffer[3, packed_shape, type]):
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
            a_row_size,
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
        alias prefetch_b_distance_k = get_matmul_prefetch_b_distance_k()
        var row_idx = start_idx
        while row_idx <= (valid_row_count - RowSize):
            MatmulInnerLoopBPacked[
                DimList.create_unknown[2](),  # shape_a
                DimList.create_unknown[2](),  # shape c
                packed_shape,  # packed_shape
                type,  # a_type
                type,  # b_type
                type,  # c_type
                simd_size,
                RowSize,
                m_loop_pack_inner_size,
                skip_col_bound,
                prefetch_b_distance_k,  # prefetch distance
                False,  # critical_stride
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

    fn _initialize_buffer_view(inout self):
        """Initializes the internal gemm operand tensors with the translated
        dynamic gemm shapes from convolution shapes.
        """
        # Output shape [N, F, Ho, Wo]
        let c_pointer = self.out._offset(Index(self.batch_idx, 0, 0, 0))
        self.c = NDBuffer[2, DimList.create_unknown[2](), type](
            c_pointer.address,
            DimList(
                self.conv_shape.f,
                self.conv_shape.out_h * self.conv_shape.out_w,
            ),
        )

        # Create 2D view for filter.
        self.a = NDBuffer[2, DimList.create_unknown[2](), type](
            self.filter.data.address,
            DimList(
                self.gemm_shape.M,
                self.gemm_shape.K,
            ),
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
            DimList(
                tile_n // n_inner_size,
                tile_k,
                n_inner_size,
            ),
        )


# TODO (Fixel): This class has massive code duplication with matmul kernels.
#  Could drastically clean up when non-inlined closure is supported or without
#   language support the conv op and matmul op should share a "gemm skeleton"
#   library to de-duplicate.
@value
struct ConvNHWCInnerLoopFilterPacked[
    shape_input: DimList,
    shape_c: DimList,
    packed_shape: DimList,
    accum_type: DType,
    value_type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    skip_boundary_check: Bool,
    use_padding: Bool,
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
    var offset_table: NDBuffer[
        2, DimList(MAX_NUM_CHANNELS_TILE, a_row_size), DType.index
    ]

    var input_base_pointer: DTypePointer[value_type]

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
        let offset_table = NDBuffer[
            2, DimList(MAX_NUM_CHANNELS_TILE, a_row_size), DType.index
        ].stack_allocation()

        let instance = ConvNHWCInnerLoopFilterPacked[
            shape_input,
            shape_c,
            packed_shape,
            accum_type,
            value_type,
            simd_size,
            a_row_size,
            pack_inner_size,
            skip_boundary_check,
            use_padding,
        ](
            c,
            input,
            b_packed,
            global_offset,
            tile_n_k,
            (
                Index(c.dim[0](), col_start_idx + total_col_count)
                - Index(global_offset.M, global_offset.N)
            ),
            conv_shape,
            offset_table,
            input.data,
        )

        instance._run_inner_loop()

    fn _initialize_offset_table(self, num_segments: Int, contiguous_len: Int):
        let k_offset = self.global_offset.K

        # tile_n_k[1] is either multiple of conv_shape.c or it fits in one
        # channel. For now, We avoid tiles that partially cover more than
        # one channel.
        # Map a channel's starting k index to the corresponding localtion in
        # the 4D input tensor
        for segment_idx in range(num_segments):
            let k = self.global_offset.K + segment_idx * contiguous_len
            let r_s_c = _k_to_r_s_c_nhwc(k, self.conv_shape)
            let r_s = Index(r_s_c[0], r_s_c[1])

            @always_inline
            @parameter
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

                let offset_idx = Index(segment_idx, row_idx)
                if hi_wi >= Index(0, 0) and hi_wi < Index(
                    self.conv_shape.h, self.conv_shape.w
                ):
                    self.offset_table[offset_idx] = (
                        linear_offset - segment_idx * contiguous_len
                    )
                else:
                    self.offset_table[offset_idx] = -1

            unroll[a_row_size, body]()

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            2,
            DimList(
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
        @parameter
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.simd_store[simd_size](
                Index(idx0, idx1 * simd_size),
                SIMD[accum_type, simd_size](0),
            )

        unroll[a_row_size, pack_inner_size // simd_size, outer_body]()

    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            2,
            DimList(
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
        @parameter
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
            var c_data: SIMD[accum_type, simd_size] = 0
            if skip_boundary_check or (
                Index(idx0, col_idx + simd_size) <= (self.c_bound - tile_idx)
            ):
                # Use simd load if all within bound
                c_data = self.c.simd_load[simd_size](global_idx)
            elif (idx0 + tile_idx[0]) < self.c_bound[0]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[accum_type, simd_size](
                    self.c._offset(global_idx),
                    0,
                    self.c_bound[1] - tile_idx[1] - col_idx,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = SIMD[accum_type, simd_size](0)

            # Store data to local buffer.
            c_local.simd_store[simd_size](local_idx, c_data)

        unroll[a_row_size, pack_inner_size // simd_size, outer_body]()

    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            2,
            DimList(
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
        @parameter
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
            let c_data = c_local.simd_load[simd_size](local_idx)

            if skip_boundary_check or (
                Index(idx0, col_idx + simd_size) <= (self.c_bound - tile_idx)
            ):
                # Use simd store if all within bound
                self.c.simd_store[simd_size](global_idx, c_data)
            elif idx0 < (self.c_bound[0] - tile_idx[0]):
                # Use partial store if row in bound but col not
                #  in simd bound.
                partial_simd_store(
                    self.c._offset(global_idx),
                    0,
                    self.c_bound[1] - tile_idx[1] - col_idx,
                    c_data,
                )

        unroll[a_row_size, pack_inner_size // simd_size, outer_body]()

    # TODO: This can be lifted to common utility.

    @always_inline
    fn _load_a(
        self, segment_idx: Int, k_idx: Int, row_idx: Int
    ) -> SIMD[value_type, 1]:
        """Utility to load one value of Im2col transformed matrix from the
        pre-transformed image.
            Args:
                index_m_k (StaticIntTuple): Index into the post im2col operandA
                    in (M, K) format.
            Returns (SIMD):
                Value loaded from the translated address of image input.
        """
        let offset_idx = Index(segment_idx, row_idx)
        let linear_offset: Int = Int(self.offset_table[offset_idx].value)

        @parameter
        if use_padding:
            if linear_offset == -1:
                return SIMD[value_type, 1](0)
            else:
                return self.input_base_pointer.load(linear_offset + k_idx)
        else:
            return self.input_base_pointer.load(linear_offset + k_idx)

    fn _accumulate(
        self,
        c_local: NDBuffer[
            2,
            DimList(
                a_row_size,
                pack_inner_size * simd_size,
            ),
            accum_type,
        ],
        segment_idx: Int,
        tile_n_k_idx: StaticIntTuple[2],
        contiguous_len: Int,
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

        @always_inline
        @parameter
        fn outer_body[idx0: Int, idx1: Int]():
            alias col_idx = idx0 * simd_size
            let c_idx = Index(idx1, col_idx)

            # Loop over local accumulator tiles.
            let b_val = self.b_packed.simd_load[simd_size](
                Index(n_outer_idx, tile_n_k_idx[1], col_idx)
            ).cast[accum_type]()

            let a_val_scalar = self._load_a(segment_idx, tile_n_k_idx[1], idx1)
            let a_val = SIMD[value_type, simd_size](a_val_scalar).cast[
                accum_type
            ]()
            var c_val = c_local.simd_load[simd_size](c_idx)

            c_val = fma(a_val, b_val, c_val)
            c_local.simd_store[simd_size](c_idx, c_val)

        unroll[pack_inner_size // simd_size, a_row_size, outer_body]()

    fn _run_inner_loop(self):
        """Utility funcion on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        # Allocate accumulation buffer.
        let c_local = NDBuffer[
            2,
            DimList(
                a_row_size,
                pack_inner_size * simd_size,
            ),
            accum_type,
        ].stack_allocation()

        let max_contiguous_len = self.conv_shape.s * self.conv_shape.c
        let contiguous_len = (
            max_contiguous_len if self.tile_n_k[1] >= max_contiguous_len
            and not use_padding else self.conv_shape.c
        )
        let num_segments = div_ceil(self.tile_n_k[1], contiguous_len)

        self._initialize_offset_table(num_segments, contiguous_len)

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, Index(0, idx_n))

            var segment_idx = 0
            while segment_idx < num_segments:
                let pos = segment_idx * contiguous_len
                let chunk = contiguous_len if segment_idx < (
                    self.tile_n_k[1] // contiguous_len
                ) else self.tile_n_k[1] - pos
                var j = 0
                while j < chunk:
                    self._accumulate(
                        c_local,
                        segment_idx,
                        Index(idx_n, pos + j),
                        contiguous_len,
                    )
                    j += 1
                segment_idx += 1
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
    TODO(Fixel): This utility should be generalized into a im2col util
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
    TODO(Fixel): This utility should be generalized into a im2col util
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
    TODO(Fixel): This utility should be generalized into a conv util
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
    let start_idx: Int
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
struct ConvIm2ColNHWC[
    shape_input: DimList,
    shape_filter: DimList,
    shape_output: DimList,
    packed_shape: DimList,
    type: DType,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
    pack_cache_size: Int,
    filter_layout: Image2DLayout,
    elementwise_epilogue_enabled: Bool,
]:
    var out: NDBuffer[4, shape_output, type]
    var input: NDBuffer[4, shape_input, type]
    var filter: NDBuffer[4, shape_filter, type]

    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]
    var gemm_shape: GemmShape
    var conv_shape: ConvShape

    # 2D view of the tensors as implicit matmul input.
    var c: NDBuffer[2, DimList.create_unknown[2](), type]
    var a: NDBuffer[2, DimList.create_unknown[2](), type]
    var b: NDBuffer[2, DimList.create_unknown[2](), type]

    var num_tasks: Int

    # Partitioned row index for the current thread.
    var row_start_idx: Int
    var total_row_count: Int

    # Partitioned col index for the current thread.
    var col_start_idx: Int
    var total_col_count: Int

    var elementwise_epilogue_fn: fn (GemmShape, GemmShape) capturing -> None

    # Interface method
    @staticmethod
    fn run(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
        out_chain: OutputChainPtr,
    ):
        Self.run(
            out, input, filter, conv_shape, null_elementwise_epilogue, out_chain
        )

    # Interface method
    @staticmethod
    fn run(
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) capturing -> None,
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
        let complexity = gemm_shape.M * gemm_shape.N * gemm_shape.K
        let num_tasks = min(
            div_ceil(complexity, get_min_task_size()), num_threads
        )

        @always_inline
        @parameter
        fn task_func(task_id: Int):
            let conv = ConvIm2ColNHWC[
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
                elementwise_epilogue_enabled,
            ](
                out,
                input,
                filter,
                conv_shape,
                gemm_shape,
                num_tasks,
                task_id,
                elementwise_epilogue_fn,
            )
            conv._run_implicit_matmul()

        # TODO (#12624): Closure captures some state on the stack so this needs
        # to be synchronous in order to keep that state alive
        sync_parallelize[task_func](out_chain, num_tasks)

    fn __init__(
        inout self,
        out: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[4, shape_filter, type],
        conv_shape: ConvShape,
        gemm_shape: GemmShape,
        num_tasks: Int,
        task_id: Int,
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) capturing -> None,
    ):
        """Constructor of an instance of the im2col conv2d operator.

        Args:
            out(NDBuffer): Pre-allocated output space.
            input(NDBuffer): The input to the convolution op.
            filter(NDBuffer): The filter to convolve the input with.
            conv_shape: Struct describing the convolution dimensions.
        """

        var tile_n_k = calculate_tile_n_k[pack_cache_size, pack_inner_size](
            GemmShape(
                gemm_shape.M,
                gemm_shape.N,
                gemm_shape.K,
            )
        )
        # We only support unit dilation. The data corresponds to conv_shape.s
        # channels are contiguous in memory.
        let contiguous_len = conv_shape.s * conv_shape.c
        # Round k to multiple of contiguous segments if applicable.
        if (
            (conv_shape.pad_h == Index(0, 0))
            & (conv_shape.pad_w == Index(0, 0))
            & (tile_n_k[1] > contiguous_len)
        ):
            tile_n_k[1] = tile_n_k[1] - tile_n_k[1] % contiguous_len
        # Round k to multiple of channel size if applicable.
        elif tile_n_k[1] > conv_shape.c:
            tile_n_k[1] = tile_n_k[1] - tile_n_k[1] % conv_shape.c

        # Output shape [N, F, Ho, Wo]
        let c_pointer = out._offset(Index(0, 0, 0, 0))
        let c = NDBuffer[2, DimList.create_unknown[2](), type](
            c_pointer.address,
            DimList(
                gemm_shape.M,
                gemm_shape.N,
            ),
        )

        # Create 2D view for input.
        let a = NDBuffer[2, DimList.create_unknown[2](), type](
            input.data.address,
            DimList(
                gemm_shape.M,
                gemm_shape.K,
            ),
        )

        var b = NDBuffer[2, DimList.create_unknown[2](), type]()
        # Create 2D view for filter.
        if filter_layout == Image2DLayout.NHWC:  # FRSC layout
            b = NDBuffer[2, DimList.create_unknown[2](), type](
                filter.data.address,
                DimList(
                    gemm_shape.N,
                    gemm_shape.K,
                ),
            )
        elif filter_layout == Image2DLayout.RSCF:  # RSCF layout
            b = NDBuffer[2, DimList.create_unknown[2](), type](
                filter.data.address,
                DimList(
                    gemm_shape.K,
                    gemm_shape.N,
                ),
            )

        let sub_matmul_config = get_partitioned_matmul[
            PartitionHeuristic.Im2col, False
        ](gemm_shape.M, gemm_shape.N, gemm_shape.K, task_id, num_tasks)
        let row_start_idx = sub_matmul_config.offset[0]
        let total_row_count = sub_matmul_config.shape[0]
        let col_start_idx = sub_matmul_config.offset[1]
        let total_col_count = sub_matmul_config.shape[1]

        self.out = out
        self.input = input
        self.filter = filter
        self.tile_n_k = tile_n_k
        self.gemm_shape = gemm_shape
        self.conv_shape = conv_shape
        self.a = a
        self.b = b
        self.c = c
        self.num_tasks = num_tasks
        self.row_start_idx = row_start_idx
        self.total_row_count = total_row_count
        self.col_start_idx = col_start_idx
        self.total_col_count = total_col_count
        self.elementwise_epilogue_fn = elementwise_epilogue_fn

    fn _run_implicit_matmul(self):
        """Wrapper utility function: Allocates packing space on the stack and
        run the matmul routine on the whole problem space.
        """
        if self.total_row_count <= 0 or self.total_col_count <= 0:
            return

        # Allocate buffer to pack transformed image.
        let _bpacked_data = stack_allocation[
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
        if self.tile_n_k[1] < self.conv_shape.c:
            self._outer_k_loop_large_channel(mapped_bpacked)
        else:
            self._outer_k_loop(mapped_bpacked)

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(self, b_packed: NDBuffer[3, packed_shape, type]):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        debug_assert(
            self.tile_n_k[1] >= self.conv_shape.c,
            "channel size must not be less than tile size.",
        )

        let tile_k = self.tile_n_k[1]
        var k_idx: Int = 0

        @always_inline
        @parameter
        fn k_iteration(k_offset: Int, k_tile_size: Int):
            @always_inline
            @parameter
            fn outer_n_switch[last_k_tile: Bool]():
                self._outer_n_loop[last_k_tile](
                    b_packed,
                    GemmShape(0, 0, k_offset),
                    k_tile_size,
                )

            unswitch[outer_n_switch](
                k_offset + k_tile_size == self.gemm_shape.K
            )

        tile[k_iteration](
            0,  # k offset
            self.gemm_shape.K,
            self.tile_n_k[1],  # max tile k size
        )

    fn _outer_k_loop_large_channel(
        self, b_packed: NDBuffer[3, packed_shape, type]
    ):
        """Iterate on the K dimension of the whole problem space.

        Args:
            b_packed(NDBuffer): B matrix in packed layout.
        """
        debug_assert(
            self.tile_n_k[1] < self.conv_shape.c,
            "channel size must be greater than tile size.",
        )

        let tile_k = self.tile_n_k[1]
        let valid_k_count = self.gemm_shape.K
        var k_idx: Int = 0

        @always_inline
        @parameter
        fn k_iteration(channel_start: Int, channel_size: Int):
            @always_inline
            @parameter
            fn k_iteration_helper(k_offset: Int, k_tile_size: Int):
                @always_inline
                @parameter
                fn outer_n_switch[last_k_tile: Bool]():
                    self._outer_n_loop[last_k_tile](
                        b_packed,
                        GemmShape(0, 0, k_offset),
                        k_tile_size,
                    )

                unswitch[outer_n_switch](
                    k_offset + k_tile_size == self.gemm_shape.K
                )

            tile[k_iteration_helper](
                channel_start, channel_start + channel_size, self.tile_n_k[1]
            )

        tile[k_iteration](0, self.gemm_shape.K, self.conv_shape.c)

    fn _outer_n_loop[
        last_k_tile: Bool
    ](
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

        @parameter
        fn m_loop[tile_inner: Int](col_idx: Int, tile_size_n: Int):
            # Remap buffer indices for current tile.
            let remapped_bpacked = self._view_buffer_as(
                b_packed.data, tile_size_n, sub_tile_k, tile_inner
            )
            self._outer_m_loop[last_k_tile, tile_inner](
                remapped_bpacked,
                global_offset + GemmShape(0, col_idx, 0),
                tile_size_n,
                sub_tile_k,
            )

        alias secondary_tiles = VariadicList[Int](pack_inner_size, simd_size)
        let primary_tiles = VariadicList[Int](tile_n, simd_size)

        tile[secondary_tiles, simd_size, m_loop](
            self.col_start_idx, valid_col_end, primary_tiles, simd_size
        )

    fn _outer_m_loop[
        last_k_tile: Bool, m_loop_pack_inner_size: Int
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
        @always_inline
        @parameter
        fn outer_m_helper_switch[skip_col_bound: Bool]():
            self._outer_m_loop_helper[
                last_k_tile,
                skip_col_bound,
                m_loop_pack_inner_size,
            ](b_packed, global_offset, sub_tile_n, sub_tile_k)

        unswitch[outer_m_helper_switch](valid_col_count >= sub_tile_n)

    fn _outer_m_loop_helper[
        last_k_tile: Bool, skip_col_bound: Bool, m_loop_pack_inner_size: Int
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
                DimList.create_unknown[2](),
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
                DimList.create_unknown[2](),
                packed_shape,
                type,
                simd_size,
                m_loop_pack_inner_size,
                # use VNNI
                False,
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
            last_k_tile, skip_col_bound, m_loop_pack_inner_size, a_row_size
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            last_k_tile, skip_col_bound, m_loop_pack_inner_size, 4
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            last_k_tile, skip_col_bound, m_loop_pack_inner_size, 3
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            last_k_tile, skip_col_bound, m_loop_pack_inner_size, 2
        ](b_packed, global_offset, sub_tile_n_k, row_idx, valid_row_end)

        row_idx = self._outer_m_loop_row_helper[
            last_k_tile, skip_col_bound, m_loop_pack_inner_size, 1
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
        last_k_tile: Bool,
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

        @always_inline
        @parameter
        fn m_loop_switch[use_padding: Bool]():
            while row_idx <= (valid_row_count - RowSize):
                let current_offset = global_offset + GemmShape(row_idx, 0, 0)
                ConvNHWCInnerLoopFilterPacked[
                    shape_input,
                    DimList.create_unknown[2](),
                    packed_shape,
                    type,
                    type,
                    simd_size,
                    RowSize,
                    m_loop_pack_inner_size,
                    skip_col_bound,
                    use_padding,
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

                @parameter
                if elementwise_epilogue_enabled & last_k_tile:
                    self.elementwise_epilogue_fn(
                        current_offset,
                        GemmShape {
                            M: RowSize,
                            # TODO: simplify propagation of bounds
                            N: min(
                                sub_tile_n_k[0],
                                self.col_start_idx
                                + self.total_col_count
                                - current_offset.N,
                            ),
                            K: 0,
                        },
                    )
                row_idx += RowSize

        unswitch[m_loop_switch](
            (self.conv_shape.pad_h != Index(0, 0))
            or (self.conv_shape.pad_w != Index(0, 0))
        )

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
            DimList(
                tile_n // n_inner_size,
                tile_k,
                n_inner_size,
            ),
        )


# ===----------------------------------------------------------------------=== #
# Direct Convolution                                                           #
# ===----------------------------------------------------------------------=== #


@value
struct ConvDirectNHWC[
    filter_rank: Int,
    shape_input: DimList,
    shape_filter: DimList,
    shape_output: DimList,
    type: DType,
    filter_packed: Bool,
]:
    """Implement the outer loops for direct convolution.
    Collapse N, HO, WO into one dimension n_ho_wo. Tile n_ho_wo, C, and F.
    The tile factor for C and F are chosen by a heuristic prioritizing C.
    n_ho_wo is tiled by micro kernel's height.

    If n_ho_wo is large enough to spill LLC, we may need to tile n_ho_wo as the
    outer most loop with a factor fit in LLC.

    Assume F is divisible at least by simd_size.
    """

    var output: NDBuffer[4, shape_output, type]
    var input: NDBuffer[4, shape_input, type]
    var filter: NDBuffer[filter_rank, shape_filter, type]

    var conv_shape: ConvShape

    # If n, ho, wo dimensions are merged (for no padding), the first three
    # dimensions are the offsets and sizes in (n_ho_wo, c, f) iteration space.
    # If they are not merged (for non-zero padding), the following denotes
    # (n, c, f, ho). Prioritize partitioning batch size (n).
    var partition_offsets: StaticIntTuple[4]
    var partition_sizes: StaticIntTuple[4]

    var cf_tile_size: StaticIntTuple[2]

    @staticmethod
    fn run(
        output: NDBuffer[4, shape_output, type],
        input: NDBuffer[4, shape_input, type],
        filter: NDBuffer[filter_rank, shape_filter, type],
        conv_shape: ConvShape,
        out_chain: OutputChainPtr,
    ):
        """Run with no padding (valid padding in TF). The micro kernel can include
        points from differeent rows or images."""
        let cf_tile_size = get_conv_tile_shape[
            type, get_direct_conv_micro_kernel_width()
        ](conv_shape)

        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias micro_kernel_width = get_direct_conv_micro_kernel_width()
        alias micro_kernel_f_size = micro_kernel_width * simd_size
        alias simd_size = simdwidthof[type]()

        # Number of partitions in n_ho_wo, c, f dimensions.
        let num_threads = out_chain.get_runtime().parallelism_level()
        let num_tasks = get_conv_num_tasks(num_threads, conv_shape)
        let num_partitions = get_conv_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](num_tasks, conv_shape)

        # Wrap the pointer inside NDBuffer so it can be properly captured by async closure.
        var output_ptr = output.data
        let output_size = conv_shape.n * conv_shape.out_h * conv_shape.out_w * conv_shape.f
        let scratch_size = num_partitions[1] * output_size
        if num_partitions[1] > 1:
            output_ptr = DTypePointer[type].alloc(scratch_size)
        let output_scratch = Buffer[Dim(), type](output_ptr, scratch_size)

        @parameter
        @always_inline
        fn task_func(task_id: Int):
            let task_id_f = task_id % num_partitions[2]
            let quotient = task_id // num_partitions[2]
            let task_id_c = quotient % num_partitions[1]
            let task_id_nhowo = quotient // num_partitions[1]

            let c_range = partition_work(
                task_id_c, num_partitions[1], conv_shape.c, 1
            )
            let f_range = partition_work(
                task_id_f,
                num_partitions[2],
                conv_shape.f,
                micro_kernel_f_size,
            )

            let task_tile_size = Index(
                min(cf_tile_size[0], c_range[1]), cf_tile_size[1]
            )

            let task_output = NDBuffer[4, shape_output, type](
                output_scratch.data.offset(task_id_c * output_size),
                Index(
                    conv_shape.n,
                    conv_shape.out_h,
                    conv_shape.out_w,
                    conv_shape.f,
                ),
            )

            # Indicate if the kernel adds padding.
            let is_any_padding = conv_shape.pad_h != Index(
                0, 0
            ) or conv_shape.pad_w != Index(0, 0)

            # No padding.
            # The partition is similar to matmul. N, HO, WO are merged to form
            # a large "M" dimension.
            if not is_any_padding:
                let nhowo_range = partition_work(
                    task_id_nhowo,
                    num_partitions[0],
                    conv_shape.n * conv_shape.out_h * conv_shape.out_w,
                    micro_kernel_height,
                )

                # Short circuit when a task gets no work. This could happen when
                # the previous tasks get more work due to alignment requirement.
                if nhowo_range[1] <= 0 or c_range[1] <= 0 or f_range[1] <= 0:
                    return

                let instance = ConvDirectNHWC[
                    filter_rank,
                    shape_input,
                    shape_filter,
                    shape_output,
                    type,
                    filter_packed,
                ](
                    task_output,
                    input,
                    filter,
                    conv_shape,
                    Index(nhowo_range[0], c_range[0], f_range[0], 0),
                    Index(nhowo_range[1], c_range[1], f_range[1], 0),
                    task_tile_size,
                )
                instance.direct_conv()
            # There are padding layers.
            # N, HO, WO can't be merged as there can be padding at each row.
            # Only N and HO are partitioned, WO is not split across tasks.
            else:
                # Partition the batch dimension if there is at least one image
                # per task.
                let partition_batch = conv_shape.n >= num_tasks
                let num_partitions_ho = 1 if partition_batch else num_partitions[
                    0
                ]
                let num_partitions_n = num_partitions[
                    0
                ] if partition_batch else 1
                let task_id_ho = 0 if partition_batch else task_id_nhowo
                let task_id_n = task_id_nhowo - task_id_ho
                # Prioritize partitioning the batch dimension
                let n_range = partition_work(
                    task_id_n, num_partitions_n, conv_shape.n, 1
                )
                let ho_range = partition_work(
                    task_id_ho, num_partitions_ho, conv_shape.out_h, 1
                )

                if (
                    n_range[1] <= 0
                    or c_range[1] <= 0
                    or f_range[1] <= 0
                    or ho_range[1] <= 0
                ):
                    return

                let instance = ConvDirectNHWC[
                    filter_rank,
                    shape_input,
                    shape_filter,
                    shape_output,
                    type,
                    filter_packed,
                ](
                    task_output,
                    input,
                    filter,
                    conv_shape,
                    Index(n_range[0], c_range[0], f_range[0], ho_range[0]),
                    Index(n_range[1], c_range[1], f_range[1], ho_range[1]),
                    task_tile_size,
                )
                instance._n_loop()

        if num_partitions[1] > 1:
            # Finish the conv computation and sync at the end.
            let runtime = out_chain.get_runtime()
            let conv_chain = OwningOutputChainPtr(runtime)
            async_parallelize[task_func](conv_chain.borrow(), num_tasks)
            conv_chain.wait()

            # Reduce from the output scratch buffer to the actual output.
            @parameter
            @always_inline
            fn reduce_task(tid: Int):
                # Use all threads in reduction.
                let reduce_range = partition_work(
                    tid, num_threads, output_size, simd_size
                )

                @parameter
                @always_inline
                fn sum[width: Int](idx: Int):
                    let tid_output_offset = reduce_range[0] + idx
                    var vec = output_scratch.data.offset(
                        tid_output_offset
                    ).simd_load[simd_size]()
                    # The number of partitions here is typically small.
                    # There may not be much benefit from unrolling the reduction axis.
                    # Only unroll the last dimension.
                    for i in range(1, num_partitions[1]):
                        vec += output_scratch.data.offset(
                            tid_output_offset + i * output_size
                        ).simd_load[simd_size]()
                    output.data.offset(tid_output_offset).simd_store[simd_size](
                        vec
                    )

                vectorize_unroll[simd_size, 4, sum](reduce_range[1])

            # NOTE: synchronous, so use of locally allocated output_ptr is safe.
            sync_parallelize[reduce_task](out_chain, num_threads)
            output_ptr.free()
        else:
            async_parallelize[task_func](out_chain, num_tasks)

    fn direct_conv(self):
        self._c_tile_loop[False](0, self.cf_tile_size[0])

    fn _c_tile_loop[padded: Bool](self, n: Int, tile_size: Int):
        """Loop over C tiles."""

        @always_inline
        @parameter
        fn c_tile_iteration(c_tile_offset: Int, c_tile_size: Int):
            self._f_tile_loop[padded](n, c_tile_offset, c_tile_size)

        tile[c_tile_iteration](
            self.partition_offsets[1],
            self.partition_offsets[1] + self.partition_sizes[1],
            tile_size,
        )

    fn _f_tile_loop[
        padded: Bool
    ](self, n: Int, c_tile_offset: Int, c_tile_size: Int):
        """Loop over F tiles."""
        alias micro_kernel_width = get_direct_conv_micro_kernel_width()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias simd_size = simdwidthof[type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        @always_inline
        @parameter
        fn f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            @parameter
            if padded:
                self._h_loop[micro_kernel_height, size // simd_size, False](
                    n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size
                )

            else:
                self._n_wo_ho_tile_loop[size, False](
                    f_tile_offset, f_tile_size, c_tile_offset, c_tile_size
                )

        let f_round_by_simd = (
            (self.partition_offsets[2] + self.partition_sizes[2]) // simd_size
        ) * simd_size

        # The first tile size is based on cache size. Within the tile
        # it's stepped by the micro kernel size in F. The rest is stepped
        # by simd_size. If F is not multiple of simd_size, the residual
        # is padded with 0 to fit a simd vector in the packed filter.
        tile[
            VariadicList[Int](
                micro_kernel_f_size, micro_kernel_f_size, simd_size
            ),
            simd_size,
            f_tile_iteration,
        ](
            self.partition_offsets[2],
            f_round_by_simd,
            VariadicList[Int](
                self.cf_tile_size[1], micro_kernel_f_size, simd_size
            ),
            simd_size,
        )

        # If this is the last partition in F and it's not a multiple of simd_size.
        # The partition is aligned by micro_kernel_f_size, so only the last
        # partition is possible to have residual.
        let residual = self.conv_shape.f - f_round_by_simd
        if (
            self.partition_offsets[2] + self.partition_sizes[2]
            == self.conv_shape.f
            and residual > 0
        ):

            @parameter
            if padded:
                self._h_loop[micro_kernel_height, 1, True](
                    n, f_round_by_simd, simd_size, c_tile_offset, c_tile_size
                )
            else:
                self._n_wo_ho_tile_loop[simd_size, True](
                    f_round_by_simd, simd_size, c_tile_offset, c_tile_size
                )

    fn _n_loop(self):
        """Loop over the batch size.
        This is the outermost loop and is used with padding."""
        for n in range(
            self.partition_offsets[0],
            self.partition_offsets[0] + self.partition_sizes[0],
        ):
            self._c_tile_loop[True](n, self.cf_tile_size[0])

    fn _n_wo_ho_tile_loop[
        micro_kernel_f_size: Int, has_residual: Bool
    ](
        self,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        """The N, HO, WO dimensions are fused and traversed with the micro
        kernel height as the step.
        Note that the micro kernel height changes for residual blocks."""
        alias simd_size = simdwidthof[type]()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias micro_kernel_width = micro_kernel_f_size // simd_size

        @always_inline
        @parameter
        fn n_ho_wo_iteration[n_ho_wo_tile_size: Int](n_ho_wo: Int):
            @always_inline
            @parameter
            fn body[c_fully_cached: Bool]():
                self._inner_loops[
                    n_ho_wo_tile_size,
                    micro_kernel_width,
                    c_fully_cached,
                    has_residual,
                ](
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n_ho_wo,
                )

            # c_fully_cached means the C dimension is fully covered in the
            # cache tile.
            unswitch[body](self.conv_shape.c == c_tile_size)

        # After the loop can't be stepped with micro_kernel_height,
        # it will step by 5, 4, 3, 2, 1. This works with micro_kernel_height > 6
        # but maybe not very efficient.
        tile[
            n_ho_wo_iteration,
            VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1),
        ](
            self.partition_offsets[0],
            self.partition_offsets[0] + self.partition_sizes[0],
        )

    fn _inner_loops[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        c_fully_cached: Bool,
        has_residual: Bool,
    ](
        self,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        n_ho_wo: Int,
    ):
        assert_param[
            not has_residual or (has_residual and micro_kernel_width == 1),
            "Use Height x 1 kernel for residual in F.",
        ]()

        alias simd_size = simdwidthof[type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size
        # Base input offsets.
        let input_base_offsets = Buffer[
            micro_kernel_height, DType.int32
        ].stack_allocation()

        # Set input base offsets, corresponding to r=s=0
        @always_inline
        @parameter
        fn set_input_base_offsets[idx: Int]():
            let HO_WO = self.conv_shape.out_h * self.conv_shape.out_w
            let n = (n_ho_wo + idx) // HO_WO
            let ho_wo = (n_ho_wo + idx) % HO_WO
            # Global wo, ho index.
            let ho = ho_wo // self.conv_shape.out_w
            let wo = ho_wo % self.conv_shape.out_w
            # Translate ho, wo to hi, wi/range
            let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]
            let w = wo * self.conv_shape.stride[1] - self.conv_shape.pad_w[0]
            input_base_offsets.simd_store[1](
                idx,
                c_tile_offset
                + self.conv_shape.c
                * (w + self.conv_shape.w * (h + self.conv_shape.h * n)),
            )

        unroll[micro_kernel_height, set_input_base_offsets]()

        alias alignment = alignof[SIMD[type, simd_size]]()
        let output_micro_tile = NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            type,
        ].aligned_stack_allocation[alignment]()

        for f in range(
            f_tile_offset, f_tile_offset + f_tile_size, micro_kernel_f_size
        ):
            if c_tile_offset == self.partition_offsets[1]:
                self._init_output_micro_tile[
                    micro_kernel_height, micro_kernel_width, simd_size
                ](output_micro_tile)
            else:
                self._load_output_micro_tile[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    has_residual,
                ](
                    self.output.data.offset(n_ho_wo * self.conv_shape.f + f),
                    output_micro_tile,
                )

            var filter_ptr: DTypePointer[type] = self.filter.data

            @parameter
            if filter_packed:
                filter_ptr = self.filter.data.offset(
                    f
                    * self.conv_shape.r
                    * self.conv_shape.s
                    * self.conv_shape.c
                    + c_tile_offset * micro_kernel_f_size
                )

            for r in range(self.conv_shape.r):
                for s in range(self.conv_shape.s):
                    var input_offset = self.conv_shape.c * (
                        s + self.conv_shape.w * r
                    )
                    # Unpacked version. For each (r, s), we first offset the
                    # filter pointer by (r, s) plus c_tile_offset. Later for
                    # each c, we access micro_kernel_f_size contiguous elements.
                    # These contiguous segments are strided by F.
                    @parameter
                    if not filter_packed:
                        filter_ptr = self.filter.data.offset(
                            (s + r * self.conv_shape.s)
                            * self.conv_shape.c
                            * self.conv_shape.f
                            + c_tile_offset * self.conv_shape.f
                            + f
                        )

                    for c in range(c_tile_size):
                        # Prefetch. The offset specifies how much ahead in terms of
                        # micro kernel width do we prefetch.
                        alias prefetch_offset = 4
                        # fmt: off
                        let dist = prefetch_offset * micro_kernel_f_size \
                                    if c_fully_cached or c < c_tile_size - prefetch_offset \
                                    else ( \
                                      prefetch_offset + self.conv_shape.c - c_tile_size \
                                    ) * micro_kernel_f_size
                        # fmt: on

                        @always_inline
                        @parameter
                        fn prefetch_body[idx: Int]():
                            filter_ptr.offset(dist + idx * simd_size).prefetch[
                                PrefetchOptions()
                                .for_read()
                                .high_locality()
                                .to_data_cache()
                            ]()

                        unroll[micro_kernel_width, prefetch_body]()

                        # Accumulate with register blocking.
                        self._micro_kernel[
                            micro_kernel_height,
                            micro_kernel_width,
                            simd_size,
                            has_residual,
                        ](
                            input_base_offsets,
                            input_offset,
                            filter_ptr,
                            output_micro_tile,
                        )

                        # Packed Version: filter elements are accessed contiguously
                        # This assumption is violated when there is residual blocks
                        # in F. The residual block has a smaller micro kernel
                        # width but the buffer is padded with zero to fill the
                        # default micro kernel width, e.x., 4 for Intel.
                        @parameter
                        if filter_packed:
                            filter_ptr = filter_ptr.offset(micro_kernel_f_size)
                        # Unpacked Version. Each c is mapped to F elements.
                        # Hence the stride between each micro tile is also F.
                        else:
                            filter_ptr = filter_ptr.offset(self.conv_shape.f)

                        input_offset += 1

                    # If the C dimension is fully covered in cache tile, then
                    # loop nests over RxSxc_tile_size fully covers RSC diemensions.
                    # With FRSCf layout, this ensures all micro_kernel_f_size
                    # segments are continuously in memory. If not, we need to
                    # reset filter_ptr to the next start of c segments in tile.
                    @parameter
                    if filter_packed and not c_fully_cached:
                        filter_ptr = filter_ptr.offset(
                            (self.conv_shape.c - c_tile_size)
                            * micro_kernel_f_size
                        )

            self._store_output_micro_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](
                output_micro_tile,
                self.output.data.offset(n_ho_wo * self.conv_shape.f + f),
            )

    @always_inline
    fn _init_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
    ](
        self,
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            type,
        ],
    ):
        """Initialize a micro tile to zero.
        Arguments:
            n_ho_wo: offset of micro tile in fused (n, ho, wo) dimension.
            f: offset of micro tile in F dimension.
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            output_micro_tile.simd_store[simd_size](
                Index(idx0, idx1 * simd_size), SIMD[type, simd_size](0.0)
            )

        unroll[micro_kernel_height, micro_kernel_width, body]()

    @always_inline
    fn _load_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_base: DTypePointer[type],
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            type,
        ],
    ):
        """Load a micro tile from the output buffer.
        Parameters:
            has_residual: True when F is not multiple of simd_size. The residual
              is loaded and padded with zero to fit a simd vector.

        Arguments:
            output_base: Point to micro tile start, (n, ho, wo, f).
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """
        var output_ptr = output_base

        @unroll
        for i in range(micro_kernel_height):

            @unroll
            for j in range(micro_kernel_width):

                @parameter
                if has_residual:
                    let residual = self.conv_shape.f - (
                        self.conv_shape.f // simd_size
                    ) * simd_size
                    output_micro_tile.simd_store[simd_size](
                        Index(i, j * simd_size),
                        partial_simd_load[type, simd_size](
                            output_ptr.offset(j * simd_size), 0, residual, 0.0
                        ),
                    )
                else:
                    output_micro_tile.simd_store[simd_size](
                        Index(i, j * simd_size),
                        output_ptr.offset(j * simd_size).simd_load[simd_size](),
                    )

            output_ptr = output_ptr.offset(self.conv_shape.f)

    @always_inline
    fn _micro_kernel[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        input_base_offsets: Buffer[micro_kernel_height, DType.int32],
        input_offset: Int,
        filter_ptr: DTypePointer[type],
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            type,
        ],
    ):
        alias alignment = alignof[SIMD[type, simd_size]]()

        @parameter
        @always_inline
        fn body[idx0: Int, idx1: Int]():
            # Broadcast a scalar from input to a simd vector.
            let input_val = self.input.data.offset(
                input_base_offsets[idx0].value + input_offset
            ).simd_load[1]()
            let input_vec = SIMD[type, simd_size](input_val)

            # Load a simd vector from filter.
            let filter_vec: SIMD[type, simd_size]
            # Partial load if filter is not multiple of simd_size.
            @parameter
            if has_residual and not filter_packed:
                let residual = self.conv_shape.f - (
                    self.conv_shape.f // simd_size
                ) * simd_size
                filter_vec = partial_simd_load[type, simd_size](
                    filter_ptr, 0, residual, 0.0
                )
            # It's always safe to load a full vector from packed filter because
            # the filter to padded to multiple simd_size during pre-packing.
            else:
                filter_vec = filter_ptr.offset(idx1 * simd_size).simd_load[
                    simd_size
                ]()

            # The following should be lifted to registers and show up as
            # FMA instructions.
            let output_micro_idx = Index(idx0, idx1 * simd_size)
            var output_vec = output_micro_tile.aligned_simd_load[
                simd_size, alignment
            ](output_micro_idx)
            output_vec = fma[type, simd_size](input_vec, filter_vec, output_vec)
            output_micro_tile.aligned_simd_store[simd_size, alignment](
                output_micro_idx, output_vec
            )

        unroll[micro_kernel_height, micro_kernel_width, body]()

    @always_inline
    fn _store_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            type,
        ],
        output_base: DTypePointer[type],
    ):
        """Store a micro tile from the output buffer.
        Parameters:
            has_residual: True when F is not multiple of simd_size. Only the
              residual elements within the simd vector are stored to output.

        Arguments:
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
            output_base: Point to micro tile start, (n, ho, wo, f).
        """
        var output_ptr = output_base

        @unroll
        for i in range(micro_kernel_height):

            @unroll
            for j in range(micro_kernel_width):
                let output_vec = output_micro_tile.simd_load[simd_size](
                    Index(i, j * simd_size)
                )

                @parameter
                if has_residual:
                    let residual = self.conv_shape.f - (
                        self.conv_shape.f // simd_size
                    ) * simd_size
                    partial_simd_store[type, simd_size](
                        output_ptr.offset(j * simd_size),
                        0,
                        residual,
                        output_vec,
                    )
                else:
                    output_ptr.offset(j * simd_size).simd_store[simd_size](
                        output_vec
                    )

            output_ptr = output_ptr.offset(self.conv_shape.f)

    @always_inline
    fn _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
    ](
        self,
        c_tile_size: Int,
        input_stride: Int,
        input_base: DTypePointer[type],
        filter_base: DTypePointer[type],
        output_ptr: DTypePointer[type],
    ):
        # Short circuit when the micro tile is in padding.
        @parameter
        if micro_kernel_height == 0:
            return

        var filter_ptr = filter_base

        for c in range(c_tile_size):

            alias micro_kernel_f_size = micro_kernel_width * simd_size
            alias prefetch_offset = 4 * micro_kernel_f_size

            # prefetch
            @unroll
            for i in range(micro_kernel_width):
                filter_base.offset(prefetch_offset + i * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

            alias alignment = alignof[SIMD[type, simd_size]]()

            @unroll
            for i in range(micro_kernel_height):

                @unroll
                for j in range(micro_kernel_width):
                    # Broadcast an scalar from input to a simd vector.
                    let input_val = input_base.offset(
                        c + i * input_stride
                    ).load()
                    let input_vec = SIMD[type, simd_size](input_val)
                    # Load a simd vector from filter.
                    let filter_vec = filter_ptr.offset(j * simd_size).simd_load[
                        simd_size
                    ]()
                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    var output_vec = output_ptr.offset(
                        i * micro_kernel_f_size + j * simd_size
                    ).aligned_simd_load[simd_size, alignment]()
                    output_vec = fma[type, simd_size](
                        input_vec, filter_vec, output_vec
                    )
                    output_ptr.offset(
                        i * micro_kernel_f_size + j * simd_size
                    ).aligned_simd_store[simd_size, alignment](output_vec)

            filter_ptr = filter_ptr.offset(micro_kernel_f_size)

    @always_inline
    fn _h_loop[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        """Loop over H dimension
        Each row is divied into three parts: (1) effected by left padding, (2)
        not effected by padding, (3) effected by right padding. Use pointwise
        micro kernel 1 x micro_kernel_width for (1) and (3) and exploits the
        default micro kernel for (2).
        """
        alias simd_size = simdwidthof[type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        # Divide each row into three part:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        let left_pad_impact_end = div_ceil(
            self.conv_shape.pad_w[0], self.conv_shape.stride[1]
        )
        let right_pad_impact_start = (
            self.conv_shape.w
            + self.conv_shape.pad_w[0]
            - self.conv_shape.s * self.conv_shape.dilation[1]
        ) // self.conv_shape.stride[1] + 1

        # fmt: off
        let filter_base = self.filter.data.offset(
            f_tile_offset * self.conv_shape.c * self.conv_shape.r * self.conv_shape.s + c_tile_offset * micro_kernel_f_size
        )
        # fmt: on

        let input_curr_image = self.input.data.offset(
            n * self.conv_shape.w * self.conv_shape.h * self.conv_shape.c
        )
        let output_curr_image = self.output.data.offset(
            n
            * self.conv_shape.out_w
            * self.conv_shape.out_h
            * self.conv_shape.f
        )

        for ho in range(
            self.partition_offsets[3],
            self.partition_offsets[3] + self.partition_sizes[3],
        ):
            let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]
            # Point to (n, 0, ho, c_tile_offset) mapped in input
            var input_base = input_curr_image.offset(
                c_tile_offset
                + self.conv_shape.c
                * (-self.conv_shape.pad_w[0] + self.conv_shape.w * h)
            )
            # Point to (n, 0, ho, f_tile_offset) mapped in input
            var output_base = output_curr_image.offset(
                f_tile_offset + self.conv_shape.f * self.conv_shape.out_w * ho
            )

            # region effected by left padding, [0, left_pad_impact_end)
            for wo in range(left_pad_impact_end):
                self._inner_loops_padding[
                    1, micro_kernel_width, True, has_residual
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    ho,
                    wo,
                )
                input_base = input_base.offset(
                    self.conv_shape.stride[1] * self.conv_shape.c
                )
                output_base = output_base.offset(self.conv_shape.f)

            # Region not effected by padding, [left_pad_impact_end, right_pad_impact_start)
            # Tile the middle points with default micro kernel.
            @always_inline
            @parameter
            fn update_middle[height: Int](wo: Int):
                self._inner_loops_padding[
                    height, micro_kernel_width, False, has_residual
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    ho,
                    wo,
                )
                input_base = input_base.offset(
                    height * self.conv_shape.stride[1] * self.conv_shape.c,
                )
                output_base = output_base.offset(height * self.conv_shape.f)

            tile[update_middle, VariadicList[Int](5, 4, 3, 2, 1)](
                left_pad_impact_end, right_pad_impact_start
            )

            # region effected by right padding, [right_pad_impact_start, WO)
            for wo in range(right_pad_impact_start, self.conv_shape.out_w):
                self._inner_loops_padding[
                    1, micro_kernel_width, True, has_residual
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    ho,
                    wo,
                )
                input_base = input_base.offset(
                    self.conv_shape.stride[1] * self.conv_shape.c
                )
                output_base = output_base.offset(self.conv_shape.f)

    @always_inline
    fn _inner_loops_padding[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        w_padding_impact: Bool,
        has_residual: Bool,
    ](
        self,
        input_base: DTypePointer[type],  # points to (ho, wo) mapped in input
        filter_base: DTypePointer[type],  # point to filter in cf tile
        output_base: DTypePointer[type],  # point to (ho, wo) in output
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        ho: Int,  # index in output height
        wo: Int,  # index in output width
    ):
        """Inner loop computation with padding
        Given input (ho, wo), this kernel accumulates over the stencil RxS.
        """
        assert_param[
            not has_residual or (has_residual and micro_kernel_width == 1),
            "Use Height x 1 kernel for residual in F.",
        ]()

        assert_param[
            not w_padding_impact
            or (w_padding_impact and micro_kernel_height == 1),
            "USE 1 x width kernel on boundary",
        ]()

        alias simd_size = simdwidthof[type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        # Shift in input when shifting 1 in filter S dimension.
        let input_shift = self.conv_shape.dilation[1] * self.conv_shape.c
        # WO dimension stride mapped in input.
        let wo_stride_in_input = self.conv_shape.stride[1] * self.conv_shape.c
        # Filter stride in S dimension in FRSCf
        let filter_S_stride = self.conv_shape.c * micro_kernel_f_size
        # Filter stride in F dimension in FRSCf
        let filter_F_stride = self.conv_shape.r * self.conv_shape.s * filter_S_stride

        # This will be all lifted to simd registers for FMA unless the micro
        # kernel is too large that spills named registers.
        alias alignment = alignof[SIMD[type, simd_size]]()
        let output_micro_tile = NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            type,
        ].aligned_stack_allocation[alignment]()

        var filter_tile = filter_base
        # TODO: #18141 Verify if using filter tile larger than micro_kernel_f_size
        # improve performance or not.
        for f in range(
            f_tile_offset, f_tile_offset + f_tile_size, micro_kernel_f_size
        ):
            # Initialize micro tile with 0 for its first use
            if c_tile_offset == self.partition_offsets[1]:
                self._init_output_micro_tile[
                    micro_kernel_height, micro_kernel_width, simd_size
                ](output_micro_tile)
            # Load micro tile from output buffer.
            else:
                self._load_output_micro_tile[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    has_residual,
                ](output_base.offset(f - f_tile_offset), output_micro_tile)

            # Shift in input H when shifting 1 in filter stencil' R dimension.
            var h_shift = 0
            # h index in input image
            let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]
            for r in range(self.conv_shape.r):
                # Skip if row falls in padding.
                if h + h_shift < 0 or h + h_shift >= self.conv_shape.h:
                    h_shift += self.conv_shape.dilation[0]
                    continue

                var input_ptr = input_base.offset(
                    h_shift * self.conv_shape.c * self.conv_shape.w
                )
                var filter_ptr = filter_tile.offset(
                    r * self.conv_shape.s * filter_S_stride
                )

                var w = wo * self.conv_shape.stride[1] - self.conv_shape.pad_w[
                    0
                ]
                for s in range(self.conv_shape.s):
                    # Skip neighbor points in padding, if current point is
                    # effected by padding.
                    @parameter
                    if w_padding_impact:
                        if w < 0 or w >= self.conv_shape.w:
                            w += self.conv_shape.dilation[1]
                            filter_ptr = filter_ptr.offset(filter_S_stride)
                            input_ptr = input_ptr.offset(input_shift)
                            continue

                    self._accumulate[
                        micro_kernel_height,
                        micro_kernel_width,
                        simd_size,
                    ](
                        c_tile_size,
                        wo_stride_in_input,
                        input_ptr,
                        filter_ptr,
                        output_micro_tile.data,
                    )

                    w += self.conv_shape.dilation[1]
                    filter_ptr = filter_ptr.offset(filter_S_stride)
                    input_ptr = input_ptr.offset(input_shift)

                h_shift += self.conv_shape.dilation[0]

            # Store the micro tile
            self._store_output_micro_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](output_micro_tile, output_base.offset(f - f_tile_offset))

            filter_tile = filter_tile.offset(filter_F_stride)


# ===----------------------------------------------------------------------=== #
# Direct Convolution Filter Packing                                            #
# ===----------------------------------------------------------------------=== #


fn get_packed_filter_shape[
    type: DType
](R: Int, S: Int, C: Int, F: Int, inout shape_ref: DynamicRankBuffer):
    """Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.
    """
    alias simd_size = simdwidthof[type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    @always_inline
    @parameter
    fn dispatch_type[int_type: DType]():
        let shape = shape_ref.to_ndbuffer[1, int_type]()
        shape[0] = div_ceil(F, micro_kernel_f_size)
        shape[1] = R
        shape[2] = S
        shape[3] = C
        shape[4] = micro_kernel_f_size

    shape_ref.type.dispatch_integral[dispatch_type]()


@always_inline
fn pack_filter[
    type: DType,
](
    filter: NDBuffer[4, DimList.create_unknown[4](), type],
    packed_filter: NDBuffer[5, DimList.create_unknown[5](), type],
):
    """This packs the filter form RSCF to FRSCf.
    Args:
        R, S, C, F - original filter dimensions
        filter: filter in RSCF layout.
        packed_filter: packed filter in FRScf layout. Here,
            F       - the index of continuous segments in micro kernel
            R, S, C - original R, S, C
            f       - the index within a continuous segments

    F is first broken down to segements of size micro_kernel_f_size, then the
    remainder is further divided by simd_size. The last residual elements if
    any is padded with zero to fill simd_size.
    """
    alias simd_size = simdwidthof[type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    let R = filter.dim[0]()
    let S = filter.dim[1]()
    let C = filter.dim[2]()
    let F = filter.dim[3]()

    debug_assert(F % simd_size == 0, "F must be multiple of simd size.")

    @always_inline
    @parameter
    fn pack[f_tile_size: Int](f_tile_start: Int):
        var packed_filter_ptr = packed_filter.data.offset(
            f_tile_start * R * S * C
        )
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    let filter_ptr = filter.data.offset(
                        f_tile_start + F * (c + C * (s + S * r))
                    )

                    @always_inline
                    @parameter
                    fn body[idx: Int]():
                        # Assume F is multiple of simd_size so that loading
                        # simd_size elements doesn't exceed bound.
                        let filter_vec = filter_ptr.offset(
                            idx * simd_size
                        ).simd_load[simd_size]()
                        packed_filter_ptr.offset(idx * simd_size).simd_store[
                            simd_size
                        ](filter_vec)

                    unroll[f_tile_size // simd_size, body]()

                    packed_filter_ptr = packed_filter_ptr.offset(f_tile_size)

    # If F % simd_size != 0, the following won't touch the remainder.
    tile[pack, VariadicList[Int](micro_kernel_f_size, simd_size)](0, F)

    # Check the remainder if any
    let F_round_by_simd = (F // simd_size) * simd_size
    let residual = F - F_round_by_simd

    # Handle the remaider if any
    if residual > 0:
        var packed_filter_ptr = packed_filter.data.offset(
            F_round_by_simd * R * S * C
        )
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    let filter_ptr = filter.data.offset(
                        F_round_by_simd + F * (c + C * (s + S * r))
                    )
                    # Load remainder elements and pad with zero to
                    # to fill a simd vector.
                    let filter_vec = partial_simd_load[type, simd_size](
                        filter_ptr, 0, residual, 0.0
                    )
                    packed_filter_ptr.simd_store[simd_size](filter_vec)
                    # Hence, packed filter is incremented by simd_size
                    packed_filter_ptr = packed_filter_ptr.offset(simd_size)

    # Set the remaining memory to zero.
    let filter_size_roundup = R * S * C * (
        F_round_by_simd + simd_size if residual > 0 else F
    )
    let remaining = packed_filter.num_elements() - filter_size_roundup
    if remaining > 0:
        memset_zero[type](
            packed_filter.data.offset(filter_size_roundup), remaining
        )


@always_inline
fn conv_shape[
    input_rank: Int,
    filter_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    filter_buf: NDBuffer[
        filter_rank, DimList.create_unknown[filter_rank](), filter_type
    ],
    strides_buf: NDBuffer[1, DimList.create_unknown[1](), strides_type],
    dilations_buf: NDBuffer[1, DimList.create_unknown[1](), dilations_type],
) -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `conv` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        filter_rank: Rank of the filter tensor.
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter tensor.
        strides_buf: The filter tensor.
        dilations_buf: The filter tensor.

    Returns:
        The output shape.
    """

    # TODO(#17512)
    debug_assert(input_rank == 4, "input rank must be 4")
    debug_assert(input_rank == filter_rank, "input rank must match filter rank")
    debug_assert(
        strides_buf.dim(0) == input_rank - 2
        and dilations_buf.dim(0) == input_rank - 2,
        "strides and dilations size must be input rank - 2",
    )

    # Assume input has layout NHWC
    let batch_size = input_buf.dim(0)
    let input_channels = input_buf.dim(3)
    # Assume filter has layout RSCF
    let filter_channels = filter_buf.dim(2)
    let output_channels = filter_buf.dim(3)

    # TODO(#17512)
    debug_assert(
        input_channels == filter_channels,
        "channel dimension of input and filter must match",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[input_rank]()
    output_shape[0] = batch_size
    output_shape[1] = get_sliding_window_out_dim(
        input_buf.dim(1),
        filter_buf.dim(0),
        dilations_buf[0].to_int(),
        strides_buf[0].to_int(),
    )
    output_shape[2] = get_sliding_window_out_dim(
        input_buf.dim(2),
        filter_buf.dim(1),
        dilations_buf[1].to_int(),
        strides_buf[1].to_int(),
    )
    output_shape[3] = output_channels

    return output_shape


# must be register passable because it is used as a parameter
@register_passable("trivial")
struct ConvInfoStatic:
    var pad_h: DimList
    var pad_w: DimList
    var stride: DimList
    var dilation: DimList

    fn __init__(
        pad_h: DimList, pad_w: DimList, stride: DimList, dilation: DimList
    ) -> Self:
        return Self {
            pad_h: pad_h, pad_w: pad_w, stride: stride, dilation: dilation
        }


struct ConvInfo[conv_info_static: ConvInfoStatic]:
    var pad_h: OptionalParamInts[2, conv_info_static.pad_h]
    var pad_w: OptionalParamInts[2, conv_info_static.pad_w]
    var stride: OptionalParamInts[2, conv_info_static.stride]
    var dilation: OptionalParamInts[2, conv_info_static.dilation]

    fn __init__(
        inout self,
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
    ):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride = stride
        self.dilation = dilation


fn conv_2d_nhwc_direct[
    filter_rank: Int,
    filter_packed: Bool,
    conv_info_static: ConvInfoStatic,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
](
    input: NDBuffer[4, input_shape, input_type],
    filter: NDBuffer[filter_rank, filter_shape, filter_type],
    output: NDBuffer[4, output_shape, output_type],
    conv_info: ConvInfo[conv_info_static],
    out_chain: OutputChainPtr,
):
    assert_param[
        input_type == filter_type and input_type == output_type,
        "conv input/output/filter types must be the same",
    ]()
    assert_param[
        (filter_packed and filter_rank == 5)
        or (not filter_packed and filter_rank == 4),
        "unexpected filter rank for filter layout",
    ]()

    let output_rebind = rebind[NDBuffer[4, output_shape, input_type]](output)
    let filter_rebind = rebind[NDBuffer[filter_rank, filter_shape, input_type]](
        filter
    )
    alias filter_layout = Image2DLayout.FRSCf if filter_packed else Image2DLayout.RSCF
    let conv_shape = get_conv2d_shape[
        filter_rank,
        output_shape,
        input_shape,
        filter_shape,
        input_type,
        Image2DLayout.NHWC,
        filter_layout,
    ](
        output_rebind,
        input,
        filter_rebind,
        conv_info.pad_h.get(),
        conv_info.pad_w.get(),
        conv_info.stride.get(),
        conv_info.dilation.get(),
    )

    ConvDirectNHWC[
        filter_rank,
        input_shape,
        filter_shape,
        output_shape,
        input_type,
        filter_packed,
    ].run(output_rebind, input, filter_rebind, conv_shape, out_chain)
