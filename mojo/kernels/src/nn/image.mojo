# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer

from utils.index import StaticIntTuple
from buffer.list import DimList


# Padding handling method.
@value
@register_passable("trivial")
struct PadHandling:
    var value: Int
    alias EXCLUDE_PAD = PadHandling(0)  # Do not count padding.
    alias INCLUDE_PAD = PadHandling(2)  # Count padding.

    @always_inline("nodebug")
    fn __eq__(self, rhs: PadHandling) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: PadHandling) -> Bool:
        return self.value != rhs.value


# Data layout encoding.
@value
@register_passable("trivial")
struct Image2DLayout:
    var value: Int
    alias UNKNOWN = Image2DLayout(-1)  # statically unknown layout.
    alias NHWC = Image2DLayout(0)  # channels last layout.
    alias NCHW = Image2DLayout(1)  # channels first layout.
    alias RSCF = Image2DLayout(2)  # TF filter layout for channels last input.
    alias FRSCf = Image2DLayout(3)  # packed filter, adopted from oneDNN

    @always_inline("nodebug")
    fn __eq__(self, rhs: Image2DLayout) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Image2DLayout) -> Bool:
        return self.value != rhs.value


@register_passable("trivial")
struct ImageData[
    shape: DimList,
    type: DType,
    static_layout: Image2DLayout,
]:
    """Utility class that generalizes conv2d data and filter tensor with a given
    data layout."""

    var data: NDBuffer[type, 4, shape]
    var dynamic_layout: Image2DLayout

    fn __init__(data: NDBuffer[type, 4, shape], layout: Image2DLayout) -> Self:
        """Construct of an image data instance with dynamic layout param.

        Args:
            data: A 4d buffer containing the actual data.
            layout: Data layout tag.

        Returns:
            An ImageData instance.
        """
        constrained[static_layout == Image2DLayout.UNKNOWN]()
        return Self {data: data, dynamic_layout: layout}

    fn __init__(data: NDBuffer[type, 4, shape]) -> Self:
        constrained[static_layout != Image2DLayout.UNKNOWN]()
        return Self {data: data, dynamic_layout: static_layout}

    fn to_static_layout[
        new_static_layout: Image2DLayout
    ](self) -> ImageData[shape, type, new_static_layout]:
        """Conversion utility from a fully dynamic data structure, e.g. from c
        shim to one with compile-time known data layout.

        Returns:
            The image data with static data layout.
        """
        constrained[static_layout == Image2DLayout.UNKNOWN]()
        return ImageData[shape, type, new_static_layout](self.data)

    fn get_layout(self) -> Image2DLayout:
        """The getter function of the underlying data layout, resolving from
        either staticall or dynamicall information.

        Returns:
            The resolved data layout tag for this image instance.
        """
        if static_layout == Image2DLayout.UNKNOWN:
            return self.dynamic_layout
        return static_layout

    fn _get_index(self, n: Int, c: Int, h: Int, w: Int) -> StaticIntTuple[4]:
        """Converts the general index to the actual index into the underlying
        data based on the tensor layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            A StaticIntTuple containing the index based on the underlying
            data layout.
        """
        if self.get_layout() == Image2DLayout.NCHW:
            return StaticIntTuple[4](n, c, h, w)
        if self.get_layout() == Image2DLayout.RSCF:
            return StaticIntTuple[4](h, w, c, n)
        return StaticIntTuple[4](n, h, w, c)

    fn get_flat_index(self, n: Int, c: Int, h: Int, w: Int) -> Int:
        """Converts the dimension index to the flat index of the underlying
        data based on the tensor layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            An integer containing the index based on the underlying
            data layout.
        """

        var image_shape = ImageShape.__init__[shape, type, static_layout](self)

        @always_inline
        @__copy_capture(image_shape)
        @parameter
        fn _compute_index_nchw() -> Int:
            # Index [N,C,H,W]
            var idx = n
            idx = idx * image_shape.C + c
            idx = idx * image_shape.H + h
            idx = idx * image_shape.W + w
            return idx

        @always_inline
        @__copy_capture(image_shape)
        @parameter
        fn _compute_index_nhwc() -> Int:
            # Index [N,H,W,C]
            var idx = n
            idx = idx * image_shape.H + h
            idx = idx * image_shape.W + w
            idx = idx * image_shape.C + c
            return idx

        @parameter
        if static_layout == Image2DLayout.NCHW:
            return _compute_index_nchw()
        elif static_layout == Image2DLayout.NHWC:
            return _compute_index_nhwc()

        debug_assert(False, "Invalid layout")
        return 0

    fn get_tuple_index(self, idx: Int) -> StaticIntTuple[4]:
        """Converts the flat index to the dimension index of the underlying
        data based on the tensor layout.

        Args:
            idx: Flat index.

        Returns:
            A StaticIntTuple containing the index in NCHW order.
        """

        var image_shape = ImageShape.__init__[shape, type, static_layout](self)

        @always_inline
        @__copy_capture(image_shape)
        @parameter
        fn _compute_index_nchw() -> StaticIntTuple[4]:
            # Index [N,C,H,W]
            var lidx = idx
            var w_idx = lidx % image_shape.W
            lidx = lidx // image_shape.W
            var h_idx = lidx % image_shape.H
            lidx = lidx // image_shape.H
            var c_idx = lidx % image_shape.C
            lidx = lidx // image_shape.C
            var n_idx = lidx
            return StaticIntTuple[4](n_idx, c_idx, h_idx, w_idx)

        @always_inline
        @__copy_capture(image_shape)
        @parameter
        fn _compute_index_nhwc() -> StaticIntTuple[4]:
            # Index [N,H,W,C]
            var lidx = idx
            var c_idx = lidx % image_shape.C
            lidx = lidx // image_shape.C
            var w_idx = lidx % image_shape.W
            lidx = lidx // image_shape.W
            var h_idx = lidx % image_shape.H
            lidx = lidx // image_shape.H
            var n_idx = lidx
            return StaticIntTuple[4](n_idx, c_idx, h_idx, w_idx)

        @parameter
        if static_layout == Image2DLayout.NCHW:
            return _compute_index_nchw()
        elif static_layout == Image2DLayout.NHWC:
            return _compute_index_nhwc()

        debug_assert(False, "Invalid layout")
        return 0

    fn __getitem__(self, n: Int, c: Int, h: Int, w: Int) -> Scalar[type]:
        """Reads the underlying data buffer based on the tensor index and under-
        lying data layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            The value stored at the given index position.
        """
        return self.data[self._get_index(n, c, h, w)]

    fn __setitem__(self, n: Int, c: Int, h: Int, w: Int, value: Scalar[type]):
        """Writes the underlying data buffer based on the tensor index and under-
        lying data layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.
            value: The value to store at the given index position.
        """
        self.data[self._get_index(n, c, h, w)] = value

    fn num_elements(self) -> Int:
        return self.data.size()


@value
@register_passable("trivial")
struct ImageShape:
    """A data-layout agnostic representation of tensor shapes used in conv2d."""

    var N: Int
    var C: Int
    var H: Int
    var W: Int

    fn __init__[
        shape: DimList,
        type: DType,
        layout: Image2DLayout,
    ](image_data: ImageData[shape, type, layout]) -> ImageShape:
        """Constructor of an ImageShape instance from an ImageData.

        Args:
            image_data: The image data instance to extract shape
              info from.

        Returns:
            An ImageShape instance containing the shape info.
        """

        if image_data.get_layout() == Image2DLayout.NCHW:
            return ImageShape {
                N: image_data.data.dim[0](),
                C: image_data.data.dim[1](),
                H: image_data.data.dim[2](),
                W: image_data.data.dim[3](),
            }

        elif image_data.get_layout() == Image2DLayout.NHWC:
            return ImageShape {
                N: image_data.data.dim[0](),
                C: image_data.data.dim[3](),
                H: image_data.data.dim[1](),
                W: image_data.data.dim[2](),
            }

        else:
            debug_assert(
                image_data.get_layout() == Image2DLayout.RSCF, "Invalid layout"
            )
            return ImageShape {
                N: image_data.data.dim[3](),
                C: image_data.data.dim[2](),
                H: image_data.data.dim[0](),
                W: image_data.data.dim[1](),
            }
