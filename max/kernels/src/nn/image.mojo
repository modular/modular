# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from layout import Layout, LayoutTensor, RuntimeTuple, UNKNOWN_VALUE
from layout.int_tuple import fill_like

from utils.index import IndexList


# Padding handling method.
@fieldwise_init
@register_passable("trivial")
struct PadHandling(ImplicitlyCopyable, Movable):
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
@fieldwise_init
@register_passable("trivial")
struct Image2DLayout(ImplicitlyCopyable, Movable):
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
    layout: Layout,
    dtype: DType,
    static_image_layout: Image2DLayout,
    origin: MutableOrigin,
]:
    """Utility class that generalizes conv2d data and filter tensor with a given
    data layout."""

    var data: LayoutTensor[dtype, layout, origin]
    var dynamic_image_layout: Image2DLayout

    fn __init__(
        out self,
        data: LayoutTensor[dtype, layout, origin],
        layout: Image2DLayout,
    ):
        """Construct of an image data instance with dynamic layout param.

        Args:
            data: A 4d buffer containing the actual data.
            layout: Data layout tag.
        """
        constrained[static_image_layout == Image2DLayout.UNKNOWN]()
        self.data = data
        self.dynamic_image_layout = layout

    fn __init__(out self, data: LayoutTensor[dtype, layout, origin]):
        constrained[static_image_layout != Image2DLayout.UNKNOWN]()
        self.data = data
        self.dynamic_image_layout = static_image_layout

    fn to_static_layout[
        new_static_image_layout: Image2DLayout
    ](self) -> ImageData[layout, dtype, new_static_image_layout, origin]:
        """Conversion utility from a fully dynamic data structure, e.g. from c
        shim to one with compile-time known data layout.

        Returns:
            The image data with static data layout.
        """
        constrained[static_image_layout == Image2DLayout.UNKNOWN]()
        return ImageData[layout, dtype, new_static_image_layout](self.data)

    fn get_image_layout(self) -> Image2DLayout:
        """The getter function of the underlying data layout, resolving from
        either statically or dynamically provided information.

        Returns:
            The resolved data layout tag for this image instance.
        """
        if static_image_layout == Image2DLayout.UNKNOWN:
            return self.dynamic_image_layout
        return static_image_layout

    fn _get_index(self, n: Int, c: Int, h: Int, w: Int) -> Int:
        """Converts the general index to the actual index into the underlying
        data based on the tensor layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.

        Returns:
            A IndexList containing the index based on the underlying
            data layout.
        """
        if self.get_image_layout() == Image2DLayout.NCHW:
            return Int(
                self.data.runtime_layout(
                    RuntimeTuple[fill_like(self.layout.shape, UNKNOWN_VALUE)](
                        IndexList[4](n, c, h, w)
                    )
                )
            )
        if self.get_image_layout() == Image2DLayout.RSCF:
            return Int(
                self.data.runtime_layout(
                    RuntimeTuple[fill_like(self.layout.shape, UNKNOWN_VALUE)](
                        IndexList[4](h, w, c, n)
                    )
                )
            )
        return Int(
            self.data.runtime_layout(
                RuntimeTuple[fill_like(self.layout.shape, UNKNOWN_VALUE)](
                    IndexList[4](n, h, w, c)
                )
            )
        )

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

        var image_shape = ImageShape(self)

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
        if static_image_layout == Image2DLayout.NCHW:
            return _compute_index_nchw()
        elif static_image_layout == Image2DLayout.NHWC:
            return _compute_index_nhwc()

        debug_assert(False, "Invalid layout")
        return 0

    fn get_tuple_index(self, idx: Int) -> IndexList[4]:
        """Converts the flat index to the dimension index of the underlying
        data based on the tensor layout.

        Args:
            idx: Flat index.

        Returns:
            A IndexList containing the index in NCHW order.
        """

        var image_shape = ImageShape(self)

        @always_inline
        @__copy_capture(image_shape)
        @parameter
        fn _compute_index_nchw() -> IndexList[4]:
            # Index [N,C,H,W]
            var lidx = idx
            var w_idx = lidx % image_shape.W
            lidx = lidx // image_shape.W
            var h_idx = lidx % image_shape.H
            lidx = lidx // image_shape.H
            var c_idx = lidx % image_shape.C
            lidx = lidx // image_shape.C
            var n_idx = lidx
            return IndexList[4](n_idx, c_idx, h_idx, w_idx)

        @always_inline
        @__copy_capture(image_shape)
        @parameter
        fn _compute_index_nhwc() -> IndexList[4]:
            # Index [N,H,W,C]
            var lidx = idx
            var c_idx = lidx % image_shape.C
            lidx = lidx // image_shape.C
            var w_idx = lidx % image_shape.W
            lidx = lidx // image_shape.W
            var h_idx = lidx % image_shape.H
            lidx = lidx // image_shape.H
            var n_idx = lidx
            return IndexList[4](n_idx, c_idx, h_idx, w_idx)

        @parameter
        if static_image_layout == Image2DLayout.NCHW:
            return _compute_index_nchw()
        elif static_image_layout == Image2DLayout.NHWC:
            return _compute_index_nhwc()

        debug_assert(False, "Invalid layout")
        return 0

    fn __getitem__(self, n: Int, c: Int, h: Int, w: Int) -> Scalar[dtype]:
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
        return self.data.ptr[self._get_index(n, c, h, w)]

    fn __setitem__(self, n: Int, c: Int, h: Int, w: Int, value: Scalar[dtype]):
        """Writes the underlying data buffer based on the tensor index and under-
        lying data layout.

        Args:
            n: Index on the batch dimension.
            c: Index on the channel dimension.
            h: Index on the height dimension.
            w: Index on the width dimension.
            value: The value to store at the given index position.
        """
        self.data.ptr[self._get_index(n, c, h, w)] = value

    fn num_elements(self) -> Int:
        return self.data.size()


@register_passable("trivial")
struct ImageShape(ImplicitlyCopyable, Movable):
    """A data-layout agnostic representation of tensor shapes used in conv2d."""

    var N: Int
    var C: Int
    var H: Int
    var W: Int

    fn __init__[
        layout: Layout,
        dtype: DType,
        image_layout: Image2DLayout,
    ](out self, image_data: ImageData[layout, dtype, image_layout]):
        """Constructor of an ImageShape instance from an ImageData.

        Args:
            image_data: The image data instance to extract shape
              info from.
        """

        if image_data.get_image_layout() == Image2DLayout.NCHW:
            self.N = image_data.data.dim[0]()
            self.C = image_data.data.dim[1]()
            self.H = image_data.data.dim[2]()
            self.W = image_data.data.dim[3]()

        elif image_data.get_image_layout() == Image2DLayout.NHWC:
            self.N = image_data.data.dim[0]()
            self.C = image_data.data.dim[3]()
            self.H = image_data.data.dim[1]()
            self.W = image_data.data.dim[2]()

        else:
            debug_assert(
                image_data.get_image_layout() == Image2DLayout.RSCF,
                "Invalid layout",
            )
            self.N = image_data.data.dim[3]()
            self.C = image_data.data.dim[2]()
            self.H = image_data.data.dim[0]()
            self.W = image_data.data.dim[1]()
