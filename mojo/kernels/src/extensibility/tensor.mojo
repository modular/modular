# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import fma
from sys.info import simdwidthof
from sys.intrinsics import strided_load

from algorithm.functional import elementwise, vectorize
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory.unsafe import bitcast
from register import *

from utils._serialize import _serialize

from .tensor_helpers import InnerStride


@always_inline
@export
fn empty_tensor[
    type: DType, rank: Int
](shape: StaticIntTuple[rank]) -> Tensor[type, rank]:
    """Creates an empty [`Tensor`](/max/api/mojo/extensibility/Tensor)
    with the given shape.

    For example, here's how to create a new tensor that matches an input shape:

    ```mojo
    fn gelu[type: DType, rank: Int](x: Tensor[type, rank]) -> Tensor[type, rank]:
        var output = empty_tensor[type](x.shape)
        # modify the output tensor here...
        return output^
    ```

    Parameters:
        type: The tensor data type.
        rank: The tensor rank.

    Args:
        shape: The tensor shape.

    Returns:
        An empty [`Tensor`](/max/api/mojo/extensibility/Tensor) with
        the specified type and shape.
    """
    var ptr = UnsafePointer[Scalar[type]].alloc(shape.flattened_length())
    return Tensor[type, rank](ptr, shape)


struct Tensor[type: DType, static_rank: Int](Stringable, Formattable):
    """A tensor type designed to extend MAX Engine with custom ops.

    Beware that this `Tensor` is completely different from the `Tensor` type
    in the Mojo standard library. Currently, this `max.extensibility.Tensor`
    is designed only for use when building custom ops for MAX Engine.

    For example, here's how you can define a custom op with this `Tensor`:

    ```mojo
    from max.extensibility import Tensor, empty_tensor
    from max import register
    from math import erf, sqrt


    @register.op("my_gelu")
    fn gelu[type: DType, rank: Int](x: Tensor[type, rank]) -> Tensor[type, rank]:
        var output = empty_tensor[type](x.shape)

        @always_inline
        @parameter
        fn func[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
            var tmp = x.simd_load[width](i)
            return tmp / 2 * (1 + erf(tmp / sqrt(2)))

        output.for_each[func]()
        return output^
    ```

    Then, you must create a Mojo package with this op and load it with your
    model into MAX Engine. For more information, read about [MAX
    extensibility](/max/extensibility/).

    Parameters:
        type: DType of the underlying data.
        static_rank: The tensor rank.

    """

    var data: UnsafePointer[Scalar[type]]
    var shape: StaticIntTuple[static_rank]
    var strides: StaticIntTuple[static_rank]

    # Empty strides...
    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: StaticIntTuple[static_rank],
    ):
        """Constructs a new `Tensor`.

        You usually should not instantiate a `Tensor` directly. Instead use
        [`empty_tensor()`](/max/api/mojo/extensibility/empty_tensor).

        Args:
            ptr: A pointer to the tensor data.
            shape: The shape of the tensor.
        """
        self.data = ptr.address
        self.shape = shape
        self.strides = StaticIntTuple[static_rank]()

        var stride = 1

        # Walk backwards to compute the fully contiguous strides.
        @parameter
        for i in reversed(range(static_rank)):
            self.strides[i] = stride
            stride *= self.shape[i]

    @always_inline
    fn __init__(
        inout self,
        ptr: UnsafePointer[Scalar[type]],
        shape: StaticIntTuple[static_rank],
        strides: StaticIntTuple[static_rank],
    ):
        """Constructs a new `Tensor`.

        You usually should not instantiate a `Tensor` directly. Instead use
        [`empty_tensor()`](/max/api/mojo/extensibility/empty_tensor).

        Args:
            ptr: A pointer to the tensor data.
            shape: The shape of the tensor.
            strides: The stride size for each dimension.
        """
        self.data = ptr.address
        self.shape = shape
        self.strides = strides

    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data
        self.shape = existing.shape
        self.strides = existing.strides
        existing.data = UnsafePointer[Scalar[type]]()

    fn __del__(owned self):
        if self.data != UnsafePointer[Scalar[type]]():
            self.data.free()

    @always_inline
    fn nelems(self) -> Int:
        """Gets the number of elements in the tensor."""
        return self.shape.flattened_length()

    @always_inline
    fn rank(self) -> Int:
        """Gets the tensor rank."""
        return static_rank

    @always_inline
    fn _compute_flat_index(
        self,
        index: StaticIntTuple[static_rank],
    ) -> Int:
        var flat_index: Int = 0

        @parameter
        for i in range(static_rank):
            flat_index = fma(index[i], self.strides[i], flat_index)

        return flat_index

    @always_inline
    fn store[
        width: Int
    ](inout self, index: StaticIntTuple[static_rank], value: SIMD[type, width]):
        """Stores multiple values at the specified indices in the tensor.

        Parameters:
            width: The SIMD width.

        Args:
            index: The indices where to store the values.
            value: The values to store.
        """
        self._simd_store_internal(index, value)

    @always_inline
    fn store[width: Int](inout self, index: Int, value: SIMD[type, width]):
        """Stores a single value at the specified index in the tensor.

        Constraints:
            The tensor's `static_rank` must be `1`.

        Parameters:
            width: The SIMD width.

        Args:
            index: The index where to store the values.
            value: The values to store.
        """
        constrained[
            self.static_rank == 1,
            (
                "Single int access to kernels only allowed on tensors"
                " statically known to be 1D"
            ),
        ]()
        var as_nd = self.get_nd_indices()
        as_nd[0] = index
        self.store(as_nd, value)

    @always_inline
    fn _simd_store_internal[
        width: Int
    ](inout self, index: StaticIntTuple[static_rank], val: SIMD[type, width]):
        var flat_index = self._compute_flat_index(index)
        SIMD.store(self.data, flat_index, val)

    @always_inline
    fn get_nd_indices(self) -> StaticIntTuple[static_rank]:
        """Creates empty indices with the same rank as this tensor."""
        return StaticIntTuple[static_rank](0)

    @always_inline
    fn simd_load[simd_width: Int](self, index: Int) -> SIMD[type, simd_width]:
        """Gets the values stored in the tensor at the given index.

        Constraints:
            The tensor's `static_rank` must be `1`.

        Parameters:
            simd_width: The SIMD width.

        Args:
            index: The index where the values are stored.

        Returns:
            The values as a [`SIMD`](/mojo/stdlib/builtin/simd/SIMD).
        """
        constrained[
            static_rank == 1,
            (
                "Single int access to kernels only allowed on tensors"
                " statically known to be 1D"
            ),
        ]()
        var as_nd = self.get_nd_indices()
        as_nd[0] = index
        return self.simd_load[simd_width](as_nd)

    @always_inline
    fn simd_load[
        simd_width: Int,
    ](self, index: StaticIntTuple[static_rank]) -> SIMD[type, simd_width]:
        """Gets the values stored in the tensor at the given indices.

        Parameters:
            simd_width: The SIMD width.

        Args:
            index: The indices where the values are stored.

        Returns:
            The values as a [`SIMD`](/mojo/stdlib/builtin/simd/SIMD).
        """
        return self._simd_load_internal[simd_width](index)

    @always_inline
    fn _simd_load_internal[
        simd_width: Int
    ](self, index: StaticIntTuple[static_rank]) -> SIMD[type, simd_width]:
        var flat_index = self._compute_flat_index(index)
        var stride = self.strides[self.rank() - 1]

        @parameter
        @always_inline
        fn _load[
            stride_type: InnerStride
        ](stride: Int) -> SIMD[type, simd_width]:
            @parameter
            if stride_type == InnerStride.Broadcast:
                return SIMD[size=simd_width].load(self.data, flat_index)
            elif stride_type == InnerStride.Contiguous:

                @parameter
                if type is DType.bool:
                    var v = SIMD[size=simd_width].load(
                        self.data.bitcast[DType.uint8](), flat_index
                    )
                    return v.cast[type]()
                else:
                    return SIMD[size=simd_width].load(self.data, flat_index)
            else:

                @parameter
                if type is DType.bool:
                    var v = strided_load[DType.uint8, simd_width](
                        self.data.bitcast[DType.uint8]()
                        .offset(flat_index)
                        .address,
                        stride,
                    )
                    return v.cast[type]()
                else:
                    return strided_load[type, simd_width](
                        self.data.offset(flat_index).address, stride
                    )

        if self.strides[self.rank() - 1] == 0:
            return _load[InnerStride.Broadcast](stride)
        elif stride > 1:
            return _load[InnerStride.Strided](stride)
        return _load[InnerStride.Contiguous](stride)

    @no_inline
    fn for_each[
        func: fn[_width: Int] (StaticIntTuple[static_rank]) capturing -> SIMD[
            type, _width
        ],
    ](inout self):
        """Executes a lambda for every element in the tensor.

        The lambda function must take an `Int` parameter for the SIMD width,
        and a `StaticIntTuple` argument for tensor indices, and then return a
        `SIMD` with the mutated value at the given indeces.

        Parameters:
            func: The lambda function to execute for each tensor indeces.
        """
        alias simd_width = simdwidthof[Self.type]()

        @always_inline
        @parameter
        fn elementwise_fn_wrapper[
            width: Int, rank: Int
        ](coords_static: StaticIntTuple[rank]) capturing:
            var coords = rebind[StaticIntTuple[static_rank]](coords_static)
            var val = func[width](coords)
            self.store(coords, val)

        elementwise[elementwise_fn_wrapper, simd_width](self.shape)

    @no_inline
    fn __str__(self) -> String:
        """Gets the tensor as a string.

        Returns:
            A compact string of the tensor.
        """

        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        """
        Formats this Tensor to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write("Tensor(")

        @parameter
        fn serialize[T: Formattable](val: T):
            writer.write(val)

        var shape = List[Int]()
        for i in range(self.rank()):
            shape.append(self.shape[i])

        _serialize[serialize_fn=serialize, serialize_end_line=False](
            self.data.address, shape
        )

        writer.write(")")
