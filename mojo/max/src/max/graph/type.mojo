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
"""Library for graph Symbol Types."""

from collections import Optional

import _mlir
from max.tensor import TensorSpec
from collections.string import StaticString

from utils.variant import Variant

import ._c


fn _dyn() -> Int64:
    return _c.dim_type_new_dynamic()


@value
struct DynamicDim(Copyable, Movable):
    """A dynamic tensor dimension.

    `DynamicDim`s are printed in MO tensor types as `?`, eg.
    `!mo.tensor<[?, 4, ?], si32]>` has 2 dynamic dimensions.

    Dynamic dimensions reduce the compiler's ability to reason about
    tensor shapes as data moves through the model, and may therefore
    limit the available optimizations it can perform. Reducing usage
    of dynamic dims can be an avenue to improving model performance.

    Create a dynamic dimension via `Dim.dynamic()`.
    """

    pass


@value
struct SymbolicDim(Copyable, Movable):
    """A symbolic tensor dimension.

    `SymbolicDims`s have a name and are printed as their name on MO types, eg.
    `!mo.tensor<[batch, x, 10], si32]>` the first and second dimensions are
    named "batch" and "x" respectively.

    Symbolic dimensions don't have a static value, but they allow a readable
    name to understand what's going on in the model IR better, and they also
    allow users to hint to the compiler that two dimensions will have the same
    value, which can often allow important speedups.

    Create a symbolic dimension via `Dim.symbolic("name")`, or just by passing
    the string name, eg. `TensorType(DType.bool, "batch", Dim.dynamic(), 10)`.
    """

    var name: String
    """The name of the dimension."""

    fn __eq__(self, other: SymbolicDim) -> Bool:
        """Whether the dimension is the same as another symbolic dimension.

        Symbolic dimensions with the same name are interpreted as the same
        dimensionality! If you use Symbolic dimensions, make sure you're naming
        them consistently, your model will likely fail to compile if you name
        two actually different dimensions the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions have the same name, False otherwise.
        """
        return self.name == other.name


@value
struct StaticDim(Copyable, Movable):
    """A static tensor dimension.

    Static tensor dimensions will always have exactly the same value,
    and are key to good model performance.

    Static dimensions can be created implicitly in most cases:
    `TensorType(DType.int64, 4, 5)` is a tensor with 2 static dimensions,
    `4` and `5` respectively.
    """

    var dim: Int64
    """The size of the static dimension."""

    @implicit
    fn __init__(out self, dim: Int):
        """Int conversion constructor.

        Args:
            dim: The size of the static dimension.
        """
        self.dim = dim

    fn __eq__(self, other: StaticDim) -> Bool:
        """Whether the dimension has the same size as another dimension.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if both dimensions have the same static size, False otherwise.
        """
        return self.dim == other.dim


@value
struct Dim(Copyable, Movable):
    """A tensor dimension.

    Tensor dimensions can be
    - Static, aka known size
    - Dynamic, aka unknown size
    - Symbolic, aka unknown size but named

    In most cases you don't need to work with a `Dim` directly, but can rely
    on conversion constructors, for instance you can specify a tensor type as

    ```mojo
    from max.graph import Dim, TensorType
    var tensor_type = TensorType(DType.int64, "batch", 10, Dim.dynamic())
    ```
    will create a tensor type with 3 dimensions: a symbolic "batch" dimension,
    a static dimension of size 10, and a dynamic dimension.
    ```

    You can still construct dimensions explicitly via helpers, eg.

    ```mojo
    var some_dims = [
        Dim.dynamic(),
        Dim.symbolic("batch"),
        Dim.static(5),
    ]
    ```

    Constraining tensor dimensions is one important way to improve model
    performance. If tensors have unknown dimensions, we can't optimize them
    as aggressively. Symoblic tensors allow the compiler to learn constraints
    on a specific dimension (eg. if 2 inputs have the same `batch` dimension)
    which can be an important improvement over dynamic dimensions, but static
    dims are the easiest to optimize and therefore the easiest to create
    and work with.
    """

    var value: Variant[DynamicDim, StaticDim, SymbolicDim]
    """The dimension data."""

    @implicit
    fn __init__(out self, dim: Int):
        """Int static dimension conversion constructor.

        Args:
            dim: The static size of the dimension.
        """
        self.value = StaticDim(dim)

    @implicit
    fn __init__(out self, name: StringLiteral):
        """Named symbolic dimension conversion constructor.

        Args:
            name: The name of the symbolic dimension.
        """
        self.value = SymbolicDim(String(name))

    @staticmethod
    fn static(dim: Int64) -> Self:
        """Explicitly constructs a static dimension.

        Args:
            dim: The static size of the dimension.

        Returns:
            A static dimension of size `dim`.
        """
        return Self(StaticDim(dim))

    @staticmethod
    fn symbolic(name: String) -> Self:
        """Explicitly constructs a symbolic dimension.

        Args:
            name: The unique name of the dimension.

        Returns:
            A symbolic dimension with the given name.
        """
        return Self(SymbolicDim(name))

    @staticmethod
    fn dynamic() -> Self:
        """Explicitly constructs a dynamic dimension.

        Returns:
            A dynamic dimension.
        """
        return Self(DynamicDim())

    fn is_dynamic(self) -> Bool:
        """Checks whether or not the dimension is a dynamic dimension.

        Returns:
            True if the dimension is dynamic, False otherwise.
        """
        return self.value.isa[DynamicDim]()

    fn is_static(self) -> Bool:
        """Checks whether or not the dimension is a static dimension.

        Returns:
            True if the dimension is static, False otherwise.
        """
        return self.value.isa[StaticDim]()

    fn is_symbolic(self) -> Bool:
        """Whether or not the dimension is a symbolic dimension.

        Returns:
            True if the dimension is symbolic, False otherwise.
        """
        return self.value.isa[SymbolicDim]()

    fn num_elements(self) -> Int64:
        """Returns the number of elements in the dimension, if known.

        Returns:
            For a static dimension, we return the known static dimension size.
            Otherwise, return an internal value representing an unknown
            dimension size.
        """
        return self.value[StaticDim].dim if self.is_static() else _dyn()

    fn maybe_num_elements(self) raises -> Optional[Int64]:
        """Returns the number of elements in the dimension, if known.

        Returns:
            For a static dimension, we return the known static dimension size.
            Otherwise, return None.
        """
        if self.is_static():
            return self.value[StaticDim].dim
        return None

    fn __eq__(self, other: Dim) -> Bool:
        """Checks whether two dimensions are equal.

        Dimensions are equal if they are the same dimension type
        (dynamic, symbolic, static). Additionally, static dimensions
        are only equal if their dimension is the same size, and symbolic
        dimensions are only equal if they have the same name.

        Args:
            other: The other dimension to check equality against.

        Returns:
            True if the dimensions are equal, False otherwise.
        """
        if self.value.isa[DynamicDim]():
            return other.value.isa[DynamicDim]()
        elif self.value.isa[SymbolicDim]():
            return (
                other.value.isa[SymbolicDim]()
                and self.value[SymbolicDim] == other.value[SymbolicDim]
            )
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            return (
                other.value.isa[StaticDim]()
                and self.value[StaticDim] == other.value[StaticDim]
            )

    fn __ne__(self, other: Dim) -> Bool:
        """Checks whether two dimensions are not equal.

        The inverse of __eq__.

        Args:
            other: The other dimension to check inequality against.

        Returns:
            False if the dimensions are equal, True otherwise.
        """
        return not (self == other)

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Attribute:
        """Creates an _mlir.Attribute representing this dimension.

        This is used internally when constructing tensor _mlir types.

        Args:
            ctx: The mlir.Context in which to create the attribute.

        Returns:
            A _mlir.Attribute in the context representing the dimension.
        """

        if self.value.isa[DynamicDim]():
            return _c.dim_new_dynamic(ctx)
        elif self.value.isa[SymbolicDim]():
            var name = self.value[SymbolicDim].name
            var result = _c.dim_new_symbolic(ctx, name)
            return result
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            var dim = self.value[StaticDim].dim
            return _c.dim_new_static(ctx, dim)

    @staticmethod
    fn from_mlir(dim_attr: _mlir.Attribute) raises -> Dim:
        """Constructs a dimension from an _mlir Attribute.

        Args:
            dim_attr: The _mlir Attribute object to parse into a dimension.

        Returns:
            The dimension represented by the _mlir Attr value.
        """
        if _c.dim_is_dynamic(dim_attr):
            return Dim.dynamic()
        elif _c.dim_is_static(dim_attr):
            return Dim.static(_c.dim_static_value(dim_attr))
        elif _c.dim_is_symbolic(dim_attr):
            return Dim.symbolic(String(_c.dim_symbolic_name(dim_attr)))
        else:
            debug_assert(
                _c.dim_is_algebraic(dim_attr),
                "unknown dim variant",
            )
            raise "Unsupported dim type: algebraic dimension"

    fn __str__(self) -> String:
        """Creates a string representation of the dimension.

        Returns:
            A human-readable string of the dimension.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats a description of the DeviceMemory to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        if self.value.isa[DynamicDim]():
            return writer.write("?")
        elif self.value.isa[SymbolicDim]():
            return writer.write(self.value[SymbolicDim].name)
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            return writer.write(self.value[StaticDim].dim)


@value
struct TensorType(Copyable, Movable):
    """A symbolic tensor type.

    It is _not_ an eager tensor type!! It contains no actual data, but instead
    represents a value at some point in time during model execution.

    Most internal values in a model will be tensors. This type represents
    their element type (dtype) and dimensions (dims) at a specific point during
    model computation. It allows us to do some optimistic optimizations and
    shape inference during graph construction, and to provide more detailed
    shape information to the compiler for further optimizatino passes.

    It can also represent a fully dynamic rank tensor. The presence of dynamic
    rank tensors in a graph will often degrade performance dramatically and
    prevents many classes of optimizations.
    """

    var dtype: DType
    """The element type of the tensor value."""
    var dims: List[Dim]
    """The dimensions of the tensor value, if it is known-rank."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    @implicit
    fn __init__(out self, dtype: DType):
        """Constructs a 0-d tensor type.

        Args:
            dtype: The element type of the tensor data.
        """
        self.dtype = dtype
        self.dims = List[Dim]()

    fn __init__(out self, dtype: DType, *dims: Dim):
        """Constructs a tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.dims = List[Dim](capacity=len(dims))
        for d in dims:
            self.dims.append(d[])

    fn __init__(out self, dtype: DType, dims: List[Dim]):
        """Constructs a ranked tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.dims = dims

    # ===------------------------------------------------------------------=== #
    # Auxiliary factories
    # ===------------------------------------------------------------------=== #

    @implicit
    fn __init__(out self, spec: TensorSpec):
        """Constructs a tensor type from a TensorSpec.

        Since TensorSpec can only contain static shapes, this will always
        construct a static tensor.

        Args:
            spec: The dtype and static shape of the tensor.
        """
        var dims = List[Dim](capacity=spec.rank())
        for i in range(spec.rank()):
            dims.append(Dim.static(spec[i]))
        self = Self(spec.dtype(), dims)

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        var dims = List[_mlir.Attribute](capacity=len(self.dims))
        for i in range(len(self.dims)):
            dims.append(self.dims[i].to_mlir(ctx))
        return _c.tensor_type_new(
            ctx,
            _c.dtype_new(ctx, self.dtype),
            dims,
            ranked=True,
        )

    @staticmethod
    fn from_mlir(t: _mlir.Type) raises -> Self:
        """Constructs a tensor type from an _mlir type.

        Args:
            t: The _mlir Type object to parse into a tensor type.

        Returns:
            The tensor type represented by the _mlir Type value.
        """
        var dtype = _c.tensor_type_get_dtype(t)
        var ranked = _c.tensor_type_is_ranked(t)
        if not ranked:
            raise "Unranked tensor types are unsupported!"

        var rank = _c.tensor_type_get_rank(t)
        var dims = List[Dim](capacity=Int(rank))
        for i in range(rank):
            var dim_attr = _c.tensor_type_get_dim(t, i)
            dims.append(Dim.from_mlir(dim_attr))
        return Self(dtype, dims)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn is_static(self) -> Bool:
        """Checks whether the tensor type has a fully static shape or not.

        A tensor must have all of its dimensions be `static` (or be 0-dimensional)
        in order to be `static`.

        Returns:
            True if the tensor has a fully static shape, False otherwise.
        """
        for i in range(self.rank()):
            if self.dims[i].is_dynamic():
                return False
        return True

    fn rank(self) -> Int:
        """Gets the rank of the tensor type.

        Returns:
            The tensor's static rank, or 0 for a dynamic tensor.
            A 0-dimensional static tensor also has a rank of 0, so
            check `ranked` directly to check if a tensor is ranked or not.
        """
        return len(self.dims)

    fn dim(self, pos: Int) raises -> Dim:
        """Gets the pos'th dimension of the tensor type.

        Supports negative-indexing, ie. `t.dim(-1)` will give the last
        dimension.

        Args:
            pos: The dimension index to retrieve.

        Returns:
            The dimension value at dimension `pos`.

        Raises:
            If the dimension is out-of-bounds, or if the tensor is unranked.
        """
        return self.dims[pos + (self.rank() if pos < 0 else 0)]

    fn __eq__(self, other: TensorType) -> Bool:
        """Checks whether the two tensors are identical (same rank, type, shape).

        Args:
            other: The other tensor to check equality against.

        Returns:
            True if the tensors have identical element type and shape,
            False otherwise.
        """
        if self.dtype != other.dtype:
            return False
        if self.rank() != other.rank():
            return False
        for i in range(self.rank()):
            if self.dims[i] != other.dims[i]:
                return False
        return True

    # ===------------------------------------------------------------------=== #
    # Utilities
    # ===------------------------------------------------------------------=== #

    fn num_elements(self) -> Int64:
        """Counts the total number of elements in the tensor type.

        For a static tensor, returns the product of all static dimensions.
        This is the number of elements the tensor will hold _during execution_,
        TensorType doesn't actually hold any element values at all.

        For any non-static tensor, ie. a tensor having any symbolic or dynamic
        dimensions, the return value will be meaningless.

        Returns:
            The number of elements the tensor contains.
        """

        var n: Int64 = 1
        for i in range(self.rank()):
            if not self.dims[i].is_static():
                return _dyn()
            n *= self.dims[i].num_elements()
        return n

    fn cast(self, dtype: DType) -> Self:
        """Constructs a new tensor type of the same shape with the new dtype.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, and the new element type.
        """
        return Self(dtype, self.dims)


@value
struct ListType(Copyable, Movable):
    """A type representing a flat list of tensor values.

    This isn't an eager list type! It doesn't contain any data, but represents
    a runtime list that contains tensors.
    """

    var eltype: TensorType
    """The tensor type of elements in the list.

    The list can currently only hold tensors, and it has a single known tensor
    type. If the list can contain tensors of different shapes, they need to
    conform to that tensor type. For instance, a list with tensors of shapes
    `[batch, x, 2, 4]` and `[batch, 1, 2, ?]` would need to represent its type
    as `[batch, ?, 2, ?]`, holding dynamic dimensions for those that vary among
    its list elements.

    Lists may not contain tensor elements of different ranks.
    """

    @always_inline
    @implicit
    fn __init__(out self, eltype: TensorType):
        self.eltype = eltype

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        return _c.list_type_new(ctx, self.eltype.to_mlir(ctx))


@value
struct _OpaqueType(Copyable, Movable):
    """A type representing an opaque type."""

    var name: String

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        return _c.opaque_type_new(ctx, self.name)

    @staticmethod
    fn from_mlir(t: _mlir.Type) -> Self:
        """Constructs an opaque type from an _mlir type.

        Args:
            t: The _mlir Type object to parse into an opaque type.

        Returns:
            The opaque type represented by the _mlir Type value.
        """
        var name = String(_c.opaque_type_name(t))
        return Self(name)


@value
struct Type(Copyable, Movable):
    """Represents any possible type for Graph Symbol values.

    Every Symbol has a Type, and that type is represented by an Type.
    This type may be inspected to get finer-grained types and learn more
    about an individual Value.
    """

    var type: Variant[TensorType, ListType, _OpaqueType]
    """The type data."""

    @implicit
    fn __init__(out self, t: TensorType):
        """Constructs a type from a tensor type.

        Args:
            t: The tensor type.
        """
        self.type = t

    @implicit
    fn __init__(out self, t: ListType):
        """Constructs a type from a list type.

        Args:
            t: The list type.
        """
        self.type = t

    @implicit
    fn __init__(out self, t: _OpaqueType):
        """Constructs a type from an opaque typ.

        Args:
            t: The opaque type.
        """
        self.type = t

    fn list(self) raises -> ListType:
        """Extracts the type as a list type.

        This doesn't have any impact at graph execution time, it just retrieves
        the underlying list type for a type which is a list.

        Returns:
            The underlying type specifically as a list type.

        Raises:
            If the type is some other data type besides a list.
        """
        if not self.type.isa[ListType]():
            raise "Not a list type!"
        return self.type[ListType]

    fn tensor(self) raises -> TensorType:
        """Extracts the type as a tensor type.

        This doesn't have any impact at graph execution time, it just retrieves
        the underlying tensor type for a type which is a tensor.

        Returns:
            The underlying type specifically as a tensor type.

        Raises:
            If the type is some other data type besides a tensor.
        """
        if not self.type.isa[TensorType]():
            raise "Not a tensor type!"
        return self.type[TensorType]

    fn _opaque(self) raises -> _OpaqueType:
        """Extracts the type as an opaque type.

        This doesn't have any impact at graph execution time, it just retrieves
        the underlying opaque type.

        Returns:
            The underlying type specifically as an opaque type.

        Raises:
            If the type is some other data type besides an an opaque type.
        """
        if not self.type.isa[_OpaqueType]():
            raise "Not an opaque type!"
        return self.type[_OpaqueType]

    fn dims(self) -> List[Dim]:
        """Returns a list of all dims referenced by the type.

        This doesn't have any impact at graph execution time, it just retrieves
        the list of referenced dimensions.

        This will only return a result if the underlying type is a TensorType.

        Returns:
            The dims referenced.
        """
        if not self.type.isa[TensorType]():
            return List[Dim]()

        return self.type[TensorType].dims

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        if self.type.isa[TensorType]():
            return self.type[TensorType].to_mlir(ctx)
        elif self.type.isa[ListType]():
            return self.type[ListType].to_mlir(ctx)
        else:
            debug_assert(self.type.isa[_OpaqueType](), "MO type variants")
            return self.type[_OpaqueType].to_mlir(ctx)

    @staticmethod
    fn from_mlir(t: _mlir.Type) raises -> Self:
        """Constructs a type from an _mlir type.

        Args:
            t: The _mlir Type object to parse into a type.

        Returns:
            The type represented by the _mlir Type value.
        """
        if _c.type_is_list(t):
            var element_type = TensorType.from_mlir(
                _c.list_type_element_type(t)
            )
            return Self(ListType(element_type))
        elif _c.type_is_tensor(t):
            return Self(TensorType.from_mlir(t))
        else:
            debug_assert(_c.type_is_opaque(t), "MO type variants")
            return Self(_OpaqueType.from_mlir(t))
