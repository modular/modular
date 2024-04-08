# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for graph Symbol Types."""

from tensor import TensorSpec
from utils.variant import Variant

import _mlir

from .module import Module
import ._c


fn _dyn() -> Int64:
    return _c.dim_type_new_dynamic()


@value
struct DynamicDim(CollectionElement):
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
struct SymbolicDim(CollectionElement):
    """A symbolic tensor dimension.

    `SymbolicDims`s have a name and are printed as their name on MO types, eg.
    `!mo.tensor<[batch, x, 10], si32]>` the first and second dimensions are
    named "batch" and "x" respectively.

    Symbolic dimensions don't have a static value, but they allow a readable
    name to understand what's going on in the model IR better, and they also
    allow users to hint to the compiler that two dimensions will have the same
    value, which can often allow important speedups.

    Create a symbolic dimension via `Dim.symbolic("name")`, or just by passing
    the string name, eg. `MOTensor(DType.bool, "batch", Dim.dynamic(), 10)`.
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
struct StaticDim(CollectionElement):
    """A static tensor dimension.

    Static tensor dimensions will always have exactly the same value,
    and are key to good model performance.

    Static dimensions can be created implicitly in most cases:
    `MOTensor(DType.int64, 4, 5)` is a tensor with 2 static dimensions,
    `4` and `5` respectively.
    """

    var dim: Int64
    """The size of the static dimension."""

    fn __init__(inout self, dim: Int):
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
struct Dim(CollectionElement):
    """A tensor dimension.

    Tensor dimensions can be
    - Static, aka known size
    - Dynamic, aka unknown size
    - Symbolic, aka unknown size but named

    In most cases you don't need to work with a `Dim` directly, but can rely
    on conversion constructors, for instance you can specify a tensor type as

    ```mojo
    from max.graph import Dim, MOTensor
    var tensor_type = MOTensor(DType.int64, "batch", 10, Dim.dynamic())
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

    fn __init__(inout self, dim: Int):
        """Int static dimension conversion constructor.

        Args:
            dim: The static size of the dimension.
        """
        self.value = StaticDim(dim)

    fn __init__(inout self, name: StringLiteral):
        """StringLiteral symbolic dimension conversion constructor.

        Args:
            name: The name of the symbolic dimension.
        """
        self.value = SymbolicDim(name)

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
        return self.value.get[StaticDim]()[].dim if self.is_static() else _dyn()

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
                and self.value.get[SymbolicDim]()[]
                == other.value.get[SymbolicDim]()[]
            )
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            return (
                other.value.isa[StaticDim]()
                and self.value.get[StaticDim]()[]
                == other.value.get[StaticDim]()[]
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
            A _mlir.Attribute in the Module's context representing the dimension.
        """

        if self.value.isa[DynamicDim]():
            return _c.dim_new_dynamic(ctx)
        elif self.value.isa[SymbolicDim]():
            var name = self.value.get[SymbolicDim]()[].name
            var result = _c.dim_new_symbolic(ctx, name._strref_dangerous())
            name._strref_keepalive()
            return result
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            var dim = self.value.get[StaticDim]()[].dim
            return _c.dim_new_static(ctx, dim)

    fn __str__(self) -> String:
        """Creates a string representation of the dimension.

        Returns:
            A human-readable string of the dimension.
        """
        if self.value.isa[DynamicDim]():
            return "?"
        elif self.value.isa[SymbolicDim]():
            return self.value.get[SymbolicDim]()[].name
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            return str(self.value.get[StaticDim]()[].dim)


trait MOType:
    """An internal helper trait for _mlir construction.

    MOTypes have methods to help us convert between our structured types
    and their _mlir representations.
    """

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        ...

    fn to_string(self, ctx: _mlir.Context) -> String:
        """Converts to a maybe-human-readable string.

        Args:
            ctx: An mlir.Context to help with string construction.

        Returns:
            A string representation of the type.
        """
        ...


@value
struct ElementType(MOType):
    """The element type of a data container, like a tensor or scalar.

    Prefer to use the standard library DType and implicitly convert
    to this type rather than using it directly.
    """

    var dtype: DType
    """The underlying dtype."""

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        return _c.dtype_new(ctx, self.dtype)

    fn to_string(self, ctx: _mlir.Context) -> String:
        """Converts to a maybe-human-readable string.

        Args:
            ctx: An mlir.Context to help with string construction.

        Returns:
            A string representation of the type.
        """
        return str(self.to_mlir(ctx))


@value
struct MOTensor(MOType, CollectionElement):
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

    var dtype: ElementType
    """The element type of the tensor value."""
    var dims: List[Dim]
    """The dimensions of the tensor value, if it is known-rank."""
    var ranked: Bool
    """Whether the tensor has a known static rank or not."""

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, dtype: ElementType, *dims: Dim):
        """Constructs a ranked tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.dims = List[Dim](capacity=len(dims))
        for d in dims:
            self.dims.append(d[])
        self.ranked = True

    fn __init__(inout self, dtype: ElementType, ranked: Bool):
        """Constructs a fully dynamic tensor or 0-dimensional tensor type.

        Args:
            dtype: The element type of the tensor data.
            ranked: If False, create a fully dynamic tensor.
                    If True, create a rank 0 tensor. This is the same as calling
                    the constructor with just a dtype argument.
        """
        self.dtype = dtype
        self.dims = List[Dim]()
        self.ranked = ranked

    fn __init__(inout self, dtype: ElementType, dim: Int):
        """Constructs a rank-1 static tensor type.

        This is a temporary overload that exists to prevent an overload
        ambiguity via implicit conversion to a DynamicTensor. It's functionally
        the same as the ranked tensor variadic constructor.

        Args:
            dtype: The element type of the tensor data.
            dim: The static shape of the singular dimension.
        """
        self.__init__(dtype, Dim(dim))

    fn __init__(inout self, dtype: ElementType, dims: List[Dim]):
        """Constructs a ranked tensor type.

        Args:
            dtype: The element type of the tensor data.
            dims: The shape dimensions of the tensor. The number of dims
                  is the rank of the tensor.
        """
        self.dtype = dtype
        self.dims = dims
        self.ranked = True

    # ===------------------------------------------------------------------=== #
    # Auxiliary factories
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, spec: TensorSpec):
        """Constructs a tensor type from a TensorSpec.

        Since TensorSpec can only contain static shapes, this will always
        construct a static tensor.

        Args:
            spec: The dtype and static shape of the tensor.
        """
        var dims = List[Dim](capacity=spec.rank())
        for i in range(spec.rank()):
            dims.append(Dim.static(spec[i]))
        self.__init__(spec.dtype(), dims)

    # ===------------------------------------------------------------------=== #
    # MOType trait
    # ===------------------------------------------------------------------=== #

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
            self.dtype.to_mlir(ctx),
            dims,
            self.ranked,
        )

    fn to_string(self, ctx: _mlir.Context) -> String:
        """Converts to a maybe-human-readable string.

        Args:
            ctx: An mlir.Context to help with string construction.

        Returns:
            A string representation of the type.
        """
        return str(self.to_mlir(ctx))

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
        if ranked:
            var rank = _c.tensor_type_get_rank(t)
            var dims = List[Dim](capacity=rank.to_int())
            for i in range(rank):
                var dim_attr = _c.tensor_type_get_dim(t, i)
                var dim: Dim
                if _c.dim_is_dynamic(dim_attr):
                    dim = Dim.dynamic()
                elif _c.dim_is_static(dim_attr):
                    dim = Dim.static(_c.dim_static_value(dim_attr))
                elif _c.dim_is_symbolic(dim_attr):
                    dim = Dim.symbolic(str(_c.dim_symbolic_name(dim_attr)))
                else:
                    debug_assert(
                        _c.dim_is_symbolic_expression(dim_attr),
                        "unknown dim variant",
                    )
                    raise "Unsupported dim type: symbolic expression"
                dims.append(dim)
            return Self(dtype, dims)
        else:
            return Self(dtype, ranked)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn is_static(self) -> Bool:
        """Checks whether the tensor type has a fully static shape or not.

        This is _not_ the same as `ranked`. A tensor must be both `ranked`
        and have all of its dimensions be `static` (or be 0-dimensional)
        in order to be `static`.

        Returns:
            True if the tensor has a fully static shape, False otherwise.
        """
        if not self.ranked:
            return False
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
        if not self.ranked:
            raise "Cannot get dim of unranked type"
        return self.dims[pos + (self.rank() if pos < 0 else 0)]

    fn __eq__(self, other: MOTensor) -> Bool:
        """Checks whether the two tensors are identical (same rank, type, shape).

        Args:
            other: The other tensor to check equality against.

        Returns:
            True if the tensors have identical element type and shape,
            False otherwise.
        """
        if self.dtype.dtype != other.dtype.dtype:
            return False
        if self.ranked != other.ranked:
            return False
        if self.ranked:
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
        MOTensor doesn't actually hold any element values at all.

        For any non-static tensor, ie. a tensor having dynamic rank
        or having any symbolic or dynamic dimensions, the return value will
        be meaningless.

        Returns:
            The number of elements the tensor contains.
        """

        var n: Int64 = 1
        for i in range(self.rank()):
            if not self.dims[i].is_static():
                return _dyn()
            n *= self.dims[i].num_elements()
        return n

    fn cast(self, dtype: ElementType) -> Self:
        """Constructs a new tensor type of the same shape with the new dtype.

        Args:
            dtype: The new element type for the tensor.

        Returns:
            A new tensor type with the same shape, and the new element type.
        """
        return Self(dtype, self.dims, self.ranked)


@value
struct MOList(MOType, CollectionElement):
    """A type representing a flat list of tensor values.

    This isn't an eager list type! It doesn't contain any data, but represents
    a runtime list that contains tensors.
    """

    var eltype: MOTensor
    """The tensor type of elements in the list.

    The list can currently only hold tensors, and it has a single known tensor
    type. If the list can contain tensors of different shapes, they need to
    conform to that tensor type. For instance, a list with tensors of shapes
    `[batch, x, 2, 4]` and `[batch, 1, 2, ?]` would need to represent its type
    as `[batch, ?, 2, ?]`, holding dynamic dimensions for those that vary among
    its list elements.

    A list that can contain tensors of different ranks must be typed as holding
    unranked tensors.
    """

    # ===------------------------------------------------------------------=== #
    # MOType trait
    # ===------------------------------------------------------------------=== #

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        return _c.list_type_new(ctx, self.eltype.to_mlir(ctx))

    fn to_string(self, ctx: _mlir.Context) -> String:
        """Converts to a maybe-human-readable string.

        Args:
            ctx: An mlir.Context to help with string construction.

        Returns:
            A string representation of the type.
        """
        return str(self.to_mlir(ctx))


@value
struct AnyMOType(MOType, CollectionElement):
    """Represents any possible type for Graph Symbol values.

    Every Symbol has a Type, and that type is represented by an AnyMOType.
    This type may be inspected to get finer-grained types and learn more
    about an individual Value.
    """

    var type: Variant[MOTensor, MOList]
    """The type data."""

    fn __init__(inout self, t: MOTensor):
        """Constructs a type from a tensor type.

        Args:
            t: The tensor type.
        """
        self.type = t

    fn __init__(inout self, t: MOList):
        """Constructs a type from a list type.

        Args:
            t: The list type.
        """
        self.type = t

    fn list(self) raises -> MOList:
        """Extracts the type as a list type.

        This doesn't have any impact at graph execution time, it just retrieves
        the underlying list type for a type which is a list.

        Returns:
            The underlying type specifically as a list type.

        Raises:
            If the type is some other data type besides a list.
        """
        if not self.type.isa[MOList]():
            raise "Not a list type!"
        return self.type.get[MOList]()[]

    fn tensor(self) raises -> MOTensor:
        """Extracts the type as a tensor type.

        This doesn't have any impact at graph execution time, it just retrieves
        the underlying tensor type for a type which is a tensor.

        Returns:
            The underlying type specifically as a tensor type.

        Raises:
            If the type is some other data type besides a tensor.
        """
        if not self.type.isa[MOTensor]():
            raise "Not a tensor type!"
        return self.type.get[MOTensor]()[]

    fn to_mlir(self, ctx: _mlir.Context) -> _mlir.Type:
        """Converts to an _mlir.Type instance.

        Args:
            ctx: The mlir.Context in which to create the type.

        Returns:
            An _mlir.Type in the specified Context.
        """
        if self.type.isa[MOTensor]():
            return self.type.get[MOTensor]()[].to_mlir(ctx)
        else:
            debug_assert(self.type.isa[MOList](), "MO type variants")
            return self.type.get[MOList]()[].to_mlir(ctx)

    @staticmethod
    fn from_mlir(t: _mlir.Type) raises -> Self:
        """Constructs a type from an _mlir type.

        Args:
            t: The _mlir Type object to parse into a type.

        Returns:
            The type represented by the _mlir Type value.
        """
        if _c.type_is_list(t):
            var element_type = MOTensor.from_mlir(_c.list_type_element_type(t))
            return Self(MOList(element_type))
        else:
            debug_assert(_c.type_is_tensor(t), "MO type variants")
            return Self(MOTensor.from_mlir(t))

    fn to_string(self, ctx: _mlir.Context) -> String:
        """Converts to a maybe-human-readable string.

        Args:
            ctx: An mlir.Context to help with string construction.

        Returns:
            A string representation of the type.
        """
        return str(self.to_mlir(ctx))


@value
struct TypeTuple(Sized):
    """A sequence of 0 or more types.

    This is a helper type for graph construction.
    """

    var elts: List[AnyMOType]
    """The sequence of types."""

    # ===------------------------------------------------------------------=== #
    # Basic constructors and accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *elts: AnyMOType):
        """Constructs a TypeTuple from any number of types.

        Args:
            elts: The sequence of types.
        """
        self.elts = List[AnyMOType]()
        for t in elts:
            self.elts.append(t[])

    # ===------------------------------------------------------------------=== #
    # Convenience adapters
    # ===------------------------------------------------------------------=== #

    # TODO: Most should go away when one can express (AnyMOType, AnyMOType)

    fn __init__(inout self, t: MOTensor):
        """Constructs a 1-element TypeTuple from a tensor type.

        Args:
            t: The tensor type.
        """
        self.elts = List[AnyMOType]()
        self.elts.append(t)

    fn __init__(inout self, t: MOList):
        """Constructs a 1-element TypeTuple from a list type.

        Args:
            t: The list type.
        """
        self.elts = List[AnyMOType]()
        self.elts.append(t)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        """Gets the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """
        return len(self.elts)

    # ===------------------------------------------------------------------=== #
    # _mlir conversion
    # ===------------------------------------------------------------------=== #

    fn to_mlir(self, ctx: _mlir.Context) -> List[_mlir.Type]:
        """Converts to a sequence of _mlir.Type instances.

        Args:
            ctx: The mlir.Context in which to create the types.

        Returns:
            A list of _mlir.Types representing the tuple's types.
        """
        var retval = List[_mlir.Type]()
        for i in range(len(self.elts)):
            retval.append(self.elts[i].to_mlir(ctx))
        return retval

    # ===------------------------------------------------------------------=== #
    # Mutators
    # ===------------------------------------------------------------------=== #

    fn append(inout self, type: AnyMOType):
        """Appends a type to the back of the tuple.

        Args:
            type: The type to add to the tuple.
        """
        self.elts.append(type)
