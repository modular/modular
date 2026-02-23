# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Unified layout system for mixed compile-time and runtime indices."""

from os import abort
from sys.intrinsics import _type_is_eq

from builtin.variadics import (
    Variadic,
    VariadicPack,
    _ReduceVariadicAndIdxToVariadic,
    _ReduceValueAndIdxToVariadic,
    _ReduceVariadicAndIdxToValue,
    _MapVariadicAndIdxToType,
)
from buffer.dimlist import Dim, DimList
from sys.intrinsics import _type_is_eq_parse_time


trait CoordLike(
    Defaultable, ImplicitlyCopyable, Representable, TrivialRegisterPassable
):
    """Trait for unified layout handling of compile-time and runtime indices."""

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike]
    comptime static_value: Int
    comptime is_static_value = False
    comptime is_tuple = False
    comptime is_value = not Self.is_tuple
    comptime DTYPE = DType.invalid

    # Note that unlike the __len__() from Sized, this is a static method.
    @staticmethod
    fn __len__() -> Int:
        """Get the number of elements in this type.

        Returns:
            The number of elements (1 for single values, >1 for tuples).
        """
        ...

    fn __repr__(self) -> String:
        """Get the string representation of this type."""
        ...

    fn value(self) -> Int:
        """Get the value of this type.
        Only valid for value types.
        """
        ...

    fn tuple(var self) -> Coord[*Self.VariadicType]:
        """Get the value of this type.
        Only valid for tuple types.
        """
        ...

    fn product(self) -> Int:
        """Calculate the product of all elements.

        Returns:
            The product of all elements.
        """
        ...

    fn sum(self) -> Int:
        """Calculate the sum of all elements.

        Returns:
            The sum of all elements.
        """
        ...


struct ComptimeInt[val: Int](CoordLike, TrivialRegisterPassable):
    """Compile-time known index value.

    Parameters:
        val: The compile-time integer value.
    """

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike] = Tuple[
        Self
    ].element_types
    comptime static_value: Int = Self.val
    comptime DTYPE = DType.int
    comptime is_static_value = True

    fn __init__(out self):
        """Initialize a compile-time integer with the specified value."""
        pass

    @staticmethod
    @always_inline("nodebug")
    fn __len__() -> Int:
        return 1

    fn __repr__(self) -> String:
        return String("ComptimeInt[", self.value(), "]()")

    @always_inline("nodebug")
    fn product(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn sum(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn value(self) -> Int:
        return Self.val

    @always_inline("nodebug")
    fn tuple(var self) -> Coord[*Self.VariadicType]:
        comptime assert False, "ComptimeInt is not a tuple type"
        return rebind[Coord[*Self.VariadicType]](self)


struct RuntimeInt[dtype: DType = DType.int](CoordLike, TrivialRegisterPassable):
    """Runtime index value with configurable precision.

    Parameters:
        dtype: The data type for the runtime integer value. Defaults to `DType.int`.
    """

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike] = Tuple[
        Self
    ].element_types
    comptime static_value: Int = -1
    comptime DTYPE = Self.dtype

    var _value: Scalar[Self.dtype]
    """The runtime scalar value."""

    fn __init__(out self):
        self._value = 0

    fn __init__(out self, value: Scalar[Self.dtype]):
        """Initialize a runtime integer with the given value.

        Args:
            value: The scalar value to store.
        """
        self._value = value

    @staticmethod
    @always_inline("nodebug")
    fn __len__() -> Int:
        return 1

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        return String("RuntimeInt(", self.value(), ")")

    @always_inline("nodebug")
    fn product(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn sum(self) -> Int:
        return self.value()

    @always_inline("nodebug")
    fn value(self) -> Int:
        return Int(self._value)

    @always_inline("nodebug")
    fn tuple(var self) -> Coord[*Self.VariadicType]:
        comptime assert False, "RuntimeInt is not a tuple type"
        return rebind[Coord[*Self.VariadicType]](self)


fn Idx(value: Int) -> RuntimeInt[DType.int]:
    """Helper to create runtime indices.

    Args:
        value: The integer value for the runtime index.

    Returns:
        A `RuntimeInt` instance with the specified value.

    Usage: Idx(5) creates a RuntimeInt with value 5.
    """
    return RuntimeInt[DType.int](Scalar[DType.int](value))


fn Idx[value: Int]() -> ComptimeInt[value]:
    """Helper to create compile-time indices.

    Parameters:
        value: The compile-time integer value.

    Returns:
        A `ComptimeInt` instance with the specified compile-time value.

    Usage: Idx[5]() creates a ComptimeInt with value 5.
    """
    return ComptimeInt[value]()


fn Idx(
    value: IntLiteral,
) -> ComptimeInt[
    Int(
        mlir_value=__mlir_attr[
            `#pop.cast_to_builtin<#pop.int_literal_convert<`,
            value.value,
            `> : !pop.scalar<index>> : index`,
        ]
    )
]:
    """Helper to create compile-time indices.

    Args:
        value: The compile-time integer value.

    Returns:
        A `ComptimeInt` instance with the specified compile-time value.

    Usage: Idx[5]() creates a ComptimeInt with value 5.
    """
    return {}


fn Idx(
    value: Scalar,
) -> RuntimeInt[value.dtype] where value.dtype.is_integral():
    """Helper to create runtime indices.
    Args:
        value: The integer value for the runtime index.
    Returns:
        A `RuntimeInt` instance with the specified value.
    Usage: Idx(5) creates a RuntimeInt with value 5.
    """
    return RuntimeInt[value.dtype](value)


@fieldwise_init("implicit")
struct Coord[*element_types: CoordLike](CoordLike, Sized, Writable):
    """A struct representing tuple-like data with compile-time and runtime elements.

    Parameters:
        element_types: The variadic pack of element types that implement `CoordLike`.
    """

    comptime VariadicType: Variadic.TypesOfTrait[CoordLike] = Self.element_types
    comptime static_value: Int = -1
    comptime is_tuple = True
    comptime all_dims_known = _AllStatic[*Self.element_types]
    comptime static_product = _StaticProduct[*Self.element_types]
    comptime rank = Variadic.size(Self.element_types)
    comptime flat_rank = Variadic.size(_Flattened[*Self.element_types])

    var _storage: _RegTuple[*Self.element_types]
    """The underlying MLIR storage for the tuple elements."""

    fn __init__(out self):
        """
        Empty initialize a tensor with static dims.
        """
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        comptime for i in range(self.rank):
            self[i] = Self.element_types[i]()

    fn __init__[
        rank: Int, dtype: DType
    ](
        out self: Coord[*_Splatted[RuntimeInt[dtype], rank]],
        index_list: std.utils.IndexList[rank, element_type=dtype],
    ):
        self = type_of(self)()

        comptime for i in range(rank):
            UnsafePointer(to=self[i]).init_pointee_copy(
                rebind[type_of(self[i])](
                    RuntimeInt[dtype](Scalar[dtype](index_list[i]))
                )
            )

    @staticmethod
    @always_inline("nodebug")
    fn size() -> Int:
        """Get the total number of elements including nested ones.

        Returns:
            The total count of all elements.
        """
        var count = 0

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            count += T.__len__()

        return count

    @staticmethod
    fn __len__() -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """

        comptime result = Variadic.size(Self.element_types)
        return result

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        var result = String("Coord(")

        comptime for i in range(Self.__len__()):
            result += self[i].__repr__()
            if i < Self.__len__() - 1:
                result += String(", ")
        return result + String(")")

    fn __len__(self) -> Int:
        """Get the length of the tuple.

        Returns:
            The number of elements in the tuple.
        """
        return Self.__len__()

    @always_inline("nodebug")
    fn __init__(out self, var *args: * Self.element_types):
        """Construct tuple from variadic arguments.

        Args:
            args: Values for each element.
        """
        self = Self(storage=args^)

    @implicit
    @always_inline("nodebug")
    fn __init__(out self, var tuple: Tuple[*Self.element_types]):
        """Construct from a Tuple with matching element types.

        Args:
            tuple: The Tuple to construct from.
        """
        self = Self()

        comptime for i in range(Self.rank):
            self._storage[i] = tuple[i]

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        var storage: VariadicPack[_, CoordLike, *Self.element_types],
    ):
        """Construct from a low-level variadic pack.

        Args:
            storage: The variadic pack storage to construct from.
        """
        var t = _RegTuple(
            storage=rebind_var[
                VariadicPack[
                    elt_is_mutable = type_of(storage).elt_is_mutable,
                    origin = type_of(storage).origin,
                    type_of(storage).is_owned,
                    TrivialRegisterPassable,
                    *Self.element_types,
                ]
            ](storage^)
        )

        self._storage = rebind[_RegTuple[*Self.element_types]](t)

    @always_inline("nodebug")
    fn __getitem__[
        idx: Int
    ](ref self) -> ref[self._storage] Self.element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element index to access.

        Returns:
            A reference to the specified element.
        """
        return self._storage[idx]

    @always_inline("nodebug")
    fn product(self) -> Int:
        var result = 1

        comptime for i in range(Self.__len__()):
            result *= self[i].product()

        return result

    @always_inline("nodebug")
    fn sum(self) -> Int:
        var result = 0

        comptime for i in range(Self.__len__()):
            result += self[i].sum()

        return result

    @always_inline("nodebug")
    fn value(self) -> Int:
        comptime assert False, "Coord is not a value type"
        abort()

    @always_inline("nodebug")
    fn inner_product(self, t: IntTuple) -> Int:
        """Calculate the inner product with an IntTuple.

        Args:
            t: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """

        var result = 0
        debug_assert(
            Self.__len__() == t.__len__(),
            "Length of Coord (",
            Self.__len__(),
            ") and IntTuple (",
            t.__len__(),
            ") must match",
        )

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            var t_elem = t[i]

            comptime if T.is_tuple:
                debug_assert(
                    t_elem.is_tuple(),
                    "Type mismatch: expected tuple in t[",
                    i,
                    "] but got value",
                )
                result += Coord(self[i]).inner_product(t_elem)
            else:
                debug_assert(
                    not t_elem.is_tuple(),
                    "Type mismatch: expected value in t[",
                    i,
                    "] but got tuple",
                )
                result += self[i].value() * t_elem.value()
        return result

    @always_inline("nodebug")
    fn inner_product[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Int:
        """Calculate the inner product with another CoordLike.

        Parameters:
            other_types: The types of the other value.

        Args:
            other: The other value to compute inner product with.

        Returns:
            The inner product of the two values.
        """
        comptime assert Self.__len__() == Coord[*other_types].__len__(), (
            "Length of Coord ("
            + String(Self.__len__())
            + ") and Coord[*other_types] ("
            + String(Coord[*other_types].__len__())
            + ") must match"
        )
        var result = 0

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            comptime if T.is_tuple and U.is_tuple:
                result += Coord(self[i]).inner_product(Coord(other[i]))
            elif T.is_value and U.is_value:
                result += self[i].value() * other[i].value()
            else:
                constrained[
                    False,
                    String(
                        "Element ",
                        i,
                        " of Coord must both be a tuple or both be a value",
                    ),
                ]()

        return result

    @always_inline("nodebug")
    fn __eq__[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Bool:
        """Check if this tuple's elements are equal to the other tuple's elements.
        """

        comptime assert Self.__len__() == Coord[*other_types].__len__(), (
            "Length of Coord ("
            + String(Self.__len__())
            + ") and Coord[*other_types] ("
            + String(Coord[*other_types].__len__())
            + ") must match"
        )

        comptime for i in range(Self.__len__()):
            comptime T = Self.element_types[i]
            comptime U = other_types[i]

            comptime if T.is_tuple and U.is_tuple:
                if Coord(self[i]) != Coord(other[i]):
                    return False
            elif T.is_value and U.is_value:
                if self[i].value() != other[i].value():
                    return False
            else:
                constrained[
                    False,
                    String(
                        "Element ",
                        i,
                        " of Coord must both be a tuple or both be",
                        " a value",
                    ),
                ]()

        return True

    @always_inline("nodebug")
    fn __ne__[
        *other_types: CoordLike
    ](self, other: Coord[*other_types]) -> Bool:
        return not self == other

    @always_inline("nodebug")
    fn tuple(var self) -> Coord[*Self.VariadicType]:
        return rebind[Coord[*Self.VariadicType]](self)

    @always_inline("nodebug")
    fn reverse(var self) -> Coord[*Variadic.reverse[*Self.element_types]]:
        return Coord[*Variadic.reverse[*Self.element_types]](
            rebind[_RegTuple[*Variadic.reverse[*Self.element_types]]](
                self._storage.reverse()
            )
        )

    @always_inline("nodebug")
    fn concat[
        *other_element_types: CoordLike
    ](var self, var other: Coord[*other_element_types]) -> Coord[
        *Variadic.concat_types[Self.element_types, other_element_types]
    ]:
        return Coord[
            *Variadic.concat_types[Self.element_types, other_element_types]
        ](
            rebind[
                _RegTuple[
                    *Variadic.concat_types[
                        Self.element_types, other_element_types
                    ]
                ]
            ](self._storage.concat(other._storage))
        )

    @always_inline("nodebug")
    fn flatten(var self) -> Coord[*_Flattened[*Self.element_types]]:
        """Convert a nested Coord to a flattened Coord.


        Returns:
            A flattened Coord containing all leaf values in order.

        Examples:
            ```mojo
            from layout._coord import Coord, Idx
            var nested = Coord(
                Idx[5](),
                Coord(Idx[3](), Idx[2]()),
                Idx(7)
            )
            var flat = nested.flatten()
            # flat is Coord(Idx[5](), Idx[3](), Idx[2](), Idx(7))
            ```
        """
        comptime FlatTypes = _Flattened[*Self.element_types]
        comptime flat_size = Variadic.size(FlatTypes)

        var flat_tuple: _RegTuple[*FlatTypes]

        # Mark the tuple as initialized so we can work on it
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(flat_tuple)
        )

        # Use _get_flattened to access each element by flat index
        comptime for i in range(flat_size):
            comptime FlatType = FlatTypes[i]

            comptime if FlatType.is_static_value:
                # Compile-time known value
                UnsafePointer(to=flat_tuple[i]).init_pointee_copy(
                    rebind[FlatType](ComptimeInt[FlatType.static_value]())
                )
            else:
                # Runtime value - use _get_flattened to get the value
                var val = _get_flattened[i](self)
                UnsafePointer(to=flat_tuple[i]).init_pointee_copy(
                    rebind[FlatType](
                        RuntimeInt[FlatType.DTYPE](Scalar[FlatType.DTYPE](val))
                    )
                )

        return Coord(flat_tuple)

    @always_inline("nodebug")
    fn make_dynamic[
        dtype: DType
    ](self) -> Coord[*_CoordToDynamic[dtype, *Self.element_types]]:
        """Convert all elements to RuntimeInt[dtype].

        Parameters:
            dtype: The data type for the resulting RuntimeInt values.

        Returns:
            A new Coord where all elements are converted to RuntimeInt[dtype].

        Examples:
            ```mojo
            from layout._coord import Coord, ComptimeInt, RuntimeInt
            var c = Coord(ComptimeInt[3](), RuntimeInt[DType.int32](5), ComptimeInt[7]())
            var dynamic = c.make_dynamic[DType.int64]()
            # dynamic is Coord(RuntimeInt[DType.int64](3), RuntimeInt[DType.int64](5), RuntimeInt[DType.int64](7))
            ```
        """
        comptime ResultTypes = _CoordToDynamic[dtype, *Self.element_types]
        var result: Coord[*ResultTypes]
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(Self.__len__()):
            # Convert all elements to RuntimeInt[dtype]
            UnsafePointer(to=result[i]).init_pointee_copy(
                rebind[ResultTypes[i]](
                    RuntimeInt[dtype](Scalar[dtype](self[i].value()))
                )
            )

        return result

    fn write_to(self, mut w: Some[Writer]):
        w.write("(")

        comptime for i in range(Self.rank):
            comptime if Self.element_types[i].is_tuple:
                self[i].tuple().write_to(w)
            else:
                w.write(self[i].value())

            comptime if i < Self.rank - 1:
                w.write(", ")
        w.write(")")


# Helper for flat indexing with nested shape/stride.
fn _crd2idx_flat[
    out_type: DType,
](crd_t: Coord, shape_t: Coord, stride_t: Coord) -> Scalar[out_type]:
    """Compute index from flat coordinate with nested shape/stride.

    For nested layouts like blocked_product, this computes the linear index
    by flattening the stride and performing element-wise dot product.

    Parameters:
        out_type: Output scalar type.

    Args:
        crd_t: Flat coordinate tuple.
        shape_t: Nested shape tuple.
        stride_t: Nested stride tuple.

    Returns:
        Linear index computed from flat coords and nested shape/stride.
    """
    # Flatten the stride and compute dot product with flat coord
    var flat_stride = stride_t.flatten()
    var result: Scalar[out_type] = 0
    comptime flat_len = type_of(crd_t).__len__()

    comptime for i in range(flat_len):
        result += Scalar[out_type](crd_t[i].value() * flat_stride[i].value())

    return result


# Implementation based off runtime_tuple.mojo's crd2idx.
fn crd2idx[
    Index: CoordLike,
    Shape: CoordLike,
    Stride: CoordLike,
    out_type: DType = DType.int64,
](crd: Index, shape: Shape, stride: Stride) -> Scalar[out_type]:
    """Calculate the index from a coordinate tuple."""
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()
    comptime crd_len = Index.__len__()

    comptime if Shape.is_tuple and Stride.is_tuple and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        var result: Scalar[out_type] = 0

        comptime if crd_len > 1:  # tuple tuple tuple
            var crd_t = crd.tuple()

            # Check if crd structure matches shape structure
            comptime if crd_len == shape_len:
                # Hierarchical indexing: crd elements map 1:1 to shape elements
                comptime for i in range(shape_len):
                    result += crd2idx[out_type=out_type](
                        crd_t[i], shape_t[i], stride_t[i]
                    )
            else:
                # Flat indexing: crd is flat, need to compute with flattened strides
                # Use _crd2idx_flat which handles flat coords with nested shape/stride
                return _crd2idx_flat[out_type](crd_t, shape_t, stride_t)

            return result
        else:  # "int" tuple tuple
            var crd_int: Int

            comptime if Index.is_tuple:
                crd_int = 0 if crd_len == 0 else crd.tuple()[0].value()
            else:
                crd_int = 0 if crd_len == 0 else crd.value()

            comptime last_elem_idx = shape_len - 1

            comptime for i in range(last_elem_idx):
                var quotient, remainder = divmod(crd_int, shape_t[i].product())
                result += crd2idx[out_type=out_type](
                    Idx(remainder), shape_t[i], stride_t[i]
                )
                crd_int = quotient
            return result + crd2idx[out_type=out_type](
                Idx(crd_int), shape_t[last_elem_idx], stride_t[last_elem_idx]
            )
    else:
        comptime if crd_len > 1:
            abort("crd is a tuple but shape and stride are not")
        else:
            return Scalar[out_type](crd.value() * stride.value())


# Implementation based off crd2idx - computes the inverse operation.
# Uses the per-element formula: coord[i] = (idx // stride[i]) % shape[i]
fn idx2crd[
    Shape: CoordLike,
    Stride: CoordLike,
    out_dtype: DType = DType.int64,
](idx: Int, shape: Shape, stride: Stride) -> Coord[
    *_Idx2CrdResultTypes[
        out_dtype,
        RuntimeInt[out_dtype],
        Stride.VariadicType,
        Shape.VariadicType,
    ]
]:
    """Calculate the coordinate tuple from a linear index.

    This is the inverse of crd2idx - given a linear index, shape, and stride,
    it computes the multi-dimensional coordinates using the per-element formula:
    ``coord[i] = (idx // stride[i]) % shape[i]``.

    When a shape dimension is statically known to be 1, the corresponding
    output coordinate is a ComptimeInt[0]. Otherwise, coordinates are
    RuntimeInt[out_dtype].

    Parameters:
        Shape: The shape type (must be CoordLike).
        Stride: The stride type (must be CoordLike).
        out_dtype: The output data type for coordinate values.

    Args:
        idx: The linear index to convert.
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        A Coord containing the coordinate values for each dimension.
        Dimensions with static shape 1 produce ComptimeInt[0], others
        produce RuntimeInt[out_dtype].

    Examples:
        For a 2D tensor with shape (3, 4) and row-major strides (4, 1):

        - idx2crd(0, shape, stride) returns (0, 0).
        - idx2crd(5, shape, stride) returns (1, 1).
        - idx2crd(11, shape, stride) returns (2, 3).
    """
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()

    debug_assert(
        shape_len == stride_len,
        "Shape length (",
        shape_len,
        ") must match stride length (",
        stride_len,
        ")",
    )

    comptime ResultTypes = _Idx2CrdResultTypes[
        out_dtype,
        RuntimeInt[out_dtype],
        Stride.VariadicType,
        Shape.VariadicType,
    ]
    comptime Result = Coord[*ResultTypes]
    var result = Result()

    comptime if Shape.is_tuple and Stride.is_tuple and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        comptime for i in range(shape_len):
            comptime if (
                Shape.VariadicType[i].is_static_value
                and Shape.VariadicType[i].static_value == 1
            ):
                UnsafePointer(to=result[i]).init_pointee_copy(
                    rebind[ResultTypes[i]](ComptimeInt[0]())
                )
            else:
                var stride_val = stride_t[i].value()
                var shape_val = shape_t[i].value()
                var coord_val = (idx // stride_val) % shape_val
                UnsafePointer(to=result[i]).init_pointee_copy(
                    rebind[ResultTypes[i]](
                        RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                    )
                )
    else:
        comptime if Shape.is_static_value and Shape.static_value == 1:
            UnsafePointer(to=result[0]).init_pointee_copy(
                rebind[ResultTypes[0]](ComptimeInt[0]())
            )
        else:
            var coord_val = (idx // stride.value()) % shape.value()

            comptime for i in range(shape_len):
                UnsafePointer(to=result[i]).init_pointee_copy(
                    rebind[ResultTypes[i]](
                        RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                    )
                )

    return result


fn idx2crd[
    Index: CoordLike,
    Shape: CoordLike,
    Stride: CoordLike,
    out_dtype: DType = DType.int64,
](idx: Index, shape: Shape, stride: Stride) -> Coord[
    *_Idx2CrdResultTypes[
        out_dtype, Index, Stride.VariadicType, Shape.VariadicType
    ]
]:
    """Calculate the coordinate tuple from a CoordLike linear index.

    This overload accepts a CoordLike index, enabling compile-time result
    computation when the index, shape, and stride are all statically known.
    Uses the per-element formula: ``coord[i] = (idx // stride[i]) % shape[i]``.

    Parameters:
        Index: The index type (must be CoordLike).
        Shape: The shape type (must be CoordLike).
        Stride: The stride type (must be CoordLike).
        out_dtype: The output data type for coordinate values.

    Args:
        idx: The CoordLike linear index to convert.
        shape: The shape of the tensor.
        stride: The stride of the tensor.

    Returns:
        A Coord containing the coordinate values for each dimension.
        When idx, shape, and stride are all compile-time known, produces
        ComptimeInt results. Otherwise produces RuntimeInt[out_dtype].
    """
    comptime shape_len = Shape.__len__()
    comptime stride_len = Stride.__len__()

    debug_assert(
        shape_len == stride_len,
        "Shape length (",
        shape_len,
        ") must match stride length (",
        stride_len,
        ")",
    )

    comptime ResultTypes = _Idx2CrdResultTypes[
        out_dtype, Index, Stride.VariadicType, Shape.VariadicType
    ]
    comptime Result = Coord[*ResultTypes]
    var result = Result()

    comptime if Shape.is_tuple and Stride.is_tuple and shape_len == stride_len:
        var shape_t = shape.tuple()
        var stride_t = stride.tuple()

        comptime for i in range(shape_len):
            comptime if (
                Shape.VariadicType[i].is_static_value
                and Shape.VariadicType[i].static_value == 1
            ):
                UnsafePointer(to=result[i]).init_pointee_copy(
                    rebind[ResultTypes[i]](ComptimeInt[0]())
                )
            elif (
                Index.is_static_value
                and Shape.VariadicType[i].is_static_value
                and Stride.VariadicType[i].is_static_value
            ):
                # All static: result is ComptimeInt, already default-initialized.
                pass
            else:
                var stride_val = stride_t[i].value()
                var shape_val = shape_t[i].value()
                var coord_val = (idx.value() // stride_val) % shape_val
                UnsafePointer(to=result[i]).init_pointee_copy(
                    rebind[ResultTypes[i]](
                        RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                    )
                )
    else:
        comptime if Shape.is_static_value and Shape.static_value == 1:
            UnsafePointer(to=result[0]).init_pointee_copy(
                rebind[ResultTypes[0]](ComptimeInt[0]())
            )
        elif (
            Index.is_static_value
            and Shape.is_static_value
            and Stride.is_static_value
        ):
            # All static: result is ComptimeInt, already default-initialized.
            pass
        else:
            var coord_val = (idx.value() // stride.value()) % shape.value()

            comptime for i in range(shape_len):
                UnsafePointer(to=result[i]).init_pointee_copy(
                    rebind[ResultTypes[i]](
                        RuntimeInt[out_dtype](Scalar[out_dtype](coord_val))
                    )
                )

    return result


fn coord_to_int_tuple[
    *element_types: CoordLike
](value: Coord[*element_types]) -> IntTuple:
    """Convert a Coord to an IntTuple, preserving the nested structure.

    This function recursively traverses the Coord and converts each element:
    - Value elements (ComptimeInt, RuntimeInt) become integer values in the IntTuple
    - Tuple elements (nested Coord) become nested IntTuples

    Parameters:
        element_types: The variadic pack of element types in the Coord.

    Args:
        value: The Coord to convert.

    Returns:
        An IntTuple with the same structure and values as the input Coord.
    """
    var result = IntTuple()

    comptime for i in range(Coord[*element_types].__len__()):
        comptime T = element_types[i]

        comptime if T.is_tuple:
            # Recursively convert nested tuples
            result.append(coord_to_int_tuple(value[i].tuple()))
        else:
            # Convert value elements to integers
            result.append(IntTuple(value[i].value()))

    return result


@always_inline
fn coord_to_index_list[
    *element_types: CoordLike
](value: Coord[*element_types]) -> std.utils.IndexList[value.rank]:
    """Convert a flat Coord to an IndexList.

    Parameters:
        element_types: The variadic pack of element types in the Coord.

    Args:
        value: The Coord to convert.

    Returns:
        An IndexList with the same rank and values as the input Coord.
    """
    var result = std.utils.IndexList[value.rank]()

    comptime for i in range(Coord[*element_types].__len__()):
        result[i] = value[i].value()

    return result


fn coord_to_int_tuple[*element_types: CoordLike]() -> IntTuple:
    """Convert a Coord to an IntTuple, preserving the nested structure.

    This function recursively traverses the Coord and converts each element:
    - Value elements (ComptimeInt, RuntimeInt) become integer values in the IntTuple
    - Tuple elements (nested Coord) become nested IntTuples

    Parameters:
        element_types: The variadic pack of element types in the Coord.

    Returns:
        An IntTuple with the same structure and values as the input Coord.
    """
    var result = IntTuple()

    comptime for i in range(Variadic.size(element_types)):
        comptime T = element_types[i]

        comptime if T.is_tuple:
            # Recursively convert nested tuples
            result.append(coord_to_int_tuple[element_types[i]]())
        else:
            comptime if T.is_static_value:
                result.append(IntTuple(T.static_value))
            else:
                result.append(layout.UNKNOWN_VALUE)

    return result


fn coord[
    dtype: DType, *element_types: Movable
](var values: Tuple[*element_types]) -> Coord[
    *_Splatted[RuntimeInt[dtype], type_of(values).__len__()]
] where _AllEqual[Int, *element_types]:
    """Helper to create a Coord from a variadic pack of integers.
    Parameters:
        dtype: The data type for the runtime integer values.
        rank: The number of elements in the tuple.
    Args:
        values: The run-time integer values.
    Returns:
        A `Coord` instance containing `ComptimeInt` elements for each value.
    Usage: coord[5, 3, 2]() creates Coord(ComptimeInt[5](), ComptimeInt[3](), ComptimeInt[2]()).
    """
    var tuple = Coord[
        *_Splatted[RuntimeInt[dtype], type_of(values).__len__()]
    ]()

    comptime for i in range(type_of(values).__len__()):
        UnsafePointer(to=tuple[i]).init_pointee_copy(
            rebind[type_of(tuple[i])](
                RuntimeInt[dtype](Scalar[dtype](rebind[Int](values[i])))
            )
        )
    return tuple


fn coord[*values: Int]() -> Coord[*_IntToComptimeInt[*values]]:
    """Helper to create a Coord from a variadic pack of integers.
    Parameters:
        values: The compile-time integer values.
    Returns:
        A `Coord` instance containing `ComptimeInt` elements for each value.
    Usage: coord[5, 3, 2]() creates Coord(ComptimeInt[5](), ComptimeInt[3](), ComptimeInt[2]()).
    """
    # values is a ZST since all elements are comptime
    var tuple = Coord[*_IntToComptimeInt[*values]]()
    return tuple


comptime DynamicCoord[dtype: DType, size: Int] = Coord[
    *_Splatted[RuntimeInt[dtype], size]
]
"""
Create a Coord full of `size` dynamic elements with `dtype`.

Parameters:
    dtype: The output element DType.
    size: The number of output elements.

Returns:
    A Coord full of `size` dynamic elements with `dtype`.
"""

comptime StaticCoord[value: Int, size: Int] = Coord[
    *_Splatted[ComptimeInt[value], size]
]
"""
Create a Coord full of `size` static elements with `dtype`.

Parameters:
    value: The value of each element.
    size: The number of output elements.

Returns:
    A Coord full of `size` static elements with `dtype`.
"""

comptime _FlattenReducer[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_types[
    Prev,
    From[idx]
    .VariadicType if From[idx]
    .is_tuple else Variadic.types[T=CoordLike, From[idx]],
]


comptime _Flattened[
    *element_types: CoordLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=element_types,
    Reducer=_FlattenReducer,
]

comptime _NextOffset[
    prev_offset: Int,
    element_type: CoordLike,
] = prev_offset + (
    1 if element_type.is_value else Variadic.size(
        _Flattened[*element_type.VariadicType]
    )
)


comptime _FlattenOffsetReducer[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_types[
    Prev,
    Variadic.types[
        T=CoordLike,
        ComptimeInt[
            0 if idx
            == 0 else _NextOffset[
                Prev[Variadic.size(Prev) - 1].static_value,
                From[idx - 1],
            ]
        ],
    ],
]


comptime _FlattenedOffsets[
    *element_types: CoordLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=element_types,
    Reducer=_FlattenOffsetReducer,
]


fn _get_flattened_helper[
    flat_idx: Int,
    current_offset: Int,
    i: Int,
    *element_types: CoordLike,
](tuple: Coord[*element_types]) -> Int:
    """Helper function to recursively access flattened elements."""

    comptime if i >= Coord[*element_types].__len__():
        comptime assert False, "flat_idx out of bounds"
        abort()

    comptime T = element_types[i]

    comptime if T.is_tuple:
        comptime count = Variadic.size(_Flattened[*T.VariadicType])

        comptime if flat_idx >= current_offset and flat_idx < current_offset + count:
            return _get_flattened[flat_idx - current_offset](tuple[i].tuple())
        else:
            return _get_flattened_helper[
                flat_idx, current_offset + count, i + 1
            ](tuple)
    else:
        comptime if flat_idx == current_offset:
            return tuple[i].value()
        else:
            return _get_flattened_helper[flat_idx, current_offset + 1, i + 1](
                tuple
            )


fn _get_flattened[
    flat_idx: Int, *element_types: CoordLike
](tuple: Coord[*element_types]) -> Int:
    """Access an element from a nested Coord using a flat index.

    Parameters:
        flat_idx: The index into the flattened representation.
        element_types: The variadic element types of the tuple.

    Args:
        tuple: The nested Coord to access.

    Returns:
        The value at the given flat index.

    Examples:
        For tuple = Coord(Idx[5](), Coord(Idx[3](), Idx[2]()), Idx(7)):
        - get_flattened[0](tuple) returns 5  (first element)
        - get_flattened[1](tuple) returns 3  (first element of nested tuple)
        - get_flattened[2](tuple) returns 2  (second element of nested tuple)
        - get_flattened[3](tuple) returns 7  (third top-level element)
    """
    return _get_flattened_helper[flat_idx, 0, 0](tuple)


comptime _AllStaticReducer[
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = (Variadic.values[From[idx].is_static_value and Prev[0]])


comptime _AllStatic[*element_types: CoordLike] = _ReduceVariadicAndIdxToValue[
    BaseVal = Variadic.values[True],
    VariadicType=element_types,
    Reducer=_AllStaticReducer,
][0]

comptime _AllEqualReducer[
    T: AnyType,
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[AnyType],
    idx: Int,
] = (
    Variadic.values[
        _type_is_eq_parse_time[From[idx], T]() and (Prev[0] or idx == 0)
    ]
)


comptime _AllEqual[
    T: AnyType, *element_types: AnyType
] = _ReduceVariadicAndIdxToValue[
    BaseVal = Variadic.values[False],
    VariadicType=element_types,
    Reducer = _AllEqualReducer[T],
][
    0
]

comptime _StaticProductReducer[
    Prev: Variadic.ValuesOfType[Int],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = (Variadic.values[From[idx].static_value * Prev[0]])


comptime _StaticProduct[
    *element_types: CoordLike
] = _ReduceVariadicAndIdxToValue[
    BaseVal = Variadic.values[1],
    VariadicType=element_types,
    Reducer=_StaticProductReducer,
][
    0
]

comptime _IntToComptimeIntMapper[
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.ValuesOfType[Int],
    idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[ComptimeInt[From[idx]]]]


comptime _IntToComptimeInt[*values: Int] = _ReduceValueAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=values,
    Reducer=_IntToComptimeIntMapper,
]

comptime _Splatted[T: CoordLike, count: Int] = __mlir_attr[
    `#kgen.variadic.splat<`,
    T,
    `,`,
    count._mlir_value,
    `> : `,
    Variadic.TypesOfTrait[type_of(T)],
]


# ===-----------------------------------------------------------------------===#
# Dim to CoordLike conversion
# ===-----------------------------------------------------------------------===#


comptime _DimToCoordLikeMapper[
    dtype: DType,
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.ValuesOfType[Dim],
    idx: Int,
] = Variadic.concat_types[
    Prev,
    Variadic.types[
        T=CoordLike, ComptimeInt[From[idx]._value_or_missing]
    ] if From[idx]._value_or_missing
    != Dim._sentinel else Variadic.types[T=CoordLike, RuntimeInt[dtype]],
]
"""Maps a single Dim value to a CoordLike type.

If the Dim has a static value, produces ComptimeInt[value].
If the Dim is dynamic, produces RuntimeInt.

Uses direct field access rather than methods for compile-time evaluation.
"""


comptime _DimsToCoordLike[
    dtype: DType, dims: DimList
] = _ReduceValueAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType = dims.value.value,
    Reducer = _DimToCoordLikeMapper[dtype],
]
"""Converts a variadic of Dim values to a variadic of CoordLike types.

Note:
    This transformation is a value-to-type mapper that is meant to be
    used in the parameter domain,.

For each Dim in the input:
- If the dim has a static value, produces `ComptimeInt[value]`
- If the dim is dynamic, produces `RuntimeInt`

Example:
    ```mojo
    from buffer import Dim, DimList
    from layout._coord import _DimsToCoordLike, Coord

    # Static dims become ComptimeInt, dynamic dims become RuntimeInt
    comptime dims = DimList(Dim(3), Dim(), Dim(5))
    comptime coord_types = _DimsToCoordLike[DType.int32, dims]
    # dims is equivalent to Variadic.types[ComptimeInt[3], RuntimeInt, ComptimeInt[5]]

    # Can be used to create a Coord type
    comptime my_coords = Coord[*coord_types]
    ```
"""

comptime _IntTupleToCoordLikeMapper[
    dtype: DType,
    tuple: IntTuple,
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_types[
    Prev,
    Variadic.types[T=CoordLike, ComptimeInt[Int(tuple[idx])]] if Int(tuple[idx])
    != layout.UNKNOWN_VALUE else Variadic.types[T=CoordLike, RuntimeInt[dtype]],
]
"""Maps a single Dim value to a CoordLike type.

If the Dim has a static value, produces ComptimeInt[value].
If the Dim is dynamic, produces RuntimeInt.

Uses direct field access rather than methods for compile-time evaluation.
"""

comptime _IntTupleToCoordLike[
    dtype: DType, tuple: IntTuple
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType = Variadic.types[
        T=CoordLike, *_Splatted[RuntimeInt[dtype], len(tuple)]
    ],
    Reducer = _IntTupleToCoordLikeMapper[dtype, tuple],
]
"""Converts a variadic of Dim values to a variadic of CoordLike types.

Note:
    This transformation is a value-to-type mapper that is meant to be
    used in the parameter domain,.

For each Dim in the input:
- If the dim has a static value, produces `ComptimeInt[value]`
- If the dim is dynamic, produces `RuntimeInt`

Example:
    ```mojo
    from buffer import Dim, DimList
    from layout._coord import _DimsToCoordLike, Coord

    # Static dims become ComptimeInt, dynamic dims become RuntimeInt
    comptime dims = DimList(Dim(3), Dim(), Dim(5))
    comptime coord_types = _DimsToCoordLike[DType.int32, dims]
    # dims is equivalent to Variadic.types[ComptimeInt[3], RuntimeInt, ComptimeInt[5]]

    # Can be used to create a Coord type
    comptime my_coords = Coord[*coord_types]
    ```
"""


comptime _CoordToDimMapper[
    Prev: Variadic.ValuesOfType[Dim],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_values[
    Prev,
    Variadic.values[
        Dim(From[idx].static_value) if From[idx].is_static_value else Dim(),
    ],
]
"""Maps a Coord to a DimList.

Uses direct field access rather than methods for compile-time evaluation.
"""


comptime _CoordToDimList[*dims: CoordLike] = DimList(
    VariadicList(
        _ReduceVariadicAndIdxToValue[
            BaseVal = Variadic.empty_of_type[Dim],
            VariadicType=dims,
            Reducer=_CoordToDimMapper,
        ]
    )
)
"""Converts a variadic of Dim values to a variadic of CoordLike types.

Note:
    This transformation is a value-to-type mapper that is meant to be
    used in the parameter domain,.

For each Dim in the input:
- If the dim has a static value, produces `ComptimeInt[value]`
- If the dim is dynamic, produces `RuntimeInt`

Example:
    ```mojo
    from buffer import Dim, DimList
    from layout._coord import _CoordToDimList, Coord, Idx

    # Static dims become ComptimeInt, dynamic dims become RuntimeInt
    var coords = Coord(Idx(3), Idx[5]())
    comptime dimlist = _CoordToDimList[*coords.element_types]
    # dims is equivalent to DimList(Dim(), 5)
    ```
"""

# ===-----------------------------------------------------------------------===#
# CoordLike to Dynamic conversion
# ===-----------------------------------------------------------------------===#


comptime _CoordToDynamicMapper[
    dtype: DType,
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = Variadic.concat_types[
    Prev,
    Variadic.types[T=CoordLike, RuntimeInt[dtype]],
]
"""Maps a single CoordLike element to RuntimeInt[dtype].
All elements (ComptimeInt, RuntimeInt of any dtype) are converted to RuntimeInt[dtype].
"""


comptime _CoordToDynamic[
    dtype: DType, *element_types: CoordLike
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=element_types,
    Reducer = _CoordToDynamicMapper[dtype],
]
"""Converts a variadic of CoordLike types to all RuntimeInt[dtype].
All elements are converted to RuntimeInt[dtype], regardless of their original type.

Example:

    ```mojo
    from layout._coord import _CoordToDynamic, ComptimeInt, RuntimeInt, Coord
    # All elements become RuntimeInt[DType.int64]
    comptime types = _CoordToDynamic[DType.int64, ComptimeInt[3], RuntimeInt[DType.int32], ComptimeInt[5]]
    # types is equivalent to Variadic.types[RuntimeInt[DType.int64], RuntimeInt[DType.int64], RuntimeInt[DType.int64]]
    ```
"""


# ===-----------------------------------------------------------------------===#
# idx2crd result type computation
# ===-----------------------------------------------------------------------===#


comptime _Idx2CrdResultMapper[
    out_dtype: DType,
    idx_type: CoordLike,
    stride_types: Variadic.TypesOfTrait[CoordLike],
    Prev: Variadic.TypesOfTrait[CoordLike],
    From: Variadic.TypesOfTrait[CoordLike],
    i: Int,
] = Variadic.concat_types[
    Prev,
    # shape == 1: always ComptimeInt[0]
    Variadic.types[T=CoordLike, ComptimeInt[0]] if From[i].is_static_value
    and From[i].static_value == 1
    # all three (idx, shape, stride) static: compute at compile time
    else Variadic.types[
        T=CoordLike,
        ComptimeInt[
            (idx_type.static_value // stride_types[i].static_value)
            % From[i].static_value
        ],
    ] if idx_type.is_static_value
    and From[i].is_static_value
    and stride_types[i].is_static_value
    # otherwise: runtime
    else Variadic.types[T=CoordLike, RuntimeInt[out_dtype]],
]
"""Maps a shape element type to an idx2crd result type.

Considers shape, stride, and index to determine the result type:
- If shape is statically 1, produces ComptimeInt[0].
- If all of idx, shape, and stride are statically known, produces
  ComptimeInt[(idx // stride) % shape].
- Otherwise produces RuntimeInt[out_dtype].
"""


comptime _Idx2CrdResultTypes[
    out_dtype: DType,
    idx_type: CoordLike,
    stride_types: Variadic.TypesOfTrait[CoordLike],
    shape_types: Variadic.TypesOfTrait[CoordLike],
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal = Variadic.empty_of_trait[CoordLike],
    VariadicType=shape_types,
    Reducer = _Idx2CrdResultMapper[out_dtype, idx_type, stride_types],
]
"""Computes the result types for idx2crd based on shape, stride, and index.

For each dimension:
- If shape is statically 1, the result is ComptimeInt[0]
- If idx, shape, and stride are all statically known, the result is
  ComptimeInt[(idx // stride) % shape]
- Otherwise, the result is RuntimeInt[out_dtype]

Example:
    ```mojo
    from layout._coord import _Idx2CrdResultTypes, ComptimeInt, RuntimeInt
    comptime stride_t = Variadic.types[T=CoordLike, ComptimeInt[4], ComptimeInt[4], ComptimeInt[1]]
    comptime shape_t = Variadic.types[T=CoordLike, ComptimeInt[3], ComptimeInt[1], ComptimeInt[4]]
    comptime types = _Idx2CrdResultTypes[DType.int64, RuntimeInt[DType.int64], stride_t, shape_t]
    ```
"""


struct _RegTuple[*element_types: TrivialRegisterPassable](
    ImplicitlyCopyable, Sized, TrivialRegisterPassable
):
    """
    A temporary internal type to represent a Tuple where
    all elements are register passable. This should
    be removed once we have conditional conformance.
    """

    comptime _mlir_type = __mlir_type[
        `!kgen.pack<:`,
        Variadic.TypesOfTrait[TrivialRegisterPassable],
        Self.element_types,
        `>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying storage for the tuple."""

    # Overload that crushes down IR generated on the caller side.
    @always_inline("nodebug")
    fn __init__(out self: _RegTuple[]):
        """Construct an empty tuple."""
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

    @always_inline("nodebug")
    fn __init__(out self, var *args: * Self.element_types):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(
        out self,
        *,
        var storage: VariadicPack[
            _, TrivialRegisterPassable, *Self.element_types
        ],
    ):
        """Construct the tuple from a low-level internal representation.

        Args:
            storage: The variadic pack storage to construct from.
        """

        # Mark 'self._mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        # Move each element into the tuple storage.
        @parameter
        fn init_elt[idx: Int](var elt: Self.element_types[idx]):
            UnsafePointer(to=self[idx]).init_pointee_move(elt)

        storage^.consume_elements[init_elt]()

    @always_inline("builtin")
    @staticmethod
    fn __len__() -> Int:
        """Return the number of elements in the tuple.

        Returns:
            The tuple length.
        """

        comptime result = Variadic.size(Self.element_types)
        return result

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return Self.__len__()

    @always_inline("nodebug")
    fn __getitem__[idx: Int](ref self) -> ref[self] Self.element_types[idx]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element to return.

        Returns:
            A reference to the specified element.
        """
        # Return a reference to an element at the specified index, propagating
        # mutability of self.
        var storage_kgen_ptr = UnsafePointer(to=self._mlir_value).address

        # KGenPointer to the element.
        var elt_kgen_ptr = __mlir_op.`kgen.pack.gep`[
            index = idx.__mlir_index__()
        ](storage_kgen_ptr)
        return UnsafePointer[_, origin_of(self)](elt_kgen_ptr)[]

    @always_inline("nodebug")
    fn __init__[
        *elt_types: TrivialRegisterPassable & Defaultable
    ](out self: _RegTuple[*elt_types]):
        """Construct a tuple with default-initialized elements.

        Parameters:
            elt_types: The types of the elements contained in the Tuple.
        """

        # Mark 'self._mlir_value' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )

        comptime for i in range(type_of(self).__len__()):
            UnsafePointer(to=self[i]).init_pointee_move(elt_types[i]())

    @always_inline
    fn __eq__[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Equatable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Equatable
        ],
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Bool:
        """Compare this tuple to another tuple using equality comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the Tuple.
            other_elt_types: The types of the elements contained in the other Tuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is equal to the other tuple, False otherwise.
        """

        # We do not use self._compare here because we only want
        # Equatable conformance for the method.
        comptime self_len = type_of(self).__len__()
        comptime other_len = type_of(other).__len__()

        comptime if self_len != other_len:
            return False

        comptime for i in range(type_of(self).__len__()):
            comptime self_type = type_of(self[i])
            comptime other_type = type_of(other[i])
            comptime assert _type_is_eq[
                self_type, other_type
            ](), "Tuple elements must be of the same type to compare."
            if self[i] != rebind[self_type](other[i]):
                return False
        return True

    @always_inline
    fn __ne__[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Equatable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Equatable
        ],
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Bool:
        """Compare this tuple to another tuple using inequality comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the _RegTuple.
            other_elt_types: The types of the elements contained in the other _RegTuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is not equal to the other tuple, False otherwise.
        """

        return not self == other

    @always_inline
    fn _compare[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Int:
        comptime self_len = type_of(self).__len__()
        comptime other_len = type_of(other).__len__()

        comptime if other_len == 0:
            return 1 if self_len > 0 else 0

        comptime min_length = min(self_len, other_len)

        comptime for i in range(min_length):
            comptime self_type = type_of(self[i])
            comptime other_type = type_of(other[i])
            comptime assert _type_is_eq[self_type, other_type](), String(
                "Mismatch between tuple elements at index ",
                i,
                " must be of the same type to compare.",
            )
            if self[i] < rebind[self_type](other[i]):
                return -1
            if rebind[self_type](other[i]) < self[i]:
                return 1

        comptime if self_len < other_len:
            return -1
        elif self_len > other_len:
            return 1
        else:
            return 0

    @always_inline
    fn __lt__[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        //,
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Bool:
        """Compare this tuple to another tuple using less than comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the _RegTuple.
            other_elt_types: The types of the elements contained in the other _RegTuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is less than the other tuple, False otherwise.
        """
        return self._compare(other) < 0

    @always_inline
    fn __le__[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        //,
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Bool:
        """Compare this tuple to another tuple using less than or equal to comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the _RegTuple.
            other_elt_types: The types of the elements contained in the other _RegTuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is less than or equal to the other tuple, False otherwise.
        """
        return self._compare(other) <= 0

    @always_inline
    fn __gt__[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        //,
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Bool:
        """Compare this tuple to another tuple using greater than comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the _RegTuple.
            other_elt_types: The types of the elements contained in the other
                _RegTuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is greater than the other tuple, False otherwise.
        """

        return self._compare(other) > 0

    @always_inline
    fn __ge__[
        self_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        other_elt_types: Variadic.TypesOfTrait[
            TrivialRegisterPassable & Comparable
        ],
        //,
    ](
        self: _RegTuple[*self_elt_types], other: _RegTuple[*other_elt_types]
    ) -> Bool:
        """Compare this tuple to another tuple using greater than or equal to comparison.

        Parameters:
            self_elt_types: The types of the elements contained in the _RegTuple.
            other_elt_types: The types of the elements contained in the other _RegTuple.

        Args:
            other: The other tuple to compare against.

        Returns:
            True if this tuple is greater than or equal to the other tuple, False otherwise.
        """

        return self._compare(other) >= 0

    @always_inline("nodebug")
    fn reverse(
        self,
        out result: _RegTuple[*Variadic.reverse[*Self.element_types]],
    ):
        """Return a new tuple with the elements in reverse order.

        Returns:
            A new tuple with the elements in reverse order.

        Usage:

        ```mojo
        image_coords = _RegTuple[Int, Int](100, 200) # row-major indexing
        screen_coords = image_coords.reverse() # (col, row) for x,y display
        print(screen_coords[0], screen_coords[1]) # output: 200, 100
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime for i in range(type_of(result).__len__()):
            UnsafePointer(to=result[i]).init_pointee_copy(
                rebind[type_of(result[i])](
                    self[Variadic.size(Self.element_types) - 1 - i]
                )
            )

    @always_inline("nodebug")
    fn concat[
        *other_element_types: TrivialRegisterPassable
    ](
        self,
        other: _RegTuple[*other_element_types],
        out result: _RegTuple[
            *Variadic.concat_types[Self.element_types, other_element_types]
        ],
    ):
        """Return a new tuple that concatenates this tuple with another.

        Args:
            other: The other tuple to concatenate.

        Parameters:
            other_element_types: The types of the elements contained in the other _RegTuple.

        Returns:
            A new tuple with the concatenated elements.

        Usage:

        ```
        var rgb = _RegTuple[Int, Int, Int](0xFF, 0xF0, 0x0)
        var rgba = rgb.concat(_RegTuple[Int](0xFF)) # Adds alpha channel
        print(rgba[0], rgba[1], rgba[2], rgba[3]) # 255 240 0 255
        ```
        """
        # Mark 'result' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )

        comptime self_len = Self.__len__()

        comptime for i in range(self_len):
            UnsafePointer(to=result[i]).init_pointee_copy(
                rebind[type_of(result[i])](self[i])
            )

        comptime for i in range(type_of(other).__len__()):
            UnsafePointer(to=result[self_len + i]).init_pointee_copy(
                rebind[type_of(result[self_len + i])](other[i])
            )

    @always_inline("nodebug")
    fn __contains__[T: Equatable](self, value: T) -> Bool:
        """Return whether the tuple contains the specified value.

        For example:

        ```mojo
        var t = Tuple(True, 1, 2.5)
        if 1 in t:
            print("t contains 1")
        ```

        Args:
            value: The value to search for.

        Parameters:
            T: The type of the value.

        Returns:
            True if the value is in the tuple, False otherwise.
        """

        comptime for i in range(type_of(self).__len__()):
            comptime if _type_is_eq[Self.element_types[i], T]():
                if rebind[T](self[i]) == value:
                    return True

        return False


comptime _MultiplyMapper[
    Rhs: Variadic.TypesOfTrait[CoordLike],
    element_types: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = ComptimeInt[element_types[idx].static_value * Rhs[idx].static_value]


comptime _Multiply[
    Lhs: Variadic.TypesOfTrait[CoordLike],
    Rhs: Variadic.TypesOfTrait[CoordLike],
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=Lhs,
    Mapper = _MultiplyMapper[Rhs=Rhs],
]


comptime _MultiplyByScalarMapper[
    scalar: Int,
    element_types: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = ComptimeInt[element_types[idx].static_value * scalar]


comptime _MultiplyByScalar[
    Types: Variadic.TypesOfTrait[CoordLike],
    scalar: Int,
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=Types,
    Mapper = _MultiplyByScalarMapper[scalar=scalar],
]
"""Multiply each element in Types by a scalar value.

Parameters:
    Types: The variadic types to multiply.
    scalar: The scalar value to multiply each element by.

Returns:
    A new variadic of ComptimeInt types with multiplied values.
"""


comptime _DivideMapper[
    Rhs: Variadic.TypesOfTrait[CoordLike],
    element_types: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = ComptimeInt[element_types[idx].static_value // Rhs[idx].static_value]


comptime _Divide[
    Lhs: Variadic.TypesOfTrait[CoordLike],
    Rhs: Variadic.TypesOfTrait[CoordLike],
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=Lhs,
    Mapper = _DivideMapper[Rhs=Rhs],
]

comptime _CeilDivMapper[
    Rhs: Variadic.TypesOfTrait[CoordLike],
    element_types: Variadic.TypesOfTrait[CoordLike],
    idx: Int,
] = ComptimeInt[
    (element_types[idx].static_value + Rhs[idx].static_value - 1)
    // Rhs[idx].static_value
]


comptime _CeilDiv[
    Lhs: Variadic.TypesOfTrait[CoordLike],
    Rhs: Variadic.TypesOfTrait[CoordLike],
] = _MapVariadicAndIdxToType[
    To=CoordLike,
    VariadicType=Lhs,
    Mapper = _CeilDivMapper[Rhs=Rhs],
]
