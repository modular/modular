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
"""Implements the VariadicList, VariadicParamList and VariadicPack types.

These are Mojo built-ins, so you don't need to import them.
"""

from std.builtin.constrained import _constrained_conforms_to
from std.builtin.rebind import downcast
from std.format._utils import FormatStruct, TypeNames
from std.sys.intrinsics import _type_is_eq_parse_time
from std.builtin.globals import global_constant


struct Variadic:
    """A namespace for variadic utilities."""

    comptime ValuesOfType[type: AnyType] = __mlir_type[
        `!kgen.variadic<`, type, `>`
    ]
    """Represents a raw variadic sequence of values of the specified type.

    Parameters:
        type: The type of values in the variadic sequence.
    """

    comptime TypesOfTrait[T: type_of(AnyType)] = __mlir_type[
        `!kgen.variadic<`, T, `>`
    ]
    """Represents a raw variadic sequence of types that satisfy the specified trait.

    Parameters:
        T: The trait that types in the variadic sequence must conform to.
    """

    @staticmethod
    @always_inline("builtin")
    def size[T: AnyType](seq: Self.ValuesOfType[T]) -> Int:
        """Returns the length of a variadic sequence.

        Parameters:
            T: The type of values in the sequence.

        Args:
            seq: The variadic sequence to measure.

        Returns:
            The length of the variadic sequence.
        """
        return Int(mlir_value=__mlir_op.`pop.variadic.size`(seq))

    @staticmethod
    @always_inline("builtin")
    def size[T: type_of(AnyType)](seq: Self.TypesOfTrait[T]) -> Int:
        """Returns the length of a variadic sequence.

        Parameters:
            T: The trait that types in the sequence must conform to.

        Args:
            seq: The variadic sequence of types to measure.

        Returns:
            The length of the variadic sequence.
        """
        return Int(mlir_value=__mlir_op.`pop.variadic.size`(seq))

    # ===-----------------------------------------------------------------------===#
    # Utils
    # ===-----------------------------------------------------------------------===#

    comptime empty_of_trait[T: type_of(AnyType)] = __mlir_attr[
        `#kgen.variadic<>: !kgen.variadic<`, T, `>`
    ]
    """Empty comptime variadic of type values.

    Parameters:
        T: The trait that types in the variadic sequence must conform to.
    """

    comptime empty_of_type[T: AnyType] = __mlir_attr[
        `#kgen.variadic<>: !kgen.variadic<`, T, `>`
    ]
    """Empty comptime variadic of values.

    Parameters:
        T: The type of values in the variadic sequence.
    """

    comptime types[T: type_of(AnyType), //, *Ts: T] = Ts
    """Turn discrete type values (bound by `T`) into a single variadic.

    Parameters:
        T: The trait that the types must conform to.
        Ts: The types to collect into a variadic sequence.
    """

    comptime values[T: AnyType, //, *values_: T]: Variadic.ValuesOfType[
        T
    ] = values_
    """Turn discrete values (bound by `T`) into a single variadic.

    Parameters:
        T: The type of the values.
        values_: The values to collect into a variadic sequence.
    """

    # ===-----------------------------------------------------------------------===#
    # VariadicConcat
    # ===-----------------------------------------------------------------------===#

    comptime concat_types[
        T: type_of(AnyType), //, *Ts: Variadic.TypesOfTrait[T]
    ] = __mlir_attr[
        `#kgen.variadic.concat<`, Ts, `> :`, Variadic.TypesOfTrait[T]
    ]
    """Represents the concatenation of multiple variadic sequences of types.

    Parameters:
        T: The trait that types in the variadic sequences must conform to.
        Ts: The variadic sequences to concatenate.
    """

    comptime concat_values[
        T: AnyType, //, *Ts: Variadic.ValuesOfType[T]
    ] = __mlir_attr[
        `#kgen.variadic.concat<`, Ts, `> :`, Variadic.ValuesOfType[T]
    ]
    """Represents the concatenation of multiple variadic sequences of values.

    Parameters:
        T: The types of the values in the variadic sequences.
        Ts: The variadic sequences to concatenate.
    """

    comptime reverse[
        T: type_of(AnyType), //, *element_types: T
    ]: Variadic.TypesOfTrait[T] = _MapVariadicAndIdxToType[
        To=T, VariadicType=element_types, Mapper=_ReversedVariadic[T, ...]
    ]
    """A wrapper to reverse a variadic sequence of types.

    Parameters:
        T: The trait that the types conform to.
        element_types: The variadic sequence of types to reverse.
    """

    comptime splat_type[
        Trait: type_of(AnyType), //, count: Int, type: Trait
    ]: Variadic.TypesOfTrait[Trait] = Self.tabulate_type[
        Trait=Trait, ToT=type, count, _SplatTypeTabulator[Trait, type, _]
    ]
    """Splat a type into a variadic sequence.

    Parameters:
        Trait: The trait that the types conform to.
        count: The number of times to splat the type.
        type: The type to splat.
    """

    comptime splat_value[
        T: AnyType, //, count: Int, value: T
    ]: Variadic.ValuesOfType[T] = Self.tabulate[
        count, _SplatValueTabulator[value, _]
    ]
    """Splat a value into a variadic sequence.

    Parameters:
        T: The type of the value to splat.
        count: The number of times to splat the value.
        value: The value to splat.
    """

    comptime tabulate[
        ToT: AnyType,
        //,
        count: Int,
        Mapper: _TabulateIntToValueGeneratorType[ToT],
    ]: Variadic.ValuesOfType[ToT] = __mlir_attr[
        `#kgen.variadic.tabulate<`,
        count._mlir_value,
        `,`,
        _IndexToIntTabulateWrap[Mapper, ...],
        `> : `,
        Variadic.ValuesOfType[ToT],
    ]
    """Apply an "index -> value" generator, N times to build a variadic.

    Parameters:
        ToT: The type of the values in the variadic sequence.
        count: The number of times to apply the generator.
        Mapper: The generator to apply, mapping from Int to ToT.
    """

    comptime tabulate_type[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        count: Int,
        Mapper: _TabulateIntToTypeGeneratorType[Trait, ToT],
    ]: Variadic.TypesOfTrait[Trait] = __mlir_attr[
        `#kgen.variadic.tabulate<`,
        count._mlir_value,
        `,`,
        _IndexToIntTypeTabulateWrap[Trait=Trait, ToT=ToT, Mapper, ...],
        `> : `,
        Variadic.TypesOfTrait[Trait],
    ]
    """Apply an "index -> value" generator, N times to build a variadic.

    Parameters:
        Trait: The trait that the types conform to.
        ToT: The type of the values in the variadic sequence.
        count: The number of times to apply the generator.
        Mapper: The generator to apply, mapping from Int to ToT.
    """

    comptime contains[
        Trait: type_of(AnyType),
        //,
        type: Trait,
        element_types: Variadic.TypesOfTrait[Trait],
    ] = _ReduceVariadicAndIdxToValue[
        BaseVal=Variadic.values[False],
        VariadicType=element_types,
        #  Curry `_ContainsMapper` to fit the reducer signature
        Reducer=_ContainsReducer[Trait=Trait, Type=type, ...],
    ][
        0
    ]
    """
    Check if a type is contained in a variadic sequence.

    Parameters:
        Trait: The trait that the types conform to.
        type: The type to check for.
        element_types: The variadic sequence of types to search.
    """

    comptime contains_value[
        T: Equatable,
        //,
        value: T,
        element_values: Variadic.ValuesOfType[T],
    ] = _ReduceValueAndIdxToValue[
        BaseVal=Variadic.values[False],
        VariadicType=element_values,
        #  Curry `_ContainsValueReducer` to fit the reducer signature
        Reducer=_ContainsValueReducer[T=T, value=value, ...],
    ][
        0
    ]
    """
    Check if a value is contained in a variadic sequence of values.

    Parameters:
        T: The type of the values. Must be `Equatable`.
        value: The value to search for.
        element_values: The variadic sequence of values to search.
    """

    comptime map_types_to_types[
        From: type_of(AnyType),
        To: type_of(AnyType),
        //,
        element_types: Variadic.TypesOfTrait[From],
        Mapper: _TypeToTypeGenerator[From, To],
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[To],
        VariadicType=element_types,
        Reducer=_MapTypeToTypeReducer[From, To, Mapper, ...],
    ]
    """Map a variadic of types to a new variadic of types using a mapper.

    Returns a new variadic of types resulting from applying `Mapper[T]` to each
    type in the input variadic.

    Parameters:
        From: The trait that the input types conform to.
        To: The trait that the output types conform to.
        element_types: The input variadic of types to map.
        Mapper: A generator that maps a type to another type. The generator type is `[T: From] -> To`.

    Examples:

    ```mojo
    from std.builtin.variadics import Variadic
    from std.testing import *

    trait MyError:
        comptime ErrorType: AnyType

    struct Foo(MyError):
        comptime ErrorType = Int

    struct Baz(MyError):
        comptime ErrorType = String

    # Given a variadic of types [Foo, Baz]
    comptime input_types = Variadic.types[T=MyError, Foo, Baz]

    # And a mapper that maps the type to it's MyError `ErrorType` type
    comptime mapper[T: MyError] = T.ErrorType

    # The resulting variadic of types is [Int, String]
    comptime output = Variadic.map_types_to_types[input_types, mapper]

    assert_equal(Variadic.size(output), 2)
    assert_true(_type_is_eq[output[0], Int]())
    assert_true(_type_is_eq[output[1], String]())
    ```
    """

    comptime slice_types[
        T: type_of(AnyType),
        //,
        element_types: Variadic.TypesOfTrait[T],
        start: Int where start >= 0 = 0,
        end: Int where (
            start <= end <= Variadic.size(element_types)
        ) = Variadic.size(element_types),
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[T],
        VariadicType=element_types,
        Reducer=_SliceReducer[T, start, end, ...],
    ]
    """Extract a contiguous subsequence from a variadic sequence.

    Returns a new variadic containing elements from index `start` (inclusive)
    to index `end` (exclusive). Similar to Python's slice notation [start:end].

    Parameters:
        T: The trait that the types conform to.
        element_types: The variadic sequence to slice.
        start: The starting index (inclusive).
        end: The ending index (exclusive).

    Constraints:
        - 0 <= start <= end <= Variadic.size(element_types)

    Examples:
        ```mojo
        from std.builtin.variadics import Variadic
        # Given a variadic of types [Int, String, Float64, Bool]
        comptime MyTypes = Tuple[Int, String, Float64, Bool].element_types
        # Extract middle elements: [String, Float64]
        comptime Sliced = Variadic.slice_types[start=1, end=3, element_types=MyTypes]
        # Extract first two: [Int, String]
        comptime First2 = Variadic.slice_types[start=0, end=2, element_types=MyTypes]
        # Extract last element: [Bool]
        comptime Last = Variadic.slice_types[start=3, end=4, element_types=MyTypes]
        ```
    """

    comptime zip_types[
        Trait: type_of(AnyType), //, *types: Variadic.TypesOfTrait[Trait]
    ] = __mlir_attr[
        `#kgen.variadic.zip<`,
        types,
        `> : !kgen.variadic<`,
        Variadic.TypesOfTrait[Trait],
        `>`,
    ]
    """
    Zips a group of variadics of types together.

    Parameters:
        Trait: The trait that the types conform to.
        types: The type to check for.
    """

    comptime zip_values[
        type: AnyType, //, *values: Variadic.ValuesOfType[type]
    ] = __mlir_attr[
        `#kgen.variadic.zip<`,
        values,
        `> : !kgen.variadic<`,
        Variadic.ValuesOfType[type],
        `>`,
    ]
    """
    Zips a group of variadics of values together.

    Parameters:
        type: The type that the values conform to.
        values: The values to zip.
    """

    comptime filter_types[
        T: type_of(AnyType),
        //,
        *element_types: T,
        predicate: _TypePredicateGenerator[T],
    ] = _ReduceVariadicAndIdxToVariadic[
        BaseVal=Variadic.empty_of_trait[T],
        VariadicType=element_types,
        Reducer=_FilterReducer[T, predicate, ...],
    ]
    """Filter types from a variadic sequence based on a predicate function.

    Returns a new variadic containing only the types for which the predicate
    returns True.

    Parameters:
        T: The trait that the types conform to.
        element_types: The input variadic sequence.
        predicate: A generator function that takes a type and returns Bool.

    Examples:

    ```mojo
    from std.builtin.variadics import Variadic
    from std.utils import Variant
    from std.sys.intrinsics import _type_is_eq

    comptime FullVariant = Variant[Int, String, Float64, Bool]

    # Exclude a single type
    comptime IsNotInt[Type: AnyType] = not _type_is_eq[Type, Int]()
    comptime WithoutInt = Variadic.filter_types[*FullVariant.Ts, predicate=IsNotInt]
    comptime FilteredVariant = Variant[*WithoutInt]
    # FilteredVariant is Variant[String, Float64, Bool]

    # Keep only specific types
    comptime IsNumeric[Type: AnyType] = (
        _type_is_eq[Type, Int]() or _type_is_eq[Type, Float64]()
    )
    comptime OnlyNumeric = Variadic.filter_types[*FullVariant.Ts, predicate=IsNumeric]
    # OnlyNumeric is Variadic.types[T=AnyType, Int, Float64]

    # Exclude multiple types using a variadic check
    comptime ExcludeList = Variadic.types[T=AnyType, Int, Bool]
    comptime NotInList[Type: AnyType] = not Variadic.contains[
        type=Type, element_types=ExcludeList
    ]
    comptime Filtered = Variadic.filter_types[*FullVariant.Ts, predicate=NotInList]
    # Filtered is Variadic.types[T=AnyType, String, Float64]
    ```

    Filter operations can be chained for complex transformations:

    ```mojo
    comptime IsNotBool[Type: AnyType] = not _type_is_eq[Type, Bool]()
    comptime Step1 = Variadic.filter_types[*FullVariant.Ts, predicate=IsNotBool]
    comptime Step2 = Variadic.filter_types[*Step1, predicate=IsNotInt]
    comptime ChainedVariant = Variant[*Step2]
    ```
    """

    comptime _ValueIdxToValueGeneratorType[
        From: AnyType, To: AnyType
    ] = __mlir_type[
        `!lit.generator<<"From": `,
        +From,
        `, "Idx":`,
        Int,
        `>`,
        +To,
        `>`,
    ]
    """This specifies a generator to generate a generator type for the reducer of
    values. The result generator type is [From, idx: Int] -> To,
    """

    comptime _ValueToValueMapper[
        FromType: AnyType,
        ToType: AnyType,
        //,
        Mapper: Variadic._ValueIdxToValueGeneratorType[FromType, ToType],
        Prev: Variadic.ValuesOfType[ToType],
        From: Variadic.ValuesOfType[FromType],
        idx: Int,
    ] = Variadic.concat_values[
        Prev,
        Variadic.values[Mapper[From[idx], idx]],
    ]


# ===-----------------------------------------------------------------------===#
# VariadicParamList
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _VariadicParamListIter[type: Copyable, //, *values: type](
    ImplicitlyCopyable, Iterable, Iterator
):
    """Const Iterator for VariadicParamList.

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.
    """

    comptime Element = Self.type
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[StaticConstantOrigin] Self.type:
        var index = self.index

        comptime params = VariadicParamList[*Self.values]()
        if index >= params.size:
            raise StopIteration()
        self.index = index + 1
        return params[index]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = VariadicParamList[*Self.values].size - self.index
        return (len, {len})


# TODO: Make this conform to Iterable when IteratorType can be conditionally
# defined only when 'type' is Copyable.
struct VariadicParamList[type: AnyType, //, *values: type](
    Sized, TrivialRegisterPassable, Writable
):
    """A utility class to access homogeneous variadic parameters.

    `VariadicParamList` is used by homogenous variadic parameter lists. Unlike
    `VariadicPack` (which is heterogeneous), `VariadicParamList` requires all
    elements to have the same type.

    `VariadicParamList` is only used for parameter lists, `VariadicList` is
    used for function arguments.

    For example, in the following function signature, `*args: Int` creates a
    `VariadicParamList` because it uses a single type `Int` instead of a variadic type
    parameter. The `*` before `args` indicates that `args` is a variadic argument,
    which means that the function can accept any number of arguments, but all
    arguments must have the same type `Int`.

    ```mojo
    def sum_values[*args: Int]() -> Int:
        var total = 0

        # Can use regular for loop because args is a VariadicParamList
        for i in range(len(args)):
            total += args[i]  # All elements are Int, so uniform access

        return total

    def main():
        print(sum_values(1, 2, 3, 4, 5))
    ```

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.
    """

    comptime size = Int(
        mlir_value=__mlir_attr[
            `#kgen.variadic.size<:`,
            type_of(Self.values),
            ` `,
            +Self.values,
            `> : index`,
        ]
    )
    """The number of elements in the list."""

    @always_inline
    def __init__(out self):
        """Constructs a VariadicParamList."""
        pass

    @always_inline
    def __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return Self.size

    @staticmethod
    def get_span() -> Span[Self.type, StaticConstantOrigin]:
        """Gets a span of the elements on the variadic list.

        Returns:
            A span of the elements on the variadic list.
        """

        # Convert 'values' to use a flat array representation.
        comptime array = __mlir_attr[
            `#pop.variadic_to_array<:`,
            type_of(Self.values),
            ` `,
            +Self.values,
            `>`,
        ]
        # Map it into a runtime constant.
        ref static_array = global_constant[array]()
        # Get a pointer to the first element, not the whole array.
        var first_elt = UnsafePointer(to=static_array).bitcast[Self.type]()
        return Span(ptr=first_elt, length=Self.size)

    @always_inline
    def __getitem__(self, idx: Int) -> ref[StaticConstantOrigin] Self.type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            The element on the list corresponding to the given index.
        """
        return self.get_span()[idx]

    comptime __getitem_param__[idx: Int]: Self.type = __mlir_attr[
        `#kgen.variadic.get<:`,
        type_of(Self.values),
        ` `,
        +Self.values,
        `, `,
        idx._mlir_value,
        `> : `,
        +Self.type,
    ]
    """Gets a single element on the variadic list."""

    def _write_elements[is_repr: Bool = False](self, mut writer: Some[Writer]):
        _constrained_conforms_to[
            conforms_to(Self.type, Writable),
            Parent=Self,
            Element=Self.type,
            ParentConformsTo="Writable",
            ElementConformsTo="Writable",
        ]()
        writer.write_string("(")
        for i in range(len(self)):
            if i > 0:
                writer.write_string(", ")

            comptime if is_repr:
                trait_downcast[Writable](self[i]).write_repr_to(writer)
            else:
                trait_downcast[Writable](self[i]).write_to(writer)
        writer.write_string(")")

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the elements of this variadic list to a writer.

        Constraints:
            `type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_elements(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this variadic list to a writer.

        Constraints:
            `type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        @parameter
        def write_fields(mut w: Some[Writer]):
            self._write_elements[is_repr=True](w)

        FormatStruct(writer, "VariadicParamList").params(
            TypeNames[Self.type](),
        ).fields[FieldsFn=write_fields]()

    # We can only support iteration when the elements are Copyable, because
    # iterators currently need to return the elements by value.
    @always_inline
    def __iter__(
        ref self,
    ) -> _VariadicParamListIter[
        *rebind[Variadic.ValuesOfType[downcast[Self.type, Copyable]]](
            Self.values
        )
    ] where conforms_to(Self.type, Copyable):
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return {0}


# ===-----------------------------------------------------------------------===#
# VariadicList
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _VariadicListIter[
    elt_is_mutable: Bool,
    //,
    elt_type: AnyType,
    elt_origin: Origin[mut=elt_is_mutable],
    list_origin: ImmutOrigin,
    is_owned: Bool,
](RegisterPassable):
    """Iterator for VariadicList.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_origin: The origin of the elements.
        list_origin: The origin of the VariadicList.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    comptime variadic_list_type = VariadicList[
        origin=Self.elt_origin,
        Self.elt_type,
        Self.is_owned,
    ]

    comptime Element = Self.elt_type

    var index: Int
    var src: Pointer[
        Self.variadic_list_type,
        Self.list_origin,
    ]

    def __init__(
        out self,
        index: Int,
        ref[Self.list_origin] list: Self.variadic_list_type,
    ):
        self.index = index
        self.src = Pointer(to=list)

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.elt_origin._mlir_origin] Self.elt_type:
        var index = self.index
        if index >= len(self.src[]):
            raise StopIteration()
        self.index = index + 1
        return rebind[Self.variadic_list_type.reference_type](
            Pointer(to=self.src[][index])
        )[]


struct VariadicList[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable],
    //,
    element_type: AnyType,
    is_owned: Bool,
](RegisterPassable, Sized, Writable):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes references to the elements in a
    way that can be enumerated.  Each element may be accessed with `elt[]`.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument.
        origin: The origin of the underlying elements.
        element_type: The type of the elements in the list.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    comptime reference_type = Pointer[Self.element_type, Self.origin]
    """The pointer type for references to elements."""

    comptime _mlir_type = Variadic.ValuesOfType[Self.reference_type._mlir_type]

    var value: Self._mlir_type
    """The underlying storage, a variadic list of references to elements of the
    given type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # Provide support for read-only variadic arguments.
    @doc_hidden
    @always_inline
    @implicit
    def __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value

    # The destructor for this type is trivial if not an "owned" list.
    comptime __del__is_trivial: Bool = not Self.is_owned

    @always_inline
    def __del__(deinit self):
        """Destructor that releases elements if owned."""

        # Destroy each element if this variadic has owned elements, destroy
        # them.  We destroy in backwards order to match how arguments are
        # normally torn down when CheckLifetimes is left to its own devices.
        comptime if Self.is_owned:
            _constrained_conforms_to[
                conforms_to(Self.element_type, ImplicitlyDestructible),
                Parent=Self,
                Element=Self.element_type,
                ParentConformsTo="ImplicitlyDestructible",
            ]()
            comptime TDestructible = downcast[
                Self.element_type, ImplicitlyDestructible
            ]

            for i in reversed(range(len(self))):
                # Safety: We own the elements in this list.
                UnsafePointer(to=self[i]).mut_cast[True]().bitcast[
                    TDestructible
                ]().destroy_pointee()

    def consume_elements[
        elt_handler: def(idx: Int, var elt: Self.element_type) capturing
    ](deinit self):
        """Consume the variadic list by transferring ownership of each element
        into the provided closure one at a time.  This is only valid on 'owned'
        variadic lists.

        Parameters:
            elt_handler: A function that will be called for each element of the
                         list.
        """

        comptime assert (
            Self.is_owned
        ), "consume_elements may only be called on owned variadic lists"

        for i in range(len(self)):
            var ptr = UnsafePointer(to=self[i])
            # TODO: Cannot use UnsafePointer.take_pointee because it requires
            # the element to be Movable, which is not required here.
            elt_handler(i, __get_address_as_owned_value(ptr.address))

    # FIXME: This is a hack to work around a miscompile, do not use.
    def _annihilate(deinit self):
        pass

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return Int(mlir_value=__mlir_op.`pop.variadic.size`(self.value))

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __getitem__[
        self_origin: ImmutOrigin
    ](ref[self_origin] self, idx: Int) -> ref[
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        origin_of(Self.origin, self_origin).unsafe_mut_cast[
            Self.elt_is_mutable
        ]()
    ] Self.element_type:
        """Gets a single element on the variadic list.

        Parameters:
            self_origin: The origin of the list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __get_litref_as_mvalue(
            __mlir_op.`pop.variadic.get`(self.value, idx._mlir_value)
        )

    def _write_elements[is_repr: Bool = False](self, mut writer: Some[Writer]):
        _constrained_conforms_to[
            conforms_to(Self.element_type, Writable),
            Parent=Self,
            Element=Self.element_type,
            ParentConformsTo="Writable",
            ElementConformsTo="Writable",
        ]()
        writer.write_string("(")
        for i in range(len(self)):
            if i > 0:
                writer.write_string(", ")

            comptime if is_repr:
                trait_downcast[Writable](self[i]).write_repr_to(writer)
            else:
                trait_downcast[Writable](self[i]).write_to(writer)
        writer.write_string(")")

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the elements of this variadic list to a writer.

        Constraints:
            `element_type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_elements(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this variadic list to a writer.

        Constraints:
            `element_type` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        @parameter
        def write_fields(mut w: Some[Writer]):
            self._write_elements[is_repr=True](w)

        FormatStruct(writer, "VariadicList").params(
            TypeNames[Self.element_type](),
        ).fields[FieldsFn=write_fields]()

    def __iter__[
        self_origin: ImmutOrigin
    ](
        ref[self_origin] self,
    ) -> _VariadicListIter[
        Self.element_type, Self.origin, self_origin, Self.is_owned
    ]:
        """Iterate over the list.

        Parameters:
            self_origin: The origin of the list.

        Returns:
            An iterator to the start of the list.
        """
        return {0, self}


# ===-----------------------------------------------------------------------===#
# VariadicPack
# ===-----------------------------------------------------------------------===#


struct VariadicPack[
    elt_is_mutable: Bool,
    origin: Origin[mut=elt_is_mutable],
    //,
    is_owned: Bool,
    element_trait: type_of(AnyType),
    *element_types: element_trait,
](Copyable, RegisterPassable, Sized):
    """A utility class to access heterogeneous variadic function arguments.

    `VariadicPack` is used when you need to accept variadic arguments where each
    argument can have a different type, but all types conform to a common trait.
    Unlike `VariadicParamList` (which is homogeneous), `VariadicPack` allows each
    element to have a different concrete type.

    `VariadicPack` is essentially a heterogeneous tuple that gets lowered to a
    struct at runtime. Because `VariadicPack` is a heterogeneous tuple (not an
    array), each element can have a different size and memory layout, which
    means the compiler needs to know the exact type of each element at compile
    time to generate the correct memory layout and access code.

    Therefore, indexing into `VariadicPack` requires compile-time indices using
    `comptime for` loops, whereas indexing into `VariadicParamList` uses runtime
    indices.

    For example, in the following function signature, `*args: *ArgTypes` creates a
    `VariadicPack` because it uses a variadic type parameter `*ArgTypes` instead
    of a single type. The `*` before `ArgTypes` indicates that `ArgTypes` is a
    variadic type parameter, which means that the function can accept any number
    of arguments, and each argument can have a different type. This allows each
    argument to have a different type while all types must conform to the
    `Intable` trait.

    ```mojo
    def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
        var total = 0

        # Must use comptime for loop because args is a VariadicPack
        comptime for i in range(args.__len__()):
            # Each args[i] has a different concrete type from *ArgTypes
            # The compiler generates specific code for each iteration
            total += Int(args[i])

        return total

    def main() raises:
        print(count_many_things(5, 11.7, 12))  # Prints: 28
    ```

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument pack.
        origin: The origin of the underlying elements.
        is_owned: Whether the elements are owned by the pack. If so, the pack
                  will release the elements when it is destroyed.
        element_trait: The trait that each element of the pack conforms to.
        element_types: The list of types held by the argument pack.
    """

    comptime _mlir_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        Self.element_trait,
        `> `,
        Self.element_types,
        `, `,
        Self.origin._mlir_origin,
        `>`,
    ]

    var _value: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline("nodebug")
    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_disable_nested_origin_exclusivity
    def __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicPack from the internal representation.

        Args:
            value: The argument to construct the pack with.
        """
        self._value = value

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Copy construct the variadic pack.

        Args:
            copy: The pack to copy from.

        Constraints:
            The variadic pack must not be owned.
        """

        comptime assert not Self.is_owned, "Cannot copy an owned variadic pack."
        self._value = copy._value

    # The destructor for this type is trivial if not an "owned" pack.
    comptime __del__is_trivial: Bool = not Self.is_owned

    @always_inline("nodebug")
    def __del__(deinit self):
        """Destructor that releases elements if owned."""

        comptime if Self.is_owned:
            comptime for i in reversed(range(Self.__len__())):
                # FIXME(MOCO-2953):
                #   Due to a compiler limitation, we can't use
                #   conforms_to() here, meaning the `trait_downcast` below
                #   could fail with a worse elaboration error than we'd get from
                #   _constrained_conforms_to.
                #
                # comptime element_type = Self.element_types[i]
                # _constrained_conforms_to[
                #     conforms_to(element_type, ImplicitlyDestructible),
                #     Parent=Self,
                #     Element=element_type,
                #     ParentConformsTo="ImplicitlyDestructible",
                # ]()

                # Safety: We own the elements in this pack.
                UnsafePointer(
                    to=trait_downcast[ImplicitlyDestructible](self[i])
                ).mut_cast[True]().destroy_pointee()

    def consume_elements[
        elt_handler: def[idx: Int](var elt: Self.element_types[idx]) capturing
    ](deinit self):
        """Consume the variadic pack by transferring ownership of each element
        into the provided closure one at a time.  This is only valid on 'owned'
        variadic packs.

        Parameters:
            elt_handler: A function that will be called for each element of the
                         pack.
        """

        comptime assert (
            Self.is_owned
        ), "consume_elements may only be called on owned variadic packs"

        comptime for i in range(Self.__len__()):
            var ptr = UnsafePointer(to=self[i])
            # TODO: Cannot use UnsafePointer.take_pointee because it requires
            # the element to be Movable, which is not required here.
            elt_handler[i](__get_address_as_owned_value(ptr.address))

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    @staticmethod
    def __len__() -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """

        comptime result = Variadic.size(Self.element_types)
        return result

    @always_inline
    def __len__(self) -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """
        return Self.__len__()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __getitem_param__[
        index: Int
    ](self) -> ref[Self.origin] Self.element_types[index]:
        """Return a reference to an element of the pack.

        Parameters:
            index: The element of the pack to return.

        Returns:
            A reference to the element.  The Pointer's mutability follows the
            mutability of the pack argument convention.
        """
        litref_elt = __mlir_op.`lit.ref.pack.extract`[
            index=index.__mlir_index__()
        ](self._value)
        return __get_litref_as_mvalue(litref_elt)

    # ===-------------------------------------------------------------------===#
    # C Pack Utilities
    # ===-------------------------------------------------------------------===#

    # FIXME: bound by AnyType
    comptime _kgen_element_types = rebind[
        Variadic.ValuesOfType[__mlir_type.`!kgen.type`]
    ](Self.element_types)
    """This is the element_types list lowered to `variadic<type>` type for kgen.
    """

    # FIXME: bound by AnyType
    comptime _variadic_pointer_types = __mlir_attr[
        `#kgen.param.expr<variadic_ptr_map, `,
        Self._kgen_element_types,
        `, 0: index>: `,
        Variadic.ValuesOfType[__mlir_type.`!kgen.type`],
    ]
    """Use variadic_ptr_map to construct the type list of the !kgen.pack that
    the !lit.ref.pack will lower to.  It exposes the pointers introduced by the
    references.
    """
    comptime _kgen_pack_with_pointer_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_pointer_types, `>`
    ]
    """This is the !kgen.pack type with pointer elements."""

    @doc_hidden
    @always_inline("nodebug")
    def get_as_kgen_pack(self) -> Self._kgen_pack_with_pointer_type:
        """This rebinds `in_pack` to the equivalent `!kgen.pack` with kgen
        pointers."""
        return rebind[Self._kgen_pack_with_pointer_type](self._value)

    # FIXME: bound by AnyType
    comptime _variadic_with_pointers_removed = __mlir_attr[
        `#kgen.param.expr<variadic_ptrremove_map, `,
        Self._variadic_pointer_types,
        `>: `,
        Variadic.ValuesOfType[__mlir_type.`!kgen.type`],
    ]
    comptime _loaded_kgen_pack_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_with_pointers_removed, `>`
    ]
    """This is the `!kgen.pack` type that happens if one loads all the elements
    of the pack.
    """

    # Returns all the elements in a kgen.pack.
    # Useful for FFI, such as calling printf. Otherwise, avoid this if possible.
    @doc_hidden
    @always_inline("nodebug")
    def get_loaded_kgen_pack(self) -> Self._loaded_kgen_pack_type:
        """This returns the stored KGEN pack after loading all of the elements.
        """
        return __mlir_op.`kgen.pack.load`(self.get_as_kgen_pack())

    def _write_to[
        O1: ImmutOrigin,
        O2: ImmutOrigin,
        O3: ImmutOrigin,
        *,
        is_repr: Bool = False,
    ](
        self: VariadicPack[_, Writable, ...],
        mut writer: Some[Writer],
        start: StringSlice[O1] = StaticString(""),
        end: StringSlice[O2] = StaticString(""),
        sep: StringSlice[O3] = StaticString(", "),
    ):
        """Writes a sequence of writable values from a pack to a writer with
        delimiters.

        This function formats a variadic pack of writable values as a delimited
        sequence, writing each element separated by the specified separator and
        enclosed by start and end delimiters.

        Parameters:
            O1: The origin of the open `StringSlice`.
            O2: The origin of the close `StringSlice`.
            O3: The origin of the separator `StringSlice`.
            is_repr: Whether to use repr formatting for elements.

        Args:
            writer: The writer to write to.
            start: The starting delimiter.
            end: The ending delimiter.
            sep: The separator between items.
        """
        writer.write_string(start)

        comptime for i in range(self.__len__()):
            comptime if i != 0:
                writer.write_string(sep)

            comptime if is_repr:
                self[i].write_repr_to(writer)
            else:
                self[i].write_to(writer)
        writer.write_string(end)

    @no_inline
    def write_to(
        self: VariadicPack[_, Writable, ...], mut writer: Some[Writer]
    ):
        """Writes the elements of this pack to a writer.

        Args:
            writer: The object to write to.
        """
        self._write_to(
            writer,
            start=StaticString("("),
            end=StaticString(")"),
        )

    @no_inline
    def write_repr_to(
        self: VariadicPack[_, Writable, ...], mut writer: Some[Writer]
    ):
        """Writes the repr of the elements of this pack to a writer.

        Args:
            writer: The object to write to.
        """
        self._write_to[is_repr=True](
            writer,
            start=StaticString("("),
            end=StaticString(")"),
        )


# ===-----------------------------------------------------------------------===#
# Tabulate Helpers
# ===-----------------------------------------------------------------------===#

comptime _TabulateIntToValueGeneratorType[ToT: AnyType] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `>`,
    +ToT,
    `>`,
]

comptime _TabulateIntToTypeGeneratorType[
    Trait: type_of(AnyType), ToT: Trait
] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `> `,
    Trait,
    `>`,
]


comptime _IndexToIntTabulateWrap[
    ToT: AnyType,
    //,
    ToWrap: _TabulateIntToValueGeneratorType[ToT],
    idx: __mlir_type.index,
]: ToT = ToWrap[Int(mlir_value=idx)]

comptime _IndexToIntTypeTabulateWrap[
    Trait: type_of(AnyType),
    ToT: Trait,
    //,
    ToWrap: _TabulateIntToTypeGeneratorType[Trait, ToT],
    idx: __mlir_type.index,
] = ToWrap[Int(mlir_value=idx)]


comptime _SplatValueTabulator[T: AnyType, //, value: T, index: Int] = value
comptime _SplatTypeTabulator[
    Trait: type_of(AnyType), T: Trait, index: Int
]: Trait = T

# ===-----------------------------------------------------------------------===#
# VariadicReduce
# ===-----------------------------------------------------------------------===#


comptime _ReduceVariadicIdxGeneratorTypeGenerator[
    Prev: AnyType, From: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"Prev": `,
    +Prev,
    `, "From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    +Prev,
    `>`,
]
"""This specifies a generator to generate a generator type for the reducer.
The generated generator type is [Prev: AnyType, Ts: Variadic.TypesOfTrait[AnyType], idx: Int] -> Prev,
"""

comptime _IndexToIntWrap[
    From: type_of(AnyType),
    ReduceT: AnyType,
    ToWrap: _ReduceVariadicIdxGeneratorTypeGenerator[ReduceT, From],
    PrevV: ReduceT,
    VA: Variadic.TypesOfTrait[From],
    idx: __mlir_type.index,
] = ToWrap[PrevV, VA, Int(mlir_value=idx)]
"""Wrapper for type -> value."""

comptime _ReduceVariadicAndIdxToVariadic[
    From: type_of(AnyType),
    To: type_of(AnyType),
    //,
    *,
    BaseVal: Variadic.TypesOfTrait[To],
    VariadicType: Variadic.TypesOfTrait[From],
    Reducer: _ReduceVariadicIdxGeneratorTypeGenerator[
        Variadic.TypesOfTrait[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntWrap[From, Variadic.TypesOfTrait[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
"""Construct a new variadic of types using a reducer. To reduce to a single
type, one could reduce the input to a single element variadic instead.

Parameters:
    From: The common trait bound for the input variadic types.
    To: The common trait bound for the output variadic types.
    BaseVal: The initial value to reduce on.
    VariadicType: The variadic to be reduced.
    Reducer: A `[BaseVal: Variadic.TypesOfTrait[To], Ts: *From, idx: index] -> To` that does the reduction.
"""


comptime _ReduceValueIdxGeneratorTypeGenerator[
    Prev: AnyType, From: AnyType
] = __mlir_type[
    `!lit.generator<<"Prev": `,
    +Prev,
    `, "From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    +Prev,
    `>`,
]
"""This specifies a generator to generate a generator type for the reducer.
The generated generator type is [Prev: AnyType, Ts: Variadic.ValuesOfType[AnyType], idx: Int] -> Prev,
"""


comptime _IndexToIntValueWrap[
    From: AnyType,
    ReduceT: AnyType,
    ToWrap: _ReduceValueIdxGeneratorTypeGenerator[ReduceT, From],
    PrevV: ReduceT,
    VA: Variadic.ValuesOfType[From],
    idx: __mlir_type.index,
] = ToWrap[PrevV, VA, Int(mlir_value=idx)]


comptime _ReduceValueAndIdxToVariadic[
    From: AnyType,
    To: type_of(AnyType),
    //,
    *,
    BaseVal: Variadic.TypesOfTrait[To],
    VariadicType: Variadic.ValuesOfType[From],
    Reducer: _ReduceValueIdxGeneratorTypeGenerator[
        Variadic.TypesOfTrait[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntValueWrap[From, Variadic.TypesOfTrait[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
"""Construct a new variadic of types using a reducer. To reduce to a single
type, one could reduce the input to a single element variadic instead.

Parameters:
    From: The type of the input variadic values.
    To: The common trait bound for the output variadic types.
    BaseVal: The initial value to reduce on.
    VariadicType: The variadic to be reduced.
    Reducer: A `[BaseVal: Variadic.ValuesOfType[To], Ts: *From, idx: index] -> To` that does the reduction.
"""


comptime _ReduceValueAndIdxToValue[
    To: AnyType,
    From: AnyType,
    //,
    *,
    BaseVal: Variadic.ValuesOfType[To],
    VariadicType: Variadic.ValuesOfType[From],
    Reducer: _ReduceValueIdxGeneratorTypeGenerator[
        Variadic.ValuesOfType[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntValueWrap[From, Variadic.ValuesOfType[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
"""Construct a new variadic of values using a reducer over an input variadic of
values.

Parameters:
    To: The type of the output variadic values.
    From: The type of the input variadic values.
    BaseVal: The initial value to reduce on.
    VariadicType: The variadic of values to be reduced.
    Reducer: A `[BaseVal: Variadic.ValuesOfType[To], Ts: Variadic.ValuesOfType[From], idx: Int] -> Variadic.ValuesOfType[To]` that does the reduction.
"""


comptime _ReduceVariadicAndIdxToValue[
    To: AnyType,
    From: type_of(AnyType),
    //,
    *,
    BaseVal: Variadic.ValuesOfType[To],
    VariadicType: Variadic.TypesOfTrait[From],
    Reducer: _ReduceVariadicIdxGeneratorTypeGenerator[
        Variadic.ValuesOfType[To], From
    ],
] = __mlir_attr[
    `#kgen.variadic.reduce<`,
    BaseVal,
    `,`,
    VariadicType,
    `,`,
    _IndexToIntWrap[From, Variadic.ValuesOfType[To], Reducer, ...],
    `> : `,
    type_of(BaseVal),
]
"""Construct a new variadic of types using a reducer. To reduce to a single
type, one could reduce the input to a single element variadic instead.

Parameters:
    To: The type of the output variadic values.
    From: The common trait bound for the input variadic types.
    BaseVal: The initial value to reduce on.
    VariadicType: The variadic to be reduced.
    Reducer: A `[BaseVal: Variadic.ValuesOfType[To], Ts: *From, idx: index] -> To` that does the reduction.
"""


# ===-----------------------------------------------------------------------===#
# VariadicMap
# ===-----------------------------------------------------------------------===#

comptime _TypeToTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[`!lit.generator<<"From":`, From, `>`, To, `>`]
"""A generator of type [T: From] -> To, which maps a type to another type."""

comptime _VariadicIdxToTypeGeneratorTypeGenerator[
    From: type_of(AnyType), To: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    To,
    `>`,
]
"""This specifies a generator to generate a generator type for the mapper.
The generated generator type is [Ts: Variadic.TypesOfTrait[AnyType], idx: Int] -> AnyType,
which maps the input variadic + index of the current element to another type.
"""


comptime _WrapVariadicIdxToTypeMapperToReducer[
    F: type_of(AnyType),
    T: type_of(AnyType),
    Mapper: _VariadicIdxToTypeGeneratorTypeGenerator[F, T],
    Prev: Variadic.TypesOfTrait[T],
    From: Variadic.TypesOfTrait[F],
    Idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[Mapper[From, Idx]]]


comptime _MapVariadicAndIdxToType[
    From: type_of(AnyType),
    //,
    *,
    To: type_of(AnyType),
    VariadicType: Variadic.TypesOfTrait[From],
    Mapper: _VariadicIdxToTypeGeneratorTypeGenerator[From, To],
] = _ReduceVariadicAndIdxToVariadic[
    BaseVal=Variadic.empty_of_trait[To],  # reduce from a empty variadic
    VariadicType=VariadicType,
    Reducer=_WrapVariadicIdxToTypeMapperToReducer[From, To, Mapper, ...],
]
"""Construct a new variadic of types using a type-to-type mapper.

Parameters:
    From: The common trait bound for the input variadic types.
    To: A common trait bound for the mapped type.
    VariadicType: The variadic to be mapped.
    Mapper: A `[Ts: *From, idx: index] -> To` that does the transform.
"""


comptime _VariadicValuesIdxToTypeGeneratorTypeGenerator[
    From: AnyType, To: type_of(AnyType)
] = __mlir_type[
    `!lit.generator<<"From": !kgen.variadic<`,
    From,
    `>, "Idx":`,
    Int,
    `>`,
    To,
    `>`,
]
"""This specifies a generator to generate a generator type for the mapper.
The generated generator type is [Ts: Variadic.TypesOfTrait[AnyType], idx: Int] -> AnyType,
which maps the input variadic + index of the current element to another type.
"""

comptime _WrapVariadicValuesIdxToTypeMapperToReducer[
    F: AnyType,
    T: type_of(AnyType),
    Mapper: _VariadicValuesIdxToTypeGeneratorTypeGenerator[F, T],
    Prev: Variadic.TypesOfTrait[T],
    From: Variadic.ValuesOfType[F],
    Idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[Mapper[From, Idx]]]

comptime _ReversedVariadic[
    T: type_of(AnyType),
    element_types: Variadic.TypesOfTrait[T],
    idx: Int,
] = element_types[Variadic.size(element_types) - 1 - idx]
"""A generator that reverses a variadic sequence of types.

Parameters:
    T: The common trait bound for the variadic types.
    element_types: The variadic sequence of types to reverse.
    idx: The index of the type to generate in the reversed sequence.
"""


comptime _ContainsReducer[
    Trait: type_of(AnyType),
    Type: Trait,
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = Variadic.values[_type_is_eq_parse_time[From[idx], Type]() or Prev[0]]

comptime _ContainsValueReducer[
    T: Equatable,
    value: T,
    Prev: Variadic.ValuesOfType[Bool],
    From: Variadic.ValuesOfType[T],
    idx: Int,
] = Variadic.values[From[idx] == value or Prev[0]]

comptime _MapTypeToTypeReducer[
    FromTrait: type_of(AnyType),
    ToTrait: type_of(AnyType),
    Mapper: _TypeToTypeGenerator[FromTrait, ToTrait],
    Prev: Variadic.TypesOfTrait[ToTrait],
    From: Variadic.TypesOfTrait[FromTrait],
    idx: Int,
] = Variadic.concat_types[Prev, Variadic.types[T=ToTrait, Mapper[From[idx]]]]

comptime _SliceReducer[
    Trait: type_of(AnyType),
    start: Int,
    end: Int,
    Prev: Variadic.TypesOfTrait[Trait],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = (
    Variadic.concat_types[Prev, Variadic.types[T=Trait, From[idx]]] if idx
    >= start
    and idx < end else Prev
)
"""A reducer that extracts elements within a specified index range.
Parameters:
    Trait: The trait that the types conform to.
    start: The starting index (inclusive).
    end: The ending index (exclusive).
    Prev: The accumulated result variadic so far.
    From: The input variadic sequence.
    idx: The current index being processed.
"""

comptime _TypePredicateGenerator[T: type_of(AnyType)] = __mlir_type[
    `!lit.generator<<"Type": `,
    T,
    `>`,
    Bool,
    `>`,
]
"""Generator type for type predicates.

A predicate takes a type and returns a boolean indicating whether to keep it.

Parameters:
    T: The trait that the types conform to.
"""

comptime _FilterReducer[
    Trait: type_of(AnyType),
    Predicate: _TypePredicateGenerator[Trait],
    Prev: Variadic.TypesOfTrait[Trait],
    From: Variadic.TypesOfTrait[Trait],
    idx: Int,
] = (
    Variadic.concat_types[
        Prev, Variadic.types[T=Trait, From[idx]]
    ] if Predicate[From[idx]] else Prev
)
"""A reducer that filters types based on a predicate function.

Parameters:
    Trait: The trait that the types conform to.
    Predicate: A generator that takes a type and returns Bool.
    Prev: The accumulated result variadic so far.
    From: The input variadic sequence.
    idx: The current index being processed.
"""
