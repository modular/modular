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
"""Implements the VariadicList, ParameterList and VariadicPack types.

These are Mojo built-ins, so you don't need to import them.
"""

from std.builtin.constrained import _constrained_conforms_to
from std.builtin.rebind import downcast
from std.format._utils import FormatStruct, TypeNames
from std.sys.intrinsics import _type_is_eq_parse_time
from std.builtin.globals import global_constant
from std.reflection.traits import AllWritable


struct _MLIR:
    comptime KGENTypeType = __mlir_type.`!kgen.type`

    comptime POPArrayType[
        size: __mlir_type.index, elt_type: AnyType
    ] = __mlir_type[`!pop.array<`, size, `, `, elt_type, `>`]

    comptime KGENParamListType[elt_type: Self.KGENTypeType] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]


# ===-----------------------------------------------------------------------===#
# ParameterList and TypeList Utilities
# ===-----------------------------------------------------------------------===#

comptime _IntToXGeneratorType[ToT: _MLIR.KGENTypeType] = __mlir_type[
    `!lit.generator<<"Idx":`,
    Int,
    `>`,
    +ToT,
    `>`,
]

# ===-----------------------------------------------------------------------===#
# TypeList
# ===-----------------------------------------------------------------------===#


struct TypeList[
    Trait: type_of(AnyType), //, values: _MLIR.KGENParamListType[Trait]
](Sized, TrivialRegisterPassable):
    """A compile-time list of types conforming to a common trait.

    `TypeList` provides type-level operations on variadic sequences of types,
    such as reversing, slicing, mapping, and membership testing.

    Parameters:
        Trait: The trait that all types in the list must conform to.
        values: The types in the list.

    Examples:

    ```mojo
    from std.builtin.variadics import TypeList
    from std.sys.intrinsics import _type_is_eq
    from std.testing import assert_equal

    # Create a type list
    comptime tl = TypeList[Trait=AnyType, Int, String, Float64]()

    def main():
        # Query size
        assert_equal(tl.size, 3)

        # Check membership
        comptime assert tl.contains[Int]
        comptime assert not tl.contains[Bool]

        # Index into the list
        comptime assert _type_is_eq[tl[0], Int]()
    ```
    """

    comptime _mlir_type = _MLIR.KGENParamListType[Self.Trait]
    """The low-level MLIR type of the type list."""

    comptime size: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            Self._mlir_type,
            ` `,
            +Self.values,
            `> : index`,
        ]
    )
    """The number of types in the list."""

    comptime __getitem_param__[idx: Int] = __mlir_attr[
        `#kgen.param_list.get<:`,
        Self._mlir_type,
        ` `,
        +Self.values,
        `, `,
        idx._mlir_value,
        `> : `,
        +Self.Trait,
    ]
    """Gets a type at the given index.

    Parameters:
        idx: The index of the type to access.
    """

    @implicit
    @always_inline("builtin")
    def __init__(
        existing: TypeList[...],
        out self: TypeList[
            Trait=Self.Trait,
            __mlir_attr[
                `#kgen.upcast<`,
                existing.values,
                `> : `,
                _MLIR.KGENParamListType[Self.Trait],
            ],
        ],
    ) where __mlir_attr[
        `#kgen.is_refined_type<`, existing.Trait, `, `, Self.Trait, `>`
    ]:
        """Upcasts a TypeList to a base trait.

        Args:
            existing: The TypeList to upcast from.

        Constraints:
            The existing.Trait is more refined than Self.Trait.
        """
        pass

    # ===-------------------------------------------------------------------===#
    # Constructors
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Constructs a TypeList."""
        pass

    comptime of[Trait: type_of(AnyType), //, *values: Trait] = TypeList[
        Trait=Trait, values.values
    ]
    """Form a compile-time list of types with some elements, uninstantiated.

    Parameters:
        Trait: The type of the elements in the list.
        values: The values in the list.

    Examples:
        ```mojo
        comptime Ts = TypeList.of[Trait=AnyType, Int, String, Float64, Bool]
        comptime Ms = TypeList.of[Trait=Movable, Int, String, Float64, Bool]
        ```
    """

    comptime _IndexToIntTypeTabulateWrap[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        ToWrap: _IntToXGeneratorType[Trait],
        idx: __mlir_type.index,
    ] = ToWrap[Int(mlir_value=idx)]

    comptime tabulate[
        Trait: type_of(AnyType),
        ToT: Trait,
        //,
        count: Int,
        Mapper: _IntToXGeneratorType[Trait],
    ] = TypeList[
        Trait=Trait,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count._int_mlir_index(),
            `,`,
            Self._IndexToIntTypeTabulateWrap[Trait=Trait, ToT=ToT, Mapper, ...],
            `> : `,
            _MLIR.KGENParamListType[Trait],
        ],
    ]
    """Builds a type list by applying an index-to-type mapper `count` times.

    Parameters:
        Trait: The trait of the generated TypeList.
        ToT: The type of the values in the generated TypeList.
        count: The number of times to apply the generator, the length of the result..
        Mapper: The generator to apply, mapping from Int to ToT.
    """

    comptime _SplatTypeTabulator[
        Trait: type_of(AnyType), T: Trait, index: Int
    ]: Trait = T

    comptime splat[
        Trait: type_of(AnyType), //, count: Int, type: Trait
    ] = TypeList.tabulate[count, Self._SplatTypeTabulator[Trait, type, _]]
    """Splats a type a given number of times.

    Parameters:
        Trait: The trait that the types conform to.
        count: The number of times to splat the type.
        type: The type to splat.
    """

    # Note: this is _concat instead of concat because it takes MLIR typelists
    comptime _concat[
        Trait: type_of(AnyType), //, *Ts: _MLIR.KGENParamListType[Trait]
    ] = TypeList[
        __mlir_attr[
            `#kgen.param_list.concat<`,
            Ts.values,
            `> :`,
            _MLIR.KGENParamListType[Trait],
        ]
    ]
    """Form a TypeList from the concatenation of multiple MLIR type lists.

    Parameters:
        Trait: The trait that types in the variadic sequences must conform to.
        Ts: The variadic sequences to concatenate.
    """

    # ===-------------------------------------------------------------------===#
    # Reductions
    # ===-------------------------------------------------------------------===#

    comptime _ReducerGeneratorType[
        FromAndTo: AnyType,
    ] = __mlir_type[
        `!lit.generator<<"Prev": `,
        +FromAndTo,
        `, "From": `,
        +Self.Trait,
        `>`,
        +FromAndTo,
        `>`,
    ]

    comptime _DiscardIndexWrapper[
        FromAndTo: AnyType,
        ToWrap: Self._ReducerGeneratorType[FromAndTo],
        PrevV: FromAndTo,
        VA: _MLIR.KGENParamListType[Self.Trait],
        idx: Int,
    ] = ToWrap[PrevV, TypeList[VA].__getitem_param__[idx]]
    """Adapts a (prev, element) reducer to the variadic reduce index signature."""

    comptime _ReduceVariadicIdxGeneratorTypeGenerator[
        Prev: AnyType
    ] = __mlir_type[
        `!lit.generator<<"Prev": `,
        +Prev,
        `, "From": `,
        _MLIR.KGENParamListType[Self.Trait],
        `, "Idx":`,
        SIMDSize,
        `>`,
        +Prev,
        `>`,
    ]
    comptime _IndexToIntWrap[
        ReduceT: AnyType,
        ToWrap: Self._ReduceVariadicIdxGeneratorTypeGenerator[ReduceT],
        PrevV: ReduceT,
        VA: _MLIR.KGENParamListType[Self.Trait],
        idx: __mlir_type.index,
    ] = ToWrap[PrevV, VA, Int(mlir_value=idx)]
    """Wrapper for type -> value."""

    comptime reduce[
        FromAndTo: AnyType,
        //,
        BaseVal: FromAndTo,
        Reducer: Self._ReducerGeneratorType[FromAndTo],
    ] = __mlir_attr[
        `#kgen.param_list.reduce<`,
        BaseVal,
        `,`,
        Self.values,
        `,`,
        Self._IndexToIntWrap[
            FromAndTo,
            Self._DiscardIndexWrapper[FromAndTo, Reducer, ...],
            ...,
        ],
        `> : `,
        +FromAndTo,
    ]
    """Folds this type list to a single value using an associative step function.

    Parameters:
        FromAndTo: The type of the accumulator and the final result.
        BaseVal: The initial accumulator value.
        Reducer: A compile-time generator
            `[prev: FromAndTo, element: Self.Trait] -> FromAndTo`.
    """

    comptime _TypeIndexReducerGenerator[Acc: AnyType] = __mlir_type[
        `!lit.generator<<"Prev": `,
        +Acc,
        `, "From": `,
        +Self.Trait,
        `, "Idx":`,
        Int,
        `>`,
        +Acc,
        `>`,
    ]

    comptime _PassIndexReducerWrapper[
        FromAndTo: AnyType,
        ToWrap: Self._TypeIndexReducerGenerator[FromAndTo],
        PrevV: FromAndTo,
        VA: _MLIR.KGENParamListType[Self.Trait],
        list_idx: Int,
    ] = ToWrap[
        PrevV,
        TypeList[VA].__getitem_param__[list_idx],
        list_idx,
    ]
    """Adapts a (prev, element, index) reducer to the variadic reduce index signature."""

    comptime reduce_idx[
        FromAndTo: AnyType,
        //,
        BaseVal: FromAndTo,
        Reducer: Self._TypeIndexReducerGenerator[FromAndTo],
    ] = __mlir_attr[
        `#kgen.param_list.reduce<`,
        BaseVal,
        `,`,
        Self.values,
        `,`,
        Self._IndexToIntWrap[
            FromAndTo,
            Self._PassIndexReducerWrapper[FromAndTo, Reducer, ...],
            ...,
        ],
        `> : `,
        +FromAndTo,
    ]
    """Folds this type list to a single value using a step function of position.

    Like `reduce`, but the reducer receives each element's index in this list,
    from `0` through `size - 1`, as a third compile-time argument.

    Parameters:
        FromAndTo: The type of the accumulator and the final result.
        BaseVal: The initial accumulator value.
        Reducer: A compile-time generator
            `[prev: FromAndTo, element: Self.Trait, idx: Int] -> FromAndTo`.
    """

    comptime _TypePredicateGenerator = __mlir_type[
        `!lit.generator<<"Type": `,
        Self.Trait,
        `>`,
        Bool,
        `>`,
    ]
    """Generator type for type predicates.

    A predicate takes a type and returns a boolean indicating whether to keep it.

    Parameters:
        T: The trait that the types conform to.
    """

    comptime _AnySatisfiesReducer[
        predicate: Self._TypePredicateGenerator,
        last_value: Bool,
        this_element: Self.Trait,
    ] = last_value or predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def any_satisfies[
        predicate: Self._TypePredicateGenerator,
    ]() -> Bool:
        """Returns true if `predicate` holds for at least one type in this list.

        Parameters:
            predicate: A compile-time generator `[T: Self.Trait] -> Bool`.

        Returns:
            True if `predicate` holds for at least one type in this list, False otherwise.
        """

        return Self.reduce[
            False,
            Self._AnySatisfiesReducer[predicate, ...],
        ]

    comptime _AllSatisfiesReducer[
        predicate: Self._TypePredicateGenerator,
        last_value: Bool,
        this_element: Self.Trait,
    ] = last_value and predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def all_satisfies[
        predicate: Self._TypePredicateGenerator,
    ]() -> Bool:
        """Returns true if `predicate` holds for every type in this list.

        For an empty list, returns true.

        Parameters:
            predicate: A compile-time generator `[T: Self.Trait] -> Bool`.

        Returns:
            True if `predicate` holds for every type in this list, False otherwise.
        """
        return Self.reduce[
            True,
            Self._AllSatisfiesReducer[predicate, ...],
        ]

    comptime _ContainsTypePredicate[
        search: Self.Trait,
        element: Self.Trait,
    ] = _type_is_eq_parse_time[element, search]()

    @always_inline("builtin")
    @staticmethod
    def contains[type: Self.Trait]() -> Bool:
        """Checks if a type is contained in this type list.

        Parameters:
            type: The type to check for.

        Returns:
            True if the type is contained in this type list, False otherwise.
        """
        return Self.any_satisfies[Self._ContainsTypePredicate[type, ...],]()

    # ===-------------------------------------------------------------------===#
    # Mappings
    # ===-------------------------------------------------------------------===#

    comptime _TypeToTypeGenerator[ToTrait: type_of(AnyType)] = __mlir_type[
        `!lit.generator<<"From":`, Self.Trait, `>`, ToTrait, `>`
    ]
    comptime _MapTabulator[
        ToTrait: type_of(AnyType),
        Mapper: Self._TypeToTypeGenerator[ToTrait],
        idx: Int,
    ]: ToTrait = Mapper[Self.__getitem_param__[idx]]

    comptime map[
        ToTrait: type_of(AnyType),
        //,
        Mapper: Self._TypeToTypeGenerator[ToTrait],
    ] = TypeList.tabulate[
        Trait=ToTrait,
        Self.size,
        Self._MapTabulator[ToTrait, Mapper, idx=_],
    ]
    """Maps types to new types using a mapper.

    Returns a new TypeList resulting from applying `Mapper[T]` to each element
    in this list.

    Parameters:
        ToTrait: The trait that the output types conform to.
        Mapper: A generator that maps a type to another type.
            The generator type is `[T: Trait] -> To`.
    """

    comptime _TypeIndexPredicateGenerator = __mlir_type[
        `!lit.generator<<"Elt": `,
        Self.Trait,
        `, "Idx":`,
        Int,
        `>`,
        Bool,
        `>`,
    ]
    """Generator type for indexed type predicates.

    A predicate takes an element type and its index in the list and returns whether
    to retain that element in the filtered list.

    Parameters:
        T: The trait that the types conform to.
    """

    comptime _FilterIdxTabulator[
        Predicate: Self._TypeIndexPredicateGenerator,
        idx: Int,
    ]: _MLIR.KGENParamListType[Self.Trait] = TypeList.of[
        Trait=Self.Trait, Self.__getitem_param__[idx]
    ]().values if Predicate[
        Self.__getitem_param__[idx], idx
    ] else TypeList.of[
        Trait=Self.Trait
    ]().values

    comptime filter_idx[
        Predicate: Self._TypeIndexPredicateGenerator,
    ] = TypeList._concat[
        *ParameterList.tabulate[
            Self.size,
            Self._FilterIdxTabulator[Predicate, _],
        ]()
    ]
    """Returns a new `TypeList` containing only elements selected by a predicate.

    The predicate is evaluated at compile time for each `(element, index)` pair.
    Indices are the positions in this list, from `0` through `size - 1`.

    Parameters:
        Predicate: A compile-time generator
            `[element: Self.Trait, idx: Int] -> Bool`. When it returns `True`,
            `element` is kept in order; when `False`, the element is dropped.

    Returns:
        A `TypeList` of the same trait containing the kept elements in order.
    """

    comptime _TypeToValueGeneratorType[ValueType: AnyType] = __mlir_type[
        `!lit.generator<<"Elt": `,
        +Self.Trait,
        `> `,
        +ValueType,
        `>`,
    ]
    """Maps a conforming element type to a compile-time value."""

    comptime _MapToValuesGeneratorType[
        ValueType: AnyType,
    ] = Self._TypeToValueGeneratorType[ValueType]

    comptime _MapToValuesIntTabulator[
        ValueType: AnyType,
        //,
        Mapper: Self._MapToValuesGeneratorType[ValueType=ValueType],
        idx: Int,
    ]: ValueType = Mapper[Self.__getitem_param__[idx]]

    comptime map_to_values[
        ValueType: AnyType,
        //,
        Mapper: Self._MapToValuesGeneratorType[ValueType=ValueType],
    ] = ParameterList.tabulate[
        type=ValueType,
        Self.size,
        Self._MapToValuesIntTabulator[
            ValueType=ValueType, Mapper=Mapper, idx=_
        ],
    ]
    """Convert each type in this list to a value, forming a ParameterList.

    This is the value analogue of `ParameterList.map_to_type`: each element
    type is passed to `Mapper`, and the resulting values share the homogeneous
    element type `ValueType`.

    Parameters:
        ValueType: The element type of the resulting `ParameterList`.
        Mapper: A compile-time generator that maps an element type to a value.
            The generator type is `[T: Self.Trait] -> ValueType`.
    """

    # ===-------------------------------------------------------------------===#
    # Other
    # ===-------------------------------------------------------------------===#

    comptime _ReverseTabulator[idx: Int]: Self.Trait = Self.__getitem_param__[
        Self.size - 1 - idx
    ]
    comptime reverse = TypeList.tabulate[Self.size, Self._ReverseTabulator[_]]
    """Returns this type list in reverse order."""

    comptime _SliceTabulator[
        start: Int,
        idx: Int,
    ]: Self.Trait = Self.__getitem_param__[start + idx]

    comptime slice[
        start: Int = 0,
        end: Int = Self.size,
    ] = TypeList.tabulate[
        max(end - start, 0),
        Self._SliceTabulator[start, _],
    ]
    """Extracts a contiguous subsequence from the type list.

    Returns a new variadic containing elements from index `start` (inclusive)
    to index `end` (exclusive). Similar to Python's slice notation [start:end].

    Parameters:
        start: The starting index (inclusive). Defaults to 0.
        end: The ending index (exclusive). Defaults to the list size.

    Constraints:
        0 <= start <= end <= size.
    """

    @always_inline
    def __len__(self) -> Int:
        """Gets the size of the TypeList.

        Returns:
            The number of elements on the TypeList.
        """
        return Self.size


# ===-----------------------------------------------------------------------===#
# ParameterList
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ParameterListIter[type: Copyable, //, *values: type](
    ImplicitlyCopyable, Iterable, Iterator, TrivialRegisterPassable
):
    """Const Iterator for ParameterList.

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

        if index >= Self.values.size:
            raise StopIteration()
        self.index = index + 1
        return Self.values[index]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = Self.values.size - self.index
        return (len, {len})


# TODO: Make this conform to Iterable when IteratorType can be conditionally
# defined only when 'type' is Copyable.
struct ParameterList[type: AnyType, //, values: _MLIR.KGENParamListType[type]](
    Sized, TrivialRegisterPassable, Writable
):
    """A utility class to access homogeneous variadic parameters.

    `ParameterList` is used by homogenous variadic parameter lists. Unlike
    `VariadicPack` (which is heterogeneous), `ParameterList` requires all
    elements to have the same type.

    `ParameterList` is only used for parameter lists, `VariadicList` is
    used for function arguments.

    For example, in the following function signature, `*args: Int` creates a
    `ParameterList` because it uses a single type `Int` instead of a variadic type
    parameter. The `*` before `args` indicates that `args` is a variadic argument,
    which means that the function can accept any number of arguments, but all
    arguments must have the same type `Int`.

    ```mojo
    def sum_values[*args: Int]() -> Int:
        var total = 0

        # Can use regular for loop because args is a ParameterList
        comptime for i in range(ParameterList[args].size):
            total += args[i]  # All elements are Int, so uniform access

        return total

    def main():
        print(sum_values[1, 2, 3, 4, 5]())
    ```

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.
    """

    comptime _mlir_type = _MLIR.KGENParamListType[Self.type]
    """The low-level MLIR type of the parameter list."""

    comptime size: Int = Int(
        mlir_value=__mlir_attr[
            `#kgen.param_list.size<:`,
            Self._mlir_type,
            ` `,
            +Self.values,
            `> : index`,
        ]
    )
    """The number of elements in the list."""

    # ===-------------------------------------------------------------------===#
    # Accessors
    # ===-------------------------------------------------------------------===#

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
            Self._mlir_type,
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
        `#kgen.param_list.get<:`,
        Self._mlir_type,
        ` `,
        +Self.values,
        `, `,
        idx._int_mlir_index(),
        `> : `,
        +Self.type,
    ]
    """Gets a single element on the variadic list."""

    # ===-------------------------------------------------------------------===#
    # Constructors
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """Constructs a ParameterList."""
        pass

    comptime empty_of[type: AnyType] = Self.of[type=type]
    """Form an empty compile-time list of values some element type.

    Parameters:
        type: The type of the elements in the list.

    Examples:
        ```mojo
        comptime Ints = ParameterList.empty_of[Int]()
        ```
    """

    comptime of[type: AnyType, //, *values: type] = ParameterList[
        type=type, values.values
    ]
    """Form a compile-time list of values with some elements, uninstantiated.

    Parameters:
        type: The type of the elements in the list.
        values: The values in the list.

    Examples:
        ```mojo
        comptime Ints = ParameterList.of[4, 5, 6]
        comptime Strings = ParameterList.of["foo", "bar", "baz"]
        ```
    """

    comptime _IndexToIntTabulateWrap[
        ToT: AnyType,
        //,
        ToWrap: _IntToXGeneratorType[ToT],
        idx: __mlir_type.index,
    ]: ToT = ToWrap[Int(mlir_value=idx)]

    comptime tabulate[
        type: AnyType,
        //,
        count: Int,
        Mapper: _IntToXGeneratorType[type],
    ] = ParameterList[
        type=type,
        __mlir_attr[
            `#kgen.param_list.tabulate<`,
            count._int_mlir_index(),
            `,`,
            Self._IndexToIntTabulateWrap[Mapper, ...],
            `> : `,
            _MLIR.KGENParamListType[type],
        ],
    ]
    """Builds a parameter list by applying an index-to-value mapper `count` times.

    Parameters:
        type: The element type of the resulting list.
        count: The length of the result; the mapper is invoked for each index in
            `0..<count`.
        Mapper: Compile-time generator mapping `Int` index to a value of `type`.
    """

    comptime _splat_tabulator[value: Some[AnyType], idx: Int] = value
    comptime splat[type: AnyType, //, count: Int, value: type] = Self.tabulate[
        count, Self._splat_tabulator[value, _]
    ]
    """Builds a homogeneous parameter list by repeating `value` `count` times.

    Parameters:
        type: The element type.
        count: The number of copies of `value` in the result.
        value: The value to repeat at every index.
    """

    comptime _concat[
        type: AnyType, //, *values: _MLIR.KGENParamListType[type]
    ] = __mlir_attr[
        `#kgen.param_list.concat<`,
        values.values,
        `> :`,
        _MLIR.KGENParamListType[type],
    ]
    """Represents the concatenation of multiple variadic sequences of values.

    Parameters:
        type: The types of the values in the variadic sequences.
        values: The variadic sequences to concatenate.
    """

    # ===-------------------------------------------------------------------===#
    # Reductions
    # ===-------------------------------------------------------------------===#

    comptime _ReducerGeneratorType[
        FromAndTo: AnyType,
    ] = __mlir_type[
        `!lit.generator<<"Prev": `,
        +FromAndTo,
        `, "From": `,
        +Self.type,
        `>`,
        +FromAndTo,
        `>`,
    ]

    comptime _DiscardIndexWrapper[
        FromAndTo: AnyType,
        ToWrap: Self._ReducerGeneratorType[FromAndTo],
        PrevV: FromAndTo,
        VA: Self._mlir_type,
        idx: __mlir_type.index,
    ] = ToWrap[PrevV, Self.__getitem_param__[Int(mlir_value=idx)]]
    """Takes an index because kgen.variadic.reduce passes it but we don't want it"""

    # TODO: This isn't returning a ParamList, so it should really be a 'def' so
    # we get parens on the caller side. However, that requires the type to be
    # materializable from parameter space to runtime.  We could split this into
    # reduce_param and reduce() where the later does materialization, or we
    # could just always use materialize?
    comptime reduce[
        FromAndTo: AnyType,
        //,
        BaseVal: FromAndTo,
        Reducer: Self._ReducerGeneratorType[FromAndTo],
    ] = __mlir_attr[
        `#kgen.param_list.reduce<`,
        BaseVal,
        `,`,
        Self.values,
        `,`,
        Self._DiscardIndexWrapper[FromAndTo, Reducer, ...],
        `> : `,
        +FromAndTo,
    ]
    """Form a value by applying a function that merges each element into a
    starting value, then return the result.

    Parameters:
        FromAndTo: The type of the input and output result.
        BaseVal: The initial value to reduce on.
        Reducer: A `[BaseVal: FromAndTo, T: Self.type] -> FromAndTo` that does the reduction.
    """

    comptime _ElementToBoolGeneratorType = __mlir_type[
        `!lit.generator<<"Elt": `, +Self.type, `>`, +Bool, `>`
    ]

    comptime _AnySatisfiesReducer[
        predicate: Self._ElementToBoolGeneratorType,
        last_value: Bool,
        this_element: Self.type,
    ]: Bool = last_value or predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def any_satisfies[predicate: Self._ElementToBoolGeneratorType]() -> Bool:
        """'any_satisfies' applies a function to each element and returns true if
        the function returns True for any element.

        Parameters:
            predicate: A `[elt: Self.Type] -> Bool` comptime expression to apply.

        Returns:
            True if the predicate returns True for any element, False otherwise.
        """
        return Self.reduce[
            False,
            Self._AnySatisfiesReducer[predicate, ...],
        ]

    comptime _AllSatisfiesReducer[
        predicate: Self._ElementToBoolGeneratorType,
        last_value: Bool,
        this_element: Self.type,
    ]: Bool = last_value and predicate[this_element]

    @always_inline("builtin")
    @staticmethod
    def all_satisfies[predicate: Self._ElementToBoolGeneratorType]() -> Bool:
        """'all_satisfies' applies a function to each element and returns true if
        the function returns True for all elements.

        Parameters:
            predicate: A `[elt: Self.Type] -> Bool` comptime expression to apply.

        Returns:
            True if the predicate returns True for all elements, False otherwise.
        """
        return Self.reduce[
            True,
            Self._AllSatisfiesReducer[predicate, ...],
        ]

    # FIXME(MOCO-3855): Add decl-where `where conforms_to(Self.type, Equatable)`
    comptime _ContainsValuePredicate[
        search_value: Self.type,
        element_value: Self.type,
    ] = trait_downcast[Equatable](search_value) == trait_downcast[Equatable](
        element_value
    )

    @always_inline("builtin")
    @staticmethod
    def contains[
        value: Self.type,
    ]() -> Bool where conforms_to(Self.type, Equatable):
        """
        Check if a value is contained in a variadic sequence of values.

        Parameters:
            value: The value to search for.

        Returns:
            True if the value is contained in the list, False otherwise.
        """
        return Self.any_satisfies[Self._ContainsValuePredicate[value, ...]]()

    # ===-------------------------------------------------------------------===#
    # Mappings
    # ===-------------------------------------------------------------------===#

    comptime _MapToTypeGeneratorType[Trait: type_of(AnyType)] = __mlir_type[
        `!lit.generator<<"Elt": `, +Self.type, `> `, Trait, `>`
    ]
    comptime _MapToTypeTabulator[
        Trait: type_of(AnyType),
        //,
        Mapper: Self._MapToTypeGeneratorType[Trait=Trait],
        idx: Int,
    ]: Trait = Mapper[Self.__getitem_param__[idx]]

    comptime map_to_type[
        Trait: type_of(AnyType),
        //,
        Mapper: Self._MapToTypeGeneratorType[Trait=Trait],
    ] = TypeList.tabulate[
        Trait=Trait,
        Self.size,
        Self._MapToTypeTabulator[Trait=Trait, Mapper=Mapper, idx=_],
    ]
    """Convert each element of this list into a type, forming a TypeList with
    the result.

    Parameters:
        Trait: The trait of the resulting TypeList.
        Mapper: A generator that maps an element of this list to a type.
    """

    # ===-------------------------------------------------------------------===#
    # Other
    # ===-------------------------------------------------------------------===#

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

        FormatStruct(writer, "ParameterList").params(
            TypeNames[Self.type](),
        ).fields[FieldsFn=write_fields]()

    # We can only support iteration when the elements are Copyable, because
    # iterators currently need to return the elements by value.
    @always_inline
    def __iter__(
        ref self,
    ) -> _ParameterListIter[
        *ParameterList[
            rebind[_MLIR.KGENParamListType[downcast[Self.type, Copyable]]](
                Self.values
            )
        ]()
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
    var src: Pointer[Self.variadic_list_type, Self.list_origin]

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
    ) raises StopIteration -> ref[self.src[][0]] Self.elt_type:
        var index = self.index
        if index == len(self.src[]):
            raise StopIteration()
        self.index = index + 1
        return self.src[][index]


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

    comptime _EltPointerType = Pointer[Self.element_type, Self.origin]
    # FIXME: This should be the origin of the container, not ExternalOrigin.
    var _value: Span[Self._EltPointerType, ExternalOrigin[mut=False]]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline
    @implicit
    def __init__[
        size: __mlir_type.index, container_origin: ImmutOrigin
    ](
        out self,
        value: Pointer[
            _MLIR.POPArrayType[size, Self._EltPointerType._mlir_type],
            container_origin,
        ]._mlir_type,
    ):
        """Constructs a VariadicList from a compiler-generated array of element
        pointers.

        Parameters:
            size: The number of elements in the variadic list.
            container_origin: The origin of the container.

        Args:
            value: The raw reference to the array of element pointers.
        """
        # Convert the !lit.ref to an UnsafePointer, then cast to a pointer to
        # the first element.
        var array_up = UnsafePointer(
            to=Pointer(_mlir_value=value)[]
        ).unsafe_origin_cast[ExternalOrigin[mut=False]]()
        var elt_ptr = UnsafePointer[_, ExternalOrigin[mut=False]](
            __mlir_op.`pop.array.gep`(
                array_up.address,
                Int(0)._int_mlir_index(),
            )
        ).bitcast[Self._EltPointerType]()
        self._value = Span(ptr=elt_ptr, length=Int(mlir_value=size))

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
        return len(self._value)

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
        return self._value.unsafe_ptr()[idx][]

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
    element_trait: type_of(AnyType),
    //,
    is_owned: Bool,
    *element_types: element_trait,
](Copyable, RegisterPassable, Sized):
    """A utility class to access heterogeneous variadic function arguments.

    `VariadicPack` is used when you need to accept variadic arguments where each
    argument can have a different type, but all types conform to a common trait.
    Unlike `ParameterList` (which is homogeneous), `VariadicPack` allows each
    element to have a different concrete type.

    `VariadicPack` is essentially a heterogeneous tuple that gets lowered to a
    struct at runtime. Because `VariadicPack` is a heterogeneous tuple (not an
    array), each element can have a different size and memory layout, which
    means the compiler needs to know the exact type of each element at compile
    time to generate the correct memory layout and access code.

    Therefore, indexing into `VariadicPack` requires compile-time indices using
    `comptime for` loops, whereas indexing into `ParameterList` uses runtime
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

    def main():
        print(count_many_things(Int8(5), UInt32(11), Int(12)))  # Prints: 28
    ```

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument pack.
        origin: The origin of the underlying elements.
        element_trait: The trait that each element of the pack conforms to.
        is_owned: Whether the elements are owned by the pack. If so, the pack
                  will release the elements when it is destroyed.
        element_types: The list of types held by the argument pack.
    """

    comptime _mlir_type = __mlir_type[
        `!lit.ref.pack<:param_list<`,
        Self.element_trait,
        `> `,
        Self.element_types.values,
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
                comptime element_type = Self.element_types[i]
                _constrained_conforms_to[
                    conforms_to(element_type, ImplicitlyDestructible),
                    Parent=Self,
                    Element=element_type,
                    ParentConformsTo="ImplicitlyDestructible",
                ]()

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
        return Self.element_types.size

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
            index=index._int_mlir_index()
        ](self._value)
        return __get_litref_as_mvalue(litref_elt)

    # ===-------------------------------------------------------------------===#
    # C Pack Utilities
    # ===-------------------------------------------------------------------===#

    # FIXME: bound by AnyType
    comptime _kgen_element_types = rebind[
        _MLIR.KGENParamListType[__mlir_type.`!kgen.type`]
    ](Self.element_types.values)
    """This is the element_types list lowered to `variadic<type>` type for kgen.
    """

    # FIXME: bound by AnyType
    comptime _variadic_pointer_types = __mlir_attr[
        `#kgen.param.expr<variadic_ptr_map, `,
        Self._kgen_element_types,
        `, 0: index>: `,
        _MLIR.KGENParamListType[__mlir_type.`!kgen.type`],
    ]
    """Use variadic_ptr_map to construct the type list of the !kgen.struct that
    the !lit.ref.pack will lower to.  It exposes the pointers introduced by the
    references.
    """
    comptime _kgen_pack_with_pointer_type = __mlir_type[
        `!kgen.struct<:param_list<type> `,
        Self._variadic_pointer_types,
        ` isParamPack>`,
    ]
    """This is the !kgen.struct type with pointer elements."""

    @doc_hidden
    @always_inline("nodebug")
    def get_as_kgen_pack(self) -> Self._kgen_pack_with_pointer_type:
        """This rebinds `in_pack` to the equivalent `!kgen.struct` with kgen
        pointers."""
        return rebind[Self._kgen_pack_with_pointer_type](self._value)

    # FIXME: bound by AnyType
    comptime _variadic_with_pointers_removed = __mlir_attr[
        `#kgen.param.expr<variadic_ptrremove_map, `,
        Self._variadic_pointer_types,
        `>: `,
        _MLIR.KGENParamListType[__mlir_type.`!kgen.type`],
    ]
    comptime _loaded_kgen_pack_type = __mlir_type[
        `!kgen.struct<:param_list<type> `,
        Self._variadic_with_pointers_removed,
        ` isParamPack>`,
    ]
    """This is the `!kgen.struct` type that happens if one loads all the elements
    of the pack.
    """

    # Returns all the elements in a kgen.pack.
    # Useful for FFI, such as calling printf. Otherwise, avoid this if possible.
    @doc_hidden
    @always_inline("nodebug")
    def get_loaded_kgen_pack(self) -> Self._loaded_kgen_pack_type:
        """This returns the stored KGEN pack after loading all of the elements.
        """
        return __mlir_op.`kgen.struct.load_indirect`(self.get_as_kgen_pack())

    def _write_to[
        O1: ImmutOrigin,
        O2: ImmutOrigin,
        O3: ImmutOrigin,
        *,
        is_repr: Bool = False,
    ](
        self,
        mut writer: Some[Writer],
        start: StringSlice[O1] = StaticString(""),
        end: StringSlice[O2] = StaticString(""),
        sep: StringSlice[O3] = StaticString(", "),
    ) where AllWritable[*Self.element_types]:
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
                trait_downcast[Writable](self[i]).write_repr_to(writer)
            else:
                trait_downcast[Writable](self[i]).write_to(writer)
        writer.write_string(end)

    @no_inline
    def write_to(
        self,
        mut writer: Some[Writer],
    ) where AllWritable[*Self.element_types]:
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
        self,
        mut writer: Some[Writer],
    ) where AllWritable[*Self.element_types]:
        """Writes the repr of the elements of this pack to a writer.

        Args:
            writer: The object to write to.
        """
        self._write_to[is_repr=True](
            writer,
            start=StaticString("("),
            end=StaticString(")"),
        )
