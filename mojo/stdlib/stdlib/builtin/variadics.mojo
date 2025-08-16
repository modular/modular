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
"""Implements the VariadicList and VariadicPack types.

These are Mojo built-ins, so you don't need to import them.
"""

from memory import Pointer

alias Variadic[type: AnyType] = __mlir_type[`!kgen.variadic<`, type, `>`]
"""Represents a raw variadic sequence of values of the specified type."""

alias VariadicOf[T: _AnyTypeMetaType] = __mlir_type[`!kgen.variadic<`, T, `>`]
"""Represents a raw variadic sequence of types that satisfy the specified trait."""


@always_inline("nodebug")
fn variadic_size[T: AnyType](seq: Variadic[T]) -> Int:
    """Returns the length of a variadic sequence.

    Parameters:
        T: The type of values in the sequence.

    Returns:
        The length of the variadic sequence.
    """
    return __mlir_op.`pop.variadic.size`(seq)


@always_inline("nodebug")
fn variadic_size[T: _AnyTypeMetaType](seq: VariadicOf[T]) -> Int:
    """Returns the length of a variadic sequence.

    Parameters:
        T: The trait that types in the sequence must conform to.

    Returns:
        The length of the variadic sequence.
    """
    return __mlir_op.`pop.variadic.size`(seq)


# ===-----------------------------------------------------------------------===#
# VariadicList / VariadicListMem
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _VariadicListIter[type: AnyTrivialRegType](Copyable, Iterator, Movable):
    """Const Iterator for VariadicList.

    Parameters:
        type: The type of the elements in the list.
    """

    alias Element = type

    var index: Int
    var src: VariadicList[type]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.index < len(self.src)

    fn __next__(mut self) -> type:
        self.index += 1
        return self.src[self.index - 1]


@register_passable("trivial")
struct VariadicList[type: AnyTrivialRegType](Sized):
    """A utility class to access homogeneous variadic function arguments.

    `VariadicList` is used when you need to accept variadic arguments where all
    arguments have the same type. Unlike `VariadicPack` (which is heterogeneous),
    `VariadicList` requires all elements to have the same concrete type.

    At runtime, `VariadicList` is treated as a homogeneous array. Because all
    the elements have the same type, each element has the same size and memory
    layout, so the compiler can generate code that works to access any index
    at runtime.

    Therefore, indexing into `VariadicList` can use runtime indices with regular
    `for` loops, whereas indexing into `VariadicPack` requires compile-time
    indices using `@parameter for` loops.

    For example, in the following function signature, `*args: Int` creates a
    `VariadicList` because it uses a single type `Int` instead of a variadic type
    parameter. The `*` before `args` indicates that `args` is a variadic argument,
    which means that the function can accept any number of arguments, but all
    arguments must have the same type `Int`.

    ```mojo
    fn sum_values(*args: Int) -> Int:
        var total = 0

        # Can use regular for loop because args is a VariadicList
        for i in range(len(args)):
            total += args[i]  # All elements are Int, so uniform access

        return total

    def main():
        print(sum_values(1, 2, 3, 4, 5))
    ```

    Parameters:
        type: The type of the elements in the list.
    """

    alias _mlir_type = Variadic[type]

    var value: Self._mlir_type
    """The underlying storage for the variadic list."""

    alias IterType = _VariadicListIter[type]

    @always_inline
    @implicit
    fn __init__(out self, *value: type):
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            value: The variadic argument list to construct the variadic list
              with.
        """
        self = value

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Parameters:
            I: A type that can be used as an index.

        Returns:
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index(idx))

    @always_inline
    fn __iter__(self) -> Self.IterType:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return Self.IterType(0, self)


@fieldwise_init
struct _VariadicListMemIter[
    elt_is_mutable: Bool, //,
    elt_type: AnyType,
    elt_origin: Origin[elt_is_mutable],
    list_origin: ImmutableOrigin,
    is_owned: Bool,
]:
    """Iterator for VariadicListMem.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_origin: The origin of the elements.
        list_origin: The origin of the VariadicListMem.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    alias variadic_list_type = VariadicListMem[
        elt_type, elt_origin._mlir_origin, is_owned
    ]

    alias Element = elt_type

    var index: Int
    var src: Pointer[
        Self.variadic_list_type,
        list_origin,
    ]

    fn __init__(
        out self, index: Int, ref [list_origin]list: Self.variadic_list_type
    ):
        self.index = index
        self.src = Pointer(to=list)

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.index < len(self.src[])

    fn __next_ref__(mut self) -> ref [elt_origin] elt_type:
        self.index += 1
        return rebind[Self.variadic_list_type.reference_type](
            Pointer(to=self.src[][self.index - 1])
        )[]


struct VariadicListMem[
    elt_is_mutable: Bool, //,
    element_type: AnyType,
    origin: Origin[elt_is_mutable],
    is_owned: Bool,
](Sized):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes references to the elements in a
    way that can be enumerated.  Each element may be accessed with `elt[]`.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument.
        element_type: The type of the elements in the list.
        origin: The origin of the underlying elements.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    alias reference_type = Pointer[element_type, origin]
    alias _mlir_type = Variadic[Self.reference_type._mlir_type]

    var value: Self._mlir_type
    """The underlying storage, a variadic list of references to elements of the
    given type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # Provide support for read-only variadic arguments.
    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value

    @always_inline
    fn __moveinit__(out self, deinit existing: Self):
        """Moves constructor.

        Args:
          existing: The existing VariadicListMem.
        """
        self.value = existing.value

    @always_inline
    fn __del__(deinit self):
        """Destructor that releases elements if owned."""

        # Destroy each element if this variadic has owned elements, destroy
        # them.  We destroy in backwards order to match how arguments are
        # normally torn down when CheckLifetimes is left to its own devices.
        @parameter
        if is_owned:
            for i in reversed(range(len(self))):
                UnsafePointer(to=self[i]).destroy_pointee()

    fn consume_elements[
        elt_handler: fn (idx: Int, var elt: element_type) capturing
    ](deinit self):
        """Consume the variadic list by transfering ownership of each element
        into the provided closure one at a time.  This is only valid on 'owned'
        variadic lists.

        Parameters:
            elt_handler: A function that will be called for each element of the
                         list.
        """

        constrained[
            is_owned,
            "consume_elements may only be called on owned variadic lists",
        ]()

        for i in range(len(self)):
            var ptr = UnsafePointer(to=self[i])
            # TODO: Cannot use UnsafePointer.take_pointee because it requires
            # the element to be Movable, which is not required here.
            elt_handler(i, __get_address_as_owned_value(ptr.address))

    # FIXME: This is a hack to work around a miscompile, do not use.
    fn _anihilate(deinit self):
        pass

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return __mlir_op.`pop.variadic.size`(self.value)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(
        self, idx: Int
    ) -> ref [
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        Origin[elt_is_mutable].cast_from[__origin_of(origin, self)]
    ] element_type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __get_litref_as_mvalue(
            __mlir_op.`pop.variadic.get`(self.value, idx.value)
        )

    fn __iter__(
        self,
        out result: _VariadicListMemIter[
            element_type, origin, __origin_of(self), is_owned
        ],
    ):
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return __type_of(result)(0, self)


# ===-----------------------------------------------------------------------===#
# VariadicPack
# ===-----------------------------------------------------------------------===#


alias _AnyTypeMetaType = __type_of(AnyType)


@register_passable
struct VariadicPack[
    elt_is_mutable: Bool, //,
    is_owned: Bool,
    origin: Origin[elt_is_mutable],
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
](Sized):
    """A utility class to access heterogeneous variadic function arguments.

    `VariadicPack` is used when you need to accept variadic arguments where each
    argument can have a different type, but all types conform to a common trait.
    Unlike `VariadicList` (which is homogeneous), `VariadicPack` allows each
    element to have a different concrete type.

    `VariadicPack` is essentially a heterogeneous tuple that gets lowered to a
    struct at runtime. Because `VariadicPack` is a heterogeneous tuple (not an
    array), each element can have a different size and memory layout, which
    means the compiler needs to know the exact type of each element at compile
    time to generate the correct memory layout and access code.

    Therefore, indexing into `VariadicPack` requires compile-time indices using
    `@parameter for` loops, whereas indexing into `VariadicList` uses runtime
    indices.

    For example, in the following function signature, `*args: *ArgTypes` creates a
    `VariadicPack` because it uses a variadic type parameter `*ArgTypes` instead
    of a single type. The `*` before `ArgTypes` indicates that `ArgTypes` is a
    variadic type parameter, which means that the function can accept any number
    of arguments, and each argument can have a different type. This allows each
    argument to have a different type while all types must conform to the
    `Intable` trait.

    ```mojo
    fn count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
        var total = 0

        # Must use @parameter for loop because args is a VariadicPack
        @parameter
        for i in range(args.__len__()):
            # Each args[i] has a different concrete type from *ArgTypes
            # The compiler generates specific code for each iteration
            total += Int(args[i])

        return total

    def main():
        print(count_many_things(5, 11.7, 12))  # Prints: 28
    ```

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument pack.
        is_owned: Whether the elements are owned by the pack. If so, the pack
                  will release the elements when it is destroyed.
        origin: The origin of the underlying elements.
        element_trait: The trait that each element of the pack conforms to.
        element_types: The list of types held by the argument pack.
    """

    alias _mlir_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        origin._mlir_origin,
        `>`,
    ]

    var _value: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_private
    @always_inline("nodebug")
    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_disable_nested_origin_exclusivity
    fn __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicPack from the internal representation.

        Args:
            value: The argument to construct the pack with.
        """
        self._value = value

    @always_inline("nodebug")
    fn __del__(deinit self):
        """Destructor that releases elements if owned."""

        @parameter
        if is_owned:

            @parameter
            for i in reversed(range(Self.__len__())):
                UnsafePointer(to=self[i]).destroy_pointee()

    fn consume_elements[
        elt_handler: fn[idx: Int] (var elt: element_types[idx]) capturing
    ](deinit self):
        """Consume the variadic pack by transfering ownership of each element
        into the provided closure one at a time.  This is only valid on 'owned'
        variadic packs.

        Parameters:
            elt_handler: A function that will be called for each element of the
                         pack.
        """

        constrained[
            is_owned,
            "consume_elements may only be called on owned variadic packs",
        ]()

        @parameter
        for i in range(Self.__len__()):
            var ptr = UnsafePointer(to=self[i])
            # TODO: Cannot use UnsafePointer.take_pointee because it requires
            # the element to be Movable, which is not required here.
            elt_handler[i](__get_address_as_owned_value(ptr.address))

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    @staticmethod
    fn __len__() -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """

        alias result = variadic_size(element_types)
        return result

    @always_inline
    fn __len__(self) -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """
        return Self.__len__()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[
        index: Int
    ](self) -> ref [Self.origin] element_types[index.value]:
        """Return a reference to an element of the pack.

        Parameters:
            index: The element of the pack to return.

        Returns:
            A reference to the element.  The Pointer's mutability follows the
            mutability of the pack argument convention.
        """
        litref_elt = __mlir_op.`lit.ref.pack.extract`[index = index.value](
            self._value
        )
        return __get_litref_as_mvalue(litref_elt)

    # ===-------------------------------------------------------------------===#
    # C Pack Utilities
    # ===-------------------------------------------------------------------===#

    alias _kgen_element_types = rebind[Variadic[AnyTrivialRegType]](
        Self.element_types
    )
    """This is the element_types list lowered to `variadic<type>` type for kgen.
    """
    alias _variadic_pointer_types = __mlir_attr[
        `#kgen.param.expr<variadic_ptr_map, `,
        Self._kgen_element_types,
        `, 0: index>: `,
        Variadic[AnyTrivialRegType],
    ]
    """Use variadic_ptr_map to construct the type list of the !kgen.pack that
    the !lit.ref.pack will lower to.  It exposes the pointers introduced by the
    references.
    """
    alias _kgen_pack_with_pointer_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_pointer_types, `>`
    ]
    """This is the !kgen.pack type with pointer elements."""

    @doc_private
    @always_inline("nodebug")
    fn get_as_kgen_pack(self) -> Self._kgen_pack_with_pointer_type:
        """This rebinds `in_pack` to the equivalent `!kgen.pack` with kgen
        pointers."""
        return rebind[Self._kgen_pack_with_pointer_type](self._value)

    alias _variadic_with_pointers_removed = __mlir_attr[
        `#kgen.param.expr<variadic_ptrremove_map, `,
        Self._variadic_pointer_types,
        `>: `,
        Variadic[AnyTrivialRegType],
    ]
    alias _loaded_kgen_pack_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_with_pointers_removed, `>`
    ]
    """This is the `!kgen.pack` type that happens if one loads all the elements
    of the pack.
    """

    # Returns all the elements in a kgen.pack.
    # Useful for FFI, such as calling printf. Otherwise, avoid this if possible.
    @doc_private
    @always_inline("nodebug")
    fn get_loaded_kgen_pack(self) -> Self._loaded_kgen_pack_type:
        """This returns the stored KGEN pack after loading all of the elements.
        """
        return __mlir_op.`kgen.pack.load`(self.get_as_kgen_pack())
