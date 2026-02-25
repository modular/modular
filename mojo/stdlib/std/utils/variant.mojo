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
"""Defines a Variant type."""

from builtin.constrained import _constrained_conforms_to
from builtin.rebind import downcast
from builtin.variadics import Variadic
from format._utils import (
    FormatStruct,
    TypeNames,
    constrained_conforms_to_writable,
)
from os import abort
from sys.intrinsics import _type_is_eq

# ===----------------------------------------------------------------------=== #
# Variant
# ===----------------------------------------------------------------------=== #


struct Variant[*Ts: AnyType](ImplicitlyCopyable, Writable):
    """A union that can hold a runtime-variant value from a set of predefined
    types.

    `Variant` is a discriminated union type, similar to `std::variant` in C++
    or `enum` in Rust. It can store exactly one value that can be any of the
    specified types, determined at runtime.

    The key feature is that the actual type stored in a `Variant` is determined
    at runtime, not compile time. This allows you to change what type a variant
    holds during program execution. Memory-wise, a variant only uses the space
    needed for the largest possible type plus a small discriminant field to
    track which type is currently active.

    Tips:

    - use `isa[T]()` to check what type a variant is
    - use `unsafe_take[T]()` to take a value from the variant
    - use `[T]` to get a value out of a variant
        - This currently does an extra copy/move until we have origins
        - It also temporarily requires the value to be mutable
    - use `set[T](var new_value: T)` to reset the variant to a new value
    - use `is_type_supported[T]` to check if the variant permits the type `T`

    **Note**: Currently, variant operations require the variant to be
    mutable (`mut`), even for read operations.

    Example:

    ```mojo
    from utils import Variant
    import random

    comptime IntOrString = Variant[Int, String]

    fn to_string(mut x: IntOrString) -> String:
        if x.isa[String]():
            return x[String]
        return String(x[Int])

    var an_int = IntOrString(4)
    var a_string = IntOrString("I'm a string!")
    var who_knows = IntOrString(0)
    # Randomly change who_knows to a string
    random.seed()
    if random.random_ui64(0, 1):
        who_knows.set[String]("I'm also a string!")

    print(a_string[String])      # => I'm a string!
    print(an_int[Int])           # => 4
    print(to_string(who_knows))  # Either 0 or "I'm also a string!"

    if who_knows.isa[String]():
        print("It's a String!")
    ```

    Example usage for error handling:

    ```mojo
    comptime Result = Variant[String, Error]

    fn process_data(data: String) -> Result:
        if len(data) == 0:
            return Result(Error("Empty data"))
        return Result(String("Processed: ", data))

    var result = process_data("Hello")
    if result.isa[String]():
        print("Success:", result[String])
    else:
        print("Error:", result[Error])
    ```

    Example usage in a `List` to create a heterogeneous list:

    ```mojo
    comptime MixedType = Variant[Int, Float64, String, Bool]

    var mixed_list = List[MixedType]()
    mixed_list.append(MixedType(42))
    mixed_list.append(MixedType(3.14))
    mixed_list.append(MixedType("hello"))
    mixed_list.append(MixedType(True))

    for item in mixed_list:
        if item.isa[String]():
            print("String:", item[String])
        elif item.isa[Int]():
            print("Integer:", item[Int])
        elif item.isa[Float64]():
            print("Float:", item[Float64])
        elif item.isa[Bool]():
            print("Boolean:", item[Bool])
    ```

    ## Layout

    The layout of `Variant` is not guaranteed and may change at any time. The
    implementation may apply niche optimizations (for example, encoding the
    discriminant inside spare bits of one of the types in `Ts`) that alter the
    resulting layout. Do not rely on `size_of[Variant[...]]()` or
    `align_of[Variant[...]]()` being stable across language versions. The only
    guarantee is that the size and alignment will be at least as large as those
    of the largest type among `Ts`.

    If you need to inspect the current size or alignment, use `size_of` and
    `align_of`, but treat the results as non-stable implementation details.

    Parameters:
        Ts: The possible types that this variant can hold. All types must
            implement `Copyable`.
    """

    comptime __del__is_trivial = _all_trivial_del[*Self.Ts]()
    comptime __copy_ctor_is_trivial = _all_trivial_copyinit[*Self.Ts]()
    comptime __move_ctor_is_trivial = _all_trivial_moveinit[*Self.Ts]()

    # Fields
    comptime _sentinel: Int = -1
    comptime _mlir_type = __mlir_type[
        `!kgen.variant<[rebind(:`, type_of(Self.Ts), ` `, Self.Ts, `)]>`
    ]
    var _impl: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, *, unsafe_uninitialized: ()):
        """Unsafely create an uninitialized Variant.

        Args:
            unsafe_uninitialized: Marker argument indicating this initializer is unsafe.
        """
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @implicit
    fn __init__[T: Movable](out self, var value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        comptime idx = Self._check[T]()
        self._get_discr() = UInt8(idx)
        self._get_ptr[T]().init_pointee_move(value^)

    fn __init__(out self, *, copy: Self):
        """Creates a deep copy of an existing variant.

        Args:
            copy: The variant to copy from.
        """

        self = Self(unsafe_uninitialized=())
        self._get_discr() = copy._get_discr()

        comptime for i in range(Variadic.size(Self.Ts)):
            comptime TUnknown = Self.Ts[i]
            _constrained_conforms_to[
                conforms_to(TUnknown, Copyable),
                Parent=Self,
                Element=TUnknown,
                ParentConformsTo="Copyable",
            ]()
            comptime T = downcast[TUnknown, Copyable]

            if self._get_discr() == UInt8(i):
                self._get_ptr[T]().init_pointee_copy(copy._get_ptr[T]()[])
                return

    fn __init__(out self, *, deinit take: Self):
        """Move initializer for the variant.

        Args:
            take: The variant to move.
        """
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        self._get_discr() = take._get_discr()

        comptime for i in range(Variadic.size(Self.Ts)):
            comptime TUnknown = Self.Ts[i]
            _constrained_conforms_to[
                conforms_to(TUnknown, Movable),
                Parent=Self,
                Element=TUnknown,
                ParentConformsTo="Movable",
            ]()
            comptime T = downcast[TUnknown, Movable]

            if self._get_discr() == UInt8(i):
                # Calls the correct move constructor
                self._get_ptr[T]().init_pointee_move_from(take._get_ptr[T]())
                return

    fn __del__(deinit self):
        """Destroy the variant."""

        comptime for i in range(Variadic.size(Self.Ts)):
            comptime TUnknown = Self.Ts[i]
            _constrained_conforms_to[
                conforms_to(TUnknown, ImplicitlyDestructible),
                Parent=Self,
                Element=TUnknown,
                ParentConformsTo="ImplicitlyDestructible",
            ]()
            comptime T = downcast[TUnknown, ImplicitlyDestructible]

            if self._get_discr() == UInt8(i):
                self._get_ptr[T]().destroy_pointee()
                return

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__[T: AnyType](ref self) -> ref[self] T:
        """Get the value out of the variant as a type-checked type.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            A reference to the internal data.
        """
        if not self.isa[T]():
            abort("get: wrong variant type")

        return self.unsafe_get[T]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _write_value_to[*, is_repr: Bool](self, mut writer: Some[Writer]):
        constrained_conforms_to_writable[*Self.Ts, Parent=Self]()

        comptime for i in range(Variadic.size(Self.Ts)):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                ref value = trait_downcast[Writable](self.unsafe_get[T]())

                comptime if is_repr:
                    value.write_repr_to(writer)
                else:
                    value.write_to(writer)

                return

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes the currently held variant value to the provided Writer.

        Args:
            writer: The object to write to.

        Constraints:
            All types in `Ts` must conform to Writable.
        """
        self._write_value_to[is_repr=False](writer)

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the Variant.

        Args:
            writer: The object to write to.

        Constraints:
            All types in `Ts` must conform to Writable.
        """

        @parameter
        fn write_field(mut w: Some[Writer]):
            self._write_value_to[is_repr=True](w)

        FormatStruct(writer, "Variant").params(TypeNames[*Self.Ts]()).fields[
            FieldsFn=write_field
        ]()

    @always_inline("nodebug")
    fn _get_ptr[T: AnyType](ref[_] self) -> UnsafePointer[T, origin_of(self)]:
        comptime idx = Self._check[T]()
        comptime assert idx != Self._sentinel, "not a union element type"
        var ptr = UnsafePointer(to=self._impl).address
        var discr_ptr = __mlir_op.`pop.variant.bitcast`[
            _type = UnsafePointer[T, origin_of(self)]._mlir_type,
            index = idx._mlir_value,
        ](ptr)
        return discr_ptr

    @always_inline("nodebug")
    fn _get_discr(ref self) -> ref[self] UInt8:
        var ptr = UnsafePointer(to=self._impl).address
        var discr_ptr = __mlir_op.`pop.variant.discr_gep`[
            _type = __mlir_type.`!kgen.pointer<scalar<ui8>>`
        ](ptr)
        return UnsafePointer[mut=True](discr_ptr).bitcast[UInt8]()[]

    @always_inline
    fn take[T: Movable](deinit self) -> T:
        """Take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[T]():
            abort("taking the wrong type!")

        return self.unsafe_take[T]()

    @always_inline
    fn unsafe_take[T: Movable](mut self) -> T:
        """Unsafely take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        debug_assert(self.isa[T](), "taking wrong type")
        # don't call the variant's deleter later
        self._get_discr() = UInt8(Self._sentinel)
        return self._get_ptr[T]().take_pointee()

    @always_inline
    fn replace[
        Tin: Movable & ImplicitlyDestructible,
        Tout: Movable,
    ](mut self, var value: Tin) -> Tout:
        """Replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[Tout]():
            abort("taking out the wrong type!")

        return self.unsafe_replace[Tin, Tout](value^)

    @always_inline
    fn unsafe_replace[
        Tin: Movable, Tout: Movable
    ](mut self, var value: Tin) -> Tout:
        """Unsafely replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        debug_assert(self.isa[Tout](), "taking out the wrong type!")

        var x = self.unsafe_take[Tout]()
        self.set[Tin](value^)
        return x^

    fn set[T: Movable](mut self, var value: T):
        """Set the variant value.

        This will call the destructor on the old value, and update the variant's
        internal type and data to the new value.

        Parameters:
            T: The new variant type. Must be one of the Variant's type arguments.

        Args:
            value: The new value to set the variant to.
        """
        self = Self(value^)

    fn isa[T: AnyType](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        comptime idx = Self._check[T]()
        return self._get_discr() == UInt8(idx)

    fn unsafe_get[T: AnyType](ref self) -> ref[self] T:
        """Get the value out of the variant as a type-checked type.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Pointer[T]`.
        """
        debug_assert(self.isa[T](), "get: wrong variant type")
        return self._get_ptr[T]()[]

    @staticmethod
    fn _check[T: AnyType]() -> Int:
        comptime for i in range(Variadic.size(Self.Ts)):
            if _type_is_eq[Self.Ts[i], T]():
                return i
        return Self._sentinel

    @staticmethod
    fn is_type_supported[T: AnyType]() -> Bool:
        """Check if a type can be used by the `Variant`.

        Parameters:
            T: The type of the value to check support for.

        Returns:
            `True` if type `T` is supported by the `Variant`.

        Example:

        ```mojo
        from utils import Variant

        def takes_variant(mut arg: Variant):
            if arg.is_type_supported[Float64]():
                arg = Float64(1.5)

        def main():
            var x = Variant[Int, Float64](1)
            takes_variant(x)
            if x.isa[Float64]():
                print(x[Float64]) # 1.5
        ```

        For example, the `Variant[Int, Bool]` permits `Int` and `Bool`.
        """
        return Self._check[T]() != Self._sentinel

    # TODO(MOCO-2367): Use a `unified` closure parameter here instead.
    fn destroy_with[T: AnyType](deinit self, destroy_func: fn(var T)):
        """Destroy a value contained in this Variant in-place using a caller
        provided destructor function.

        This method can be used to destroy linear types in a `Variant` in-place,
        without requiring that they be `Movable`.

        This method will abort if this variant does not current contain an
        element of the specified type `T`.

        Parameters:
            T: The element type the variant is expected to currently contain,
                and which will be destroyed by `destroy_func`.

        Args:
            destroy_func: Caller-provided destructor function for destroying
                an instance of `T`.
        """
        if not self.isa[T]():
            abort("Variant.destroy_with: wrong variant type")

        var ptr = self._get_ptr[T]()

        ptr.destroy_pointee_with(destroy_func)


# ===-------------------------------------------------------------------===#
# Helper functions
# ===-------------------------------------------------------------------===#


fn _all_trivial_del[*Ts: AnyType]() -> Bool:
    comptime for i in range(Variadic.size(Ts)):
        comptime if conforms_to(Ts[i], ImplicitlyDestructible):
            if not downcast[Ts[i], ImplicitlyDestructible].__del__is_trivial:
                return False
        else:
            return False
    return True


fn _all_trivial_copyinit[*Ts: AnyType]() -> Bool:
    comptime for i in range(Variadic.size(Ts)):
        comptime if conforms_to(Ts[i], Copyable):
            if not downcast[Ts[i], Copyable].__copy_ctor_is_trivial:
                return False
        else:
            return False

    return True


fn _all_trivial_moveinit[*Ts: AnyType]() -> Bool:
    comptime for i in range(Variadic.size(Ts)):
        comptime if conforms_to(Ts[i], Movable):
            if not downcast[Ts[i], Movable].__move_ctor_is_trivial:
                return False
        else:
            return False
    return True
