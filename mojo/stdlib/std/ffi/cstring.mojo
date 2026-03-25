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

from std.collections.string.string_slice import _unsafe_strlen
from std.memory import UnsafeMaybeUninit
from std.memory._nonnull import NonNullUnsafePointer
from std.sys import size_of
from std.utils._nicheable import UnsafeNicheable, NicheIndex


@always_inline
def _validate_bytes(slice: Span[Byte, _]) raises:
    var length = Int(_unsafe_strlen(slice.unsafe_ptr(), UInt(len(slice))))
    if length == len(slice) - 1:
        return
    elif length == 0 or length == len(slice):
        raise Error("CStringSlice is not nul-terminated")
    else:
        raise Error("CStringSlice has interior nul byte")


struct CStringSlice[origin: ImmutOrigin](
    Equatable,
    ImplicitlyCopyable,
    RegisterPassable,
    Sized,
    UnsafeNicheable,
    Writable,
):
    """A non-owning immutable view to a nul-terminated C string (`const char*`).

    This type can be safely constructed from any sort of `StringSlice` or
    `Span[Byte]` that is nul-terminated, or unsafely from a raw pointer.

    Parameters:
        origin: The origin of the `CStringSlice`.
    """

    comptime _PointerType = NonNullUnsafePointer[Int8, Self.origin]

    var _data: Self._PointerType

    @always_inline
    def __init__(
        out self,
        *,
        unsafe_from_ptr: UnsafePointer[Int8, Self.origin],
    ):
        """Construct a `CStringSlice` from an `UnsafePointer`.

        Args:
            unsafe_from_ptr: The `UnsafePointer` to construct the `CStringSlice` from.

        Safety:
            The `UnsafePointer` must be a valid nul-terminated C string.
            The pointer cannot be null. To represent nullability, use
            `Optional[CStringSlice]`.

        Example:

        ```mojo
        from std.ffi import c_char, CStringSlice, external_call

        def getenv_wrapper(
            name: CStringSlice,
        ) raises -> CStringSlice[StaticConstantOrigin]:
            # External call to 'getenv'.
            # C signature: const char *getenv(const char *name);
            var result = external_call[
                "getenv",
                Optional[CStringSlice[StaticConstantOrigin]],
            ](name)

            try:
                # Optional.__getitem__ raises an error if empty.
                return result[]
            except:
                raise Error("getenv returned an error!")
        ```
        """
        assert Bool(unsafe_from_ptr), (
            "Cannot construct a CStringSlice from a null pointer. Use"
            " Optional[CStringSlice] instead to represent nullability."
        )
        self._data = {unsafe_from_nullable = unsafe_from_ptr}

    @always_inline
    def __init__(out self, slice: StringSlice[Self.origin]) raises:
        """Construct a `CStringSlice` from a `StringSlice`.

        Args:
            slice: The `String` to construct the `CStringSlice` from.

        Raises:
            An error if the slice is not nul-terminated or has interior nul
            bytes.

        Example:

        ```mojo
        from std.ffi import CStringSlice
        from std.testing import assert_raises

        var string = String("Hello, World!")

        with assert_raises():
            # This will raise an error since the string is not nul-terminated.
            _ = CStringSlice(string)
        ```
        """
        _validate_bytes(slice.as_bytes())
        # Safety: _validate_bytes ensures span is a non-null terminated cstring.
        self._data = {unsafe_from_nullable = slice.unsafe_ptr().bitcast[Int8]()}

    @always_inline
    def __init__(out self, span: Span[Byte, Self.origin]) raises:
        """Construct a `CStringSlice` from a `Span[Byte]`.

        Args:
            span: The `Span[Byte]` to construct the `CStringSlice` from.

        Raises:
            An error if the slice is not nul-terminated or has interior nul
            bytes.
        """
        _validate_bytes(span)
        # Safety: _validate_bytes ensures span is a non-null terminated cstring.
        self._data = {unsafe_from_nullable = span.unsafe_ptr().bitcast[Int8]()}

    @always_inline
    def __eq__(self, rhs_same: Self) -> Bool:
        """Compare two `CStringSlice`s for equality.

        Args:
            rhs_same: The `CStringSlice` to compare against.

        Returns:
            True if the `CStringSlice`s are equal, False otherwise.
        """
        return Self.__eq__(self, rhs=rhs_same)

    @always_inline
    def __eq__(self, rhs: CStringSlice) -> Bool:
        """Compare two `CStringSlice`s for equality.

        Args:
            rhs: The `CStringSlice` to compare against.

        Returns:
            True if the `CStringSlice`s are equal, False otherwise.
        """
        var a = self.unsafe_ptr()
        var b = rhs.unsafe_ptr()
        if a == b:
            return True

        while a[] == b[]:
            if a[] == Int8(0):
                return True
            a += 1
            b += 1
        return False

    @always_inline
    def __ne__(self, rhs: CStringSlice) -> Bool:
        """Compare two `CStringSlice`s for inequality.

        Args:
            rhs: The `CStringSlice` to compare against.

        Returns:
            True if the `CStringSlice`s are not equal, False otherwise.
        """
        return not (self == rhs)

    @always_inline
    def __len__(self) -> Int:
        """Get the length of the C string. Like C's strlen this does not include
        the nul terminator.

        Returns:
            The length of the C string.
        """
        return Int(_unsafe_strlen(self._data.bitcast[Byte]()))

    def write_to(self, mut writer: Some[Writer]):
        """Write the `CStringSlice` to a `Writer`, the nul terminator is
        omitted.

        Args:
            writer: The `Writer` to write the `CStringSlice` to.
        """
        # TODO: This should error if the bytes are not valid UTF-8.
        writer.write_string(StringSlice(unsafe_from_utf8=self.as_bytes()))

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of this `CStringSlice` to a `Writer`.

        Args:
            writer: The `Writer` to write the `CStringSlice` to.
        """
        t"CStringSlice({self.as_bytes_with_nul()})".write_to(writer)

    @always_inline
    def unsafe_ptr(self) -> UnsafePointer[Int8, Self.origin]:
        """Get a pointer to the underlying `CStringSlice`.

        Returns:
            A pointer to the underlying `CStringSlice`.
        """
        return self._data

    @always_inline
    def as_bytes(self) -> Span[Byte, Self.origin]:
        """Get a span of the underlying `CStringSlice` as bytes.

        The returned span does not include the nul terminator.
        If you want a byte span including the nul terminator, use
        `as_bytes_with_nul()`.

        Returns:
            A span of the underlying `CStringSlice` as bytes.
        """
        return Span(
            ptr=self._data.bitcast[Byte](),
            length=len(self),
        )

    @always_inline
    def as_bytes_with_nul(self) -> Span[Byte, Self.origin]:
        """Get a span of the underlying `CStringSlice` as bytes including the
        nul terminator.

        If you want a byte span not including the nul terminator, use
        `as_bytes()`.

        Returns:
            A span of the underlying `CStringSlice` as bytes.
        """
        return Span(
            ptr=self._data.bitcast[Byte](),
            length=len(self) + 1,
        )

    @staticmethod
    @doc_hidden
    @always_inline
    def niche_count() -> Int:
        return Self._PointerType.niche_count()

    @staticmethod
    @doc_hidden
    @always_inline
    def write_niche[
        index: Int
    ](memory: UnsafePointer[mut=True, UnsafeMaybeUninit[Self], _]):
        comptime assert size_of[Self]() == size_of[Self._PointerType]()
        Self._PointerType.write_niche[index](
            memory.bitcast[UnsafeMaybeUninit[Self._PointerType]]()
        )

    @staticmethod
    @doc_hidden
    @always_inline
    def classify_niche(
        memory: UnsafePointer[mut=False, UnsafeMaybeUninit[Self], _]
    ) -> NicheIndex:
        comptime assert size_of[Self]() == size_of[Self._PointerType]()
        return Self._PointerType.classify_niche(
            memory.bitcast[UnsafeMaybeUninit[Self._PointerType]]()
        )
