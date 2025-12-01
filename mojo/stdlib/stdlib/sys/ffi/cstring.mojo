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

from collections.string.string_slice import _unsafe_strlen


@always_inline
fn _validate_bytes(slice: Span[Byte]) raises:
    var length = Int(_unsafe_strlen(slice.unsafe_ptr(), UInt(len(slice))))
    if length == len(slice) - 1:
        return
    elif length == 0 or length == len(slice):
        raise Error("CStringSlice is not nul-terminated")
    else:
        raise Error("CStringSlice has interior nul byte")


@register_passable
struct CStringSlice[origin: ImmutOrigin](
    Defaultable,
    Equatable,
    ImplicitlyCopyable,
    Movable,
    Sized,
    Writable,
):
    """A non-owning immutable view to a nul-terminated C string (`const char*`).

    This type can be safely constructed from any sort of `StringSlice` or
    `Span[Byte]` that is nul-terminated, or unsafely from a raw pointer.

    Parameters:
        origin: The origin of the `CStringSlice`.
    """

    var _data: UnsafePointer[Int8, Self.origin]

    @always_inline
    fn __init__(out self):
        """Constructs a null `CStringSlice`."""
        self._data = {}

    @always_inline
    fn __init__(
        out self,
        *,
        unsafe_from_ptr: UnsafePointer[Int8, Self.origin],
    ):
        """Construct a `CStringSlice` from an `UnsafePointer`.

        Args:
            unsafe_from_ptr: The `UnsafePointer` to construct the `CStringSlice` from.

        Safety:
            The `UnsafePointer` must be a valid nul-terminated C string or null.

        Example:

        ```mojo
        from sys.ffi import c_char, CStringSlice, external_call

        fn getenv_wrapper(
            name: CStringSlice,
        ) raises -> CStringSlice[StaticConstantOrigin]:
            # External call to 'getenv'.
            # C signature: const char *getenv(const char *name);
            var result = external_call[
                "getenv",
                UnsafePointer[c_char, StaticConstantOrigin],
            ](name.unsafe_ptr())

            if result:
                return CStringSlice(unsafe_from_ptr=result)
            else:
                raise Error("getenv returned an error!")
        ```
        """
        self._data = unsafe_from_ptr

    @always_inline
    fn __init__(out self, slice: StringSlice[Self.origin]) raises:
        """Construct a `CStringSlice` from a `StringSlice`.

        Args:
            slice: The `String` to construct the `CStringSlice` from.

        Raises:
            An error if the slice is not nul-terminated or has interior nul
            bytes.

        Example:

        ```mojo
        from sys.ffi import CStringSlice
        from testing import assert_raises

        var string = String("Hello, World!")

        with assert_raises():
            # This will raise an error since the string is not nul-terminated.
            _ = CStringSlice(string)
        ```
        """
        _validate_bytes(slice.as_bytes())
        self._data = slice.unsafe_ptr().bitcast[Int8]()

    @always_inline
    fn __init__(out self, span: Span[Byte, Self.origin]) raises:
        """Construct a `CStringSlice` from a `Span[Byte]`.

        Args:
            span: The `Span[Byte]` to construct the `CStringSlice` from.

        Raises:
            An error if the slice is not nul-terminated or has interior nul
            bytes.
        """
        _validate_bytes(span)
        self._data = span.unsafe_ptr().bitcast[Int8]()

    @always_inline
    fn __eq__(self, rhs_same: Self) -> Bool:
        """Compare two `CStringSlice`s for equality.

        Args:
            rhs_same: The `CStringSlice` to compare against.

        Returns:
            True if the `CStringSlice`s are equal, False otherwise.
        """
        return Self.__eq__(self, rhs=rhs_same)

    @always_inline
    fn __eq__(self, rhs: CStringSlice) -> Bool:
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
        if not a or not b:
            return False

        while a[] == b[]:
            if a[] == Int8(0):
                return True
            a += 1
            b += 1
        return False

    @always_inline
    fn __ne__(self, rhs: CStringSlice) -> Bool:
        """Compare two `CStringSlice`s for inequality.

        Args:
            rhs: The `CStringSlice` to compare against.

        Returns:
            True if the `CStringSlice`s are not equal, False otherwise.
        """
        return not (self == rhs)

    @always_inline
    fn __len__(self) -> Int:
        """Get the length of the C string. Like C's strlen this does not include
        the nul terminator.

        Returns:
            The length of the C string.
        """
        if self._data:
            return Int(_unsafe_strlen(self._data.bitcast[Byte]()))
        else:
            return 0

    fn write_to(self, mut writer: Some[Writer]):
        """Write the `CStringSlice` to a `Writer`, the nul terminator is
        omitted.

        Args:
            writer: The `Writer` to write the `CStringSlice` to.
        """
        writer.write_bytes(self.as_bytes())

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Int8, Self.origin]:
        """Get a pointer to the underlying `CStringSlice`.

        Returns:
            A pointer to the underlying `CStringSlice`.
        """
        return self._data

    @always_inline
    fn as_bytes(self) -> Span[Byte, Self.origin]:
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
    fn as_bytes_with_nul(self) -> Span[Byte, Self.origin]:
        """Get a span of the underlying `CStringSlice` as bytes including the
        nul terminator.

        If you want a byte span not including the nul terminator, use
        `as_bytes()`.

        Returns:
            A span of the underlying `CStringSlice` as bytes.
        """
        return Span(
            ptr=self._data.bitcast[Byte](),
            length=0 if not self._data else len(self) + 1,
        )
