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
"""Annotations for testing Mojo mirrors of C declarations against C."""


struct CStruct(ImplicitlyCopyable):
    """Describes how a struct that mirrors a C struct is checked against C.

    Attach it to the mirror with `@__annotation`. A mirror without one is
    checked against the C struct of the same name, field by field.

    Example:

    ```mojo
    from std.testing._ffi import CStruct

    @__annotation(CStruct(rename="kevent"))
    struct kevent_s:
        var ident: UInt64
    ```
    """

    var rename: Optional[String]
    """The C struct's name, when it differs from the Mojo struct's name."""

    var skip_fields: Bool
    """Whether to check only the struct's size and alignment."""

    def __init__(
        out self,
        *,
        rename: Optional[String] = None,
        skip_fields: Bool = False,
    ):
        """Creates the annotation.

        Args:
            rename: The C struct's name, when it differs from the Mojo
                struct's name.
            skip_fields: Whether to check only the struct's size and
                alignment, and skip each field's offset, size, and type.
        """
        self.rename = rename
        self.skip_fields = skip_fields


struct CField(ImplicitlyCopyable):
    """Describes how a field of a struct that mirrors a C struct is checked
    against C.

    Attach it to the field with `@__annotation`. A field without one is
    checked against the C field of the same name, at the C type that its Mojo
    type implies.

    Example:

    ```mojo
    from std.testing._ffi import CField, CStruct

    @__annotation(CStruct(rename="kevent"))
    struct kevent_s:
        @__annotation(CField(rename="udata", c_type="void *"))
        var _udata: Array[UInt32, 2]
    ```
    """

    var rename: Optional[String]
    """The C field's name, when it differs from the Mojo field's name."""

    var c_type: Optional[String]
    """The C field's type, overriding the one that the Mojo field's type
    implies."""

    def __init__(
        out self,
        *,
        rename: Optional[String] = None,
        c_type: Optional[String] = None,
    ):
        """Creates the annotation.

        Args:
            rename: The C field's name, when it differs from the Mojo field's
                name.
            c_type: The C field's type as spelled in C, such as `void *`.
                The field's type check uses it instead of the type that the
                Mojo field's type implies. It can't be an array type.
        """
        self.rename = rename
        self.c_type = c_type
