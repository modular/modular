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
