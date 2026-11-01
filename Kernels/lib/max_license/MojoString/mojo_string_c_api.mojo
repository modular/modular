# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""C API for creating Mojo String from C strings."""

from std.memory import alloc
from std.collections import Span


@export
def create_mojo_string(
    data: Optional[Pointer[Byte, MutUntrackedOrigin]],
    size: Int,
) abi("C") -> Pointer[String, MutUntrackedOrigin]:
    """Create a Mojo String from C string data.

    Args:
        data: Pointer to the source C string data (UTF-8).
        size: Length of the string in bytes.

    Returns:
        An Pointer[String] owning the Mojo String.
    """
    # Create the Mojo String by copying the UTF-8 data
    var s = String()

    if data and size > 0:
        var byte_span = Span(unsafe_ptr=data.value(), length=size)
        s = String(unsafe_from_utf8=byte_span)

    # Allocate and initialize the pointer
    var ptr = alloc[String]({length = 1}).unsafe_leak()
    ptr.unsafe_write(s^)

    return ptr


@export
def destroy_mojo_string(
    ptr: Pointer[String, MutUntrackedOrigin]
) abi("C") -> None:
    """Free a Mojo String created by create_mojo_string.

    Args:
        ptr: The pointer to free.
    """
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()


@export
def concat_mojo_strings(
    prefix: Pointer[String, MutUntrackedOrigin],
    end: Pointer[String, MutUntrackedOrigin],
) abi("C") -> Pointer[String, MutUntrackedOrigin]:
    """Concatenate two Mojo Strings and return a new String.

    Args:
        prefix: Pointer to the first Mojo String.
        end: Pointer to the second Mojo String.

    Returns:
        An Pointer[String] owning the concatenated Mojo String.
    """
    var result = prefix[] + end[]

    var ptr = alloc[String]({length = 1}).unsafe_leak()
    ptr.unsafe_write(result^)

    return ptr


@export
def get_mojo_string_data(
    ptr: Pointer[String, MutUntrackedOrigin],
) abi("C") -> Pointer[Byte, ImmUntrackedOrigin]:
    """Get a pointer to the string data.

    Args:
        ptr: Pointer to the Mojo String.

    Returns:
        A pointer to the underlying string data (not null-terminated).
    """
    return ptr[].as_bytes().unsafe_ptr()


@export
def get_mojo_string_length(
    ptr: Pointer[String, MutUntrackedOrigin],
) abi("C") -> Int:
    """Get the length of the string in bytes.

    Args:
        ptr: Pointer to the Mojo String.

    Returns:
        The length of the string in bytes.
    """
    return ptr[].byte_length()
