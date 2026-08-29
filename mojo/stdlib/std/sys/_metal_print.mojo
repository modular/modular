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
"""Metal GPU print support via os_log.

Provides GPU-side helpers to output text from Metal kernels using Apple's
os_log infrastructure. Text is chunked into 64-byte blocks and emitted
via a sentinel call that the compiler replaces with air.os_log intrinsics.

To see output, set MTL_LOG_TO_STDERR=1 before running, or the runtime
sets it automatically.
"""

from std.ffi import external_call
from std.memory import (
    unsafe_memcpy,
    unsafe_memset,
    Pointer,
    MutUntrackedOrigin,
    unsafe_stack_allocation,
)


# Maximum bytes per os_log call (Metal os_log supports 64 %c arguments).
comptime _CHUNK_SIZE = 64


@always_inline
def _metal_os_log_chunk(data: Array[UInt8, 64]):
    """Emit a single 64-byte chunk via the os_log sentinel.

    The compiler's InstructionRewrite pass replaces this sentinel call
    with a Metal air.os_log intrinsic that prints each byte as a %c.

    Args:
        data: Pointer to exactly 64 bytes of data (zero-padded).
    """
    external_call["__mojo_metal_os_log_64", NoneType](data.unsafe_ptr())


@always_inline
def _metal_print_write(text: StringSlice[_]):
    """Write bytes to Metal GPU output via os_log chunking.

    Splits the input into 64-byte chunks, zero-pads each chunk,
    and emits via the os_log sentinel.

    Args:
        text: Sequence of bytes to write.
    """
    var length = text.byte_length()

    if length <= 0:
        return

    var offset = 0
    while offset < length:
        # Allocate a zero-initialized 64-byte buffer on the stack.
        var chunk = Array[UInt8, _CHUNK_SIZE](fill=0)

        # Copy up to 64 bytes from the source.
        var remaining = length - offset
        var copy_len = remaining if remaining < _CHUNK_SIZE else _CHUNK_SIZE
        Span(chunk)[:copy_len].copy_from(
            text.as_bytes()[offset : offset + copy_len]
        )

        # Emit the chunk.
        _metal_os_log_chunk(chunk)
        offset += _CHUNK_SIZE
