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

"""Integration test for BufferProtocolBuilder (bf_getbuffer, bf_releasebuffer)."""

import buffer_mojo_module as mojo_module  # type: ignore[import-not-found]


def test_buffer_protocol() -> None:
    print("Testing buffer protocol...")

    obj = mojo_module.FloatBuffer.from_count(4)

    # memoryview basics
    mv = memoryview(obj)
    assert mv.format == "d", f"expected format 'd', got {mv.format!r}"
    assert mv.itemsize == 8, f"expected itemsize 8, got {mv.itemsize}"
    assert len(mv) == 4, f"expected len 4, got {len(mv)}"
    assert mv.readonly, "expected readonly buffer"
    print("  memoryview basics: ok")

    # element access via memoryview
    assert mv[0] == 0.0
    assert mv[1] == 1.0
    assert mv[2] == 2.0
    assert mv[3] == 3.0
    print("  memoryview element access: ok")

    # tolist()
    assert mv.tolist() == [0.0, 1.0, 2.0, 3.0]
    print("  memoryview tolist: ok")

    # bytes() round-trip: struct.unpack should recover the original doubles
    import struct

    raw = bytes(mv)
    unpacked = struct.unpack("4d", raw)
    assert unpacked == (0.0, 1.0, 2.0, 3.0), (
        f"struct unpack mismatch: {unpacked}"
    )
    print("  bytes/struct round-trip: ok")

    # numpy (optional — skip gracefully if not installed)
    try:
        import numpy as np  # type: ignore[import-not-found]

        arr = np.frombuffer(obj, dtype=np.float64)
        assert arr.shape == (4,), f"expected shape (4,), got {arr.shape}"
        assert list(arr) == [0.0, 1.0, 2.0, 3.0], (
            f"numpy values mismatch: {list(arr)}"
        )
        print("  numpy.frombuffer: ok")
    except ImportError:
        print("  numpy not available, skipping numpy test")

    # Empty buffer
    empty = mojo_module.FloatBuffer.from_count(0)
    mv_empty = memoryview(empty)
    assert len(mv_empty) == 0
    print("  empty buffer: ok")

    # Writable-rejection branch in `_getbufferproc_wrapper`: requesting
    # PyBUF_WRITABLE on a read-only FloatBuffer must raise BufferError.
    # Mutating a memoryview surfaces the rejection as TypeError (memoryview
    # itself enforces readonly before reaching our slot), so we exercise
    # the wrapper directly via the PEP 688 `__buffer__` API (3.12+).
    import sys

    if sys.version_info >= (3, 12):
        import inspect

        try:
            obj.__buffer__(inspect.BufferFlags.WRITABLE)
            raise Exception("BufferError expected for PyBUF_WRITABLE request")
        except BufferError as ex:
            assert "not writable" in str(ex), (
                f"expected 'not writable' message, got {ex!r}"
            )
        print("  writable-rejection: ok")
    else:
        print("  writable-rejection: skipped (requires Python 3.12+)")

    print("Buffer protocol tests passed!")


if __name__ == "__main__":
    test_buffer_protocol()
