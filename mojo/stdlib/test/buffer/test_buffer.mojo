# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug -debug-level full %s | FileCheck %s

from pathlib import Path
from tempfile import NamedTemporaryFile

from buffer import Buffer
from buffer.dimlist import Dim


# CHECK-LABEL: test_buffer
fn test_buffer():
    print("== test_buffer")

    alias vec_size = 4
    var data = UnsafePointer[Float32].alloc(vec_size)

    var b1 = Buffer[DType.float32, 4](data)
    var b2 = Buffer[DType.float32, 4](data, 4)
    var b3 = Buffer[DType.float32](data, 4)

    # CHECK: 4 4 4
    print(len(b1), len(b2), len(b3))

    data.free()


# CHECK-LABEL: test_buffer
def test_buffer_tofile():
    print("== test_buffer")
    var buf = Buffer[DType.float32, 4].stack_allocation()
    buf.fill(2.0)
    with NamedTemporaryFile(name=str("test_buffer")) as TEMP_FILE:
        buf.tofile(TEMP_FILE.name)

        with open(TEMP_FILE.name, "r") as f:
            var str = f.read()
            var buf_read = Buffer[DType.float32, 4](
                str.unsafe_ptr().bitcast[Float32]()
            )
            for i in range(4):
                # CHECK: 0.0
                print(buf[i] - buf_read[i])

            # Ensure string is not destroyed before the above check.
            _ = str[0]


def main():
    test_buffer()
    test_buffer_tofile()
