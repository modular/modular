# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug -debug-level full %s | FileCheck %s

from math import iota
from pathlib import Path
from sys.intrinsics import PrefetchOptions
from tempfile import NamedTemporaryFile

from buffer.buffer import NDBuffer, _compute_ndbuffer_offset
from buffer.dimlist import DimList
from memory import memcmp, memset_zero
from testing import assert_equal

from utils.index import Index, IndexList


# CHECK-LABEL: test_ndbuffer
fn test_ndbuffer():
    print("== test_ndbuffer")
    # Create a matrix of the form
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    # ...
    #  [12, 13, 14, 15]]
    var matrix_stack = InlineArray[Scalar[DType.index], 4 * 4](
        uninitialized=True
    )
    var matrix = NDBuffer[
        DType.index,
        2,
        DimList(4, 4),
    ](matrix_stack.unsafe_ptr())

    matrix[IndexList[2](0, 0)] = 0
    matrix[IndexList[2](0, 1)] = 1
    matrix[IndexList[2](0, 2)] = 2
    matrix[IndexList[2](0, 3)] = 3
    matrix[IndexList[2](1, 0)] = 4
    matrix[IndexList[2](1, 1)] = 5
    matrix[IndexList[2](1, 2)] = 6
    matrix[IndexList[2](1, 3)] = 7
    matrix[IndexList[2](2, 0)] = 8
    matrix[IndexList[2](2, 1)] = 9
    matrix[IndexList[2](2, 2)] = 10
    matrix[IndexList[2](2, 3)] = 11
    matrix[IndexList[2](3, 0)] = 12
    matrix[IndexList[2](3, 1)] = 13
    matrix[IndexList[2](3, 2)] = 14
    matrix[IndexList[2](3, 3)] = 15

    # CHECK: 11
    print(_compute_ndbuffer_offset(matrix, IndexList[2](2, 3)))

    # CHECK: 14
    print(_compute_ndbuffer_offset(matrix, IndexList[2](3, 2)))

    # CHECK: 15
    print(_compute_ndbuffer_offset(matrix, IndexList[2](3, 3)))

    # CHECK: 2
    print(matrix.get_rank())

    # CHECK: 16
    print(matrix.size())

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 1
    print(matrix[0, 1])

    # CHECK: 2
    print(matrix[0, 2])

    # CHECK: 3
    print(matrix[0, 3])

    # CHECK: 4
    print(matrix[1, 0])

    # CHECK: 5
    print(matrix[1, 1])

    # CHECK: 6
    print(matrix[1, 2])

    # CHECK: 7
    print(matrix[1, 3])

    # CHECK: 8
    print(matrix[2, 0])

    # CHECK: 9
    print(matrix[2, 1])

    # CHECK: 10
    print(matrix[2, 2])

    # CHECK: 11
    print(matrix[2, 3])

    # CHECK: 12
    print(matrix[3, 0])

    # CHECK: 13
    print(matrix[3, 1])

    # CHECK: 14
    print(matrix[3, 2])

    # CHECK: 15
    print(matrix[3, 3])


# CHECK-LABEL: test_fill
fn test_fill():
    print("== test_fill")

    var buf_stack = InlineArray[Scalar[DType.index], 3 * 3](uninitialized=True)
    var buf = NDBuffer[DType.index, 2, DimList(3, 3)](buf_stack.unsafe_ptr())
    buf[IndexList[2](0, 0)] = 1
    buf[IndexList[2](0, 1)] = 1
    buf[IndexList[2](0, 2)] = 1
    buf[IndexList[2](1, 0)] = 1
    buf[IndexList[2](1, 1)] = 1
    buf[IndexList[2](1, 2)] = 1
    buf[IndexList[2](2, 0)] = 1
    buf[IndexList[2](2, 1)] = 1
    buf[IndexList[2](2, 2)] = 1

    var filled_stack = InlineArray[Scalar[DType.index], 3 * 3](
        uninitialized=True
    )
    var filled = NDBuffer[DType.index, 2, DimList(3, 3)](
        filled_stack.unsafe_ptr()
    )
    filled.fill(1)

    var err = memcmp(buf.data, filled.data, filled.num_elements())
    # CHECK: 0
    print(err)

    memset_zero(filled.data, filled.num_elements())
    filled.fill(1)
    err = memcmp(buf.data, filled.data, filled.num_elements())
    # CHECK: 0
    print(err)

    memset_zero(buf.data, buf.num_elements())
    filled.fill(0)
    err = memcmp(buf.data, filled.data, filled.num_elements())
    # CHECK: 0
    print(err)


# CHECK-LABEL: test_ndbuffer_prefetch
fn test_ndbuffer_prefetch():
    print("== test_ndbuffer_prefetch")
    # Create a matrix of the form
    # [[0, 1, 2],
    #  [3, 4, 5]]
    var matrix_stack = InlineArray[Scalar[DType.index], 2 * 3](
        uninitialized=True
    )
    var matrix = NDBuffer[DType.index, 2, DimList(2, 3)](
        matrix_stack.unsafe_ptr()
    )

    # Prefetch for write
    for i0 in range(2):
        for j0 in range(3):
            matrix.prefetch[PrefetchOptions().high_locality().for_write()](
                i0, j0
            )

    # Set values
    for i1 in range(2):
        for j1 in range(3):
            matrix[Index(i1, j1)] = i1 * 3 + j1

    # Prefetch for read
    for i2 in range(2):
        for j2 in range(3):
            matrix.prefetch[PrefetchOptions().high_locality().for_read()](
                i2, j2
            )

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 1
    print(matrix[0, 1])

    # CHECK: 2
    print(matrix[0, 2])

    # CHECK: 3
    print(matrix[1, 0])

    # CHECK: 4
    print(matrix[1, 1])

    # CHECK: 5
    print(matrix[1, 2])


# CHECK-LABEL: test_aligned_load_store
fn test_aligned_load_store():
    print("== test_aligned_load_store")
    var matrix = NDBuffer[
        DType.index,
        2,
        DimList(4, 4),
    ].stack_allocation[alignment=128]()

    # Set values
    for i1 in range(4):
        for j1 in range(4):
            matrix[Index(i1, j1)] = i1 * 4 + j1

    # CHECK: [0, 1, 2, 3]
    print(matrix.load[width=4, alignment=16](0, 0))

    # CHECK: [12, 13, 14, 15]
    print(matrix.load[width=4, alignment=16](3, 0))

    # CHECK: [0, 1, 2, 3]
    matrix.store[width=4, alignment=32](Index(3, 0), iota[DType.index, 4]())
    print(matrix.load[width=4, alignment=32](3, 0))


fn test_get_nd_index():
    print("== test_get_nd_index\n")
    var matrix0_stack = InlineArray[Scalar[DType.index], 2 * 3](
        uninitialized=True
    )
    var matrix0 = NDBuffer[DType.index, 2, DimList(2, 3)](
        matrix0_stack.unsafe_ptr()
    )

    var matrix1_stack = InlineArray[Scalar[DType.index], 3 * 5 * 7](
        uninitialized=True
    )
    var matrix1 = NDBuffer[DType.index, 3, DimList(3, 5, 7)](
        matrix1_stack.unsafe_ptr()
    )

    # CHECK: (0, 0)
    print(matrix0.get_nd_index(0))

    # CHECK: (0, 1)
    print(matrix0.get_nd_index(1))

    # CHECK: (1, 0)
    print(matrix0.get_nd_index(3))

    # CHECK: (1, 2)
    print(matrix0.get_nd_index(5))

    # CHECK: (0, 2, 6)
    print(matrix1.get_nd_index(20))

    # CHECK: (2, 4, 6)
    print(matrix1.get_nd_index(104))


def test_print():
    var buf_stack = InlineArray[Scalar[DType.index], 2 * 2 * 3](
        uninitialized=True
    )
    var buffer = NDBuffer[DType.index, 3, DimList(2, 2, 3)](
        buf_stack.unsafe_ptr()
    )
    for i in range(2):
        for j in range(2):
            for k in range(3):
                buffer[Index(i, j, k)] = i * 2 * 3 + j * 3 + k

    assert_equal(
        String(buffer),
        (
            "NDBuffer([[[0, 1, 2],\n"
            "[3, 4, 5]],\n"
            "[[6, 7, 8],\n"
            "[9, 10, 11]]], dtype=index, shape=2x2x3)"
        ),
    )


# CHECK-LABEL: test_ndbuffer
def test_ndbuffer_tofile():
    print("== test_ndbuffer")
    var buf_stack = InlineArray[Float32, 2 * 2](uninitialized=True)
    var buf = NDBuffer[DType.float32, 2, DimList(2, 2)](buf_stack.unsafe_ptr())
    buf.fill(2.0)
    with NamedTemporaryFile(name=String("test_ndbuffer")) as TEMP_FILE:
        buf.tofile(TEMP_FILE.name)

        with open(TEMP_FILE.name, "r") as f:
            var str = f.read()
            var buf_read = NDBuffer[DType.float32, 2, DimList(2, 2)](
                str.unsafe_ptr().bitcast[Float32]()
            )
            for i in range(2):
                for j in range(2):
                    # CHECK: 0.0
                    print(buf[i, j] - buf_read[i, j])

            # Ensure string is not destroyed before the above check.
            _ = str[0]


def test_ndbuffer_tile():
    print("== test_ndbuffer")

    alias M = 8
    alias N = 8
    alias m0_tile_size = 4
    alias n0_tile_size = 4
    alias m1_tile_size = 2
    alias n1_tile_size = 2

    var buf_stack = InlineArray[Float32, M * N](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(M, N)](buf_stack.unsafe_ptr())

    fn linspace(buffer: NDBuffer):
        for i in range(buffer.dim[0]()):
            for j in range(buffer.dim[1]()):
                buffer[i, j] = i * buffer.dim[1]() + j

    fn print_buffer(buffer: NDBuffer):
        for i in range(buffer.dim[0]()):
            for j in range(buffer.dim[1]()):
                print(buffer[i, j], " ", end="")
            print("")

    linspace(buff)
    # CHECK: 0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0
    # CHECK: 8.0  9.0  10.0  11.0  12.0  13.0  14.0  15.0
    # CHECK: 16.0  17.0  18.0  19.0  20.0  21.0  22.0  23.0
    # CHECK: 24.0  25.0  26.0  27.0  28.0  29.0  30.0  31.0
    # CHECK: 32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
    # CHECK: 40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0
    # CHECK: 48.0  49.0  50.0  51.0  52.0  53.0  54.0  55.0
    # CHECK: 56.0  57.0  58.0  59.0  60.0  61.0  62.0  63.0
    print_buffer(buff)

    # CHECK: tile-0[ 0 0 ]
    # CHECK: 0.0  1.0  2.0  3.0
    # CHECK: 8.0  9.0  10.0  11.0
    # CHECK: 16.0  17.0  18.0  19.0
    # CHECK: 24.0  25.0  26.0  27.0
    # CHECK: ----------->
    # CHECK: tile-1[ 0 0 ]
    # CHECK: 0.0  1.0
    # CHECK: 8.0  9.0
    # CHECK: ------
    # CHECK: tile-1[ 0 1 ]
    # CHECK: 2.0  3.0
    # CHECK: 10.0  11.0
    # CHECK: ------
    # CHECK: tile-1[ 1 0 ]
    # CHECK: 16.0  17.0
    # CHECK: 24.0  25.0
    # CHECK: ------
    # CHECK: tile-1[ 1 1 ]
    # CHECK: 18.0  19.0
    # CHECK: 26.0  27.0
    # CHECK: ------
    # CHECK: tile-0[ 0 1 ]
    # CHECK: 4.0  5.0  6.0  7.0
    # CHECK: 12.0  13.0  14.0  15.0
    # CHECK: 20.0  21.0  22.0  23.0
    # CHECK: 28.0  29.0  30.0  31.0
    # CHECK: ----------->
    # CHECK: tile-1[ 0 0 ]
    # CHECK: 4.0  5.0
    # CHECK: 12.0  13.0
    # CHECK: ------
    # CHECK: tile-1[ 0 1 ]
    # CHECK: 6.0  7.0
    # CHECK: 14.0  15.0
    # CHECK: ------
    # CHECK: tile-1[ 1 0 ]
    # CHECK: 20.0  21.0
    # CHECK: 28.0  29.0
    # CHECK: ------
    # CHECK: tile-1[ 1 1 ]
    # CHECK: 22.0  23.0
    # CHECK: 30.0  31.0
    # CHECK: ------
    # CHECK: tile-0[ 1 0 ]
    # CHECK: 32.0  33.0  34.0  35.0
    # CHECK: 40.0  41.0  42.0  43.0
    # CHECK: 48.0  49.0  50.0  51.0
    # CHECK: 56.0  57.0  58.0  59.0
    # CHECK: ----------->
    # CHECK: tile-1[ 0 0 ]
    # CHECK: 32.0  33.0
    # CHECK: 40.0  41.0
    # CHECK: ------
    # CHECK: tile-1[ 0 1 ]
    # CHECK: 34.0  35.0
    # CHECK: 42.0  43.0
    # CHECK: ------
    # CHECK: tile-1[ 1 0 ]
    # CHECK: 48.0  49.0
    # CHECK: 56.0  57.0
    # CHECK: ------
    # CHECK: tile-1[ 1 1 ]
    # CHECK: 50.0  51.0
    # CHECK: 58.0  59.0
    # CHECK: ------
    # CHECK: tile-0[ 1 1 ]
    # CHECK: 36.0  37.0  38.0  39.0
    # CHECK: 44.0  45.0  46.0  47.0
    # CHECK: 52.0  53.0  54.0  55.0
    # CHECK: 60.0  61.0  62.0  63.0
    # CHECK: ----------->
    # CHECK: tile-1[ 0 0 ]
    # CHECK: 36.0  37.0
    # CHECK: 44.0  45.0
    # CHECK: ------
    # CHECK: tile-1[ 0 1 ]
    # CHECK: 38.0  39.0
    # CHECK: 46.0  47.0
    # CHECK: ------
    # CHECK: tile-1[ 1 0 ]
    # CHECK: 52.0  53.0
    # CHECK: 60.0  61.0
    # CHECK: ------
    # CHECK: tile-1[ 1 1 ]
    # CHECK: 54.0  55.0
    # CHECK: 62.0  63.0
    # CHECK: ------
    for tile_i in range(M // m0_tile_size):
        for tile_j in range(N // n0_tile_size):
            print("tile-0[", tile_i, tile_j, "]")
            var tile_4x4 = buff.tile[m0_tile_size, n0_tile_size](
                tile_coords=Index(tile_i, tile_j)
            )
            print_buffer(tile_4x4)
            print("----------->")
            for tile_ii in range(m0_tile_size // m1_tile_size):
                for tile_jj in range(n0_tile_size // n1_tile_size):
                    print("tile-1[", tile_ii, tile_jj, "]")
                    var tile_2x2 = tile_4x4.tile[m1_tile_size, n1_tile_size](
                        tile_coords=Index(tile_ii, tile_jj)
                    )
                    print_buffer(tile_2x2)
                    print("------")


def main():
    test_ndbuffer()
    test_fill()
    test_ndbuffer_prefetch()
    test_aligned_load_store()
    test_get_nd_index()
    test_print()
    test_ndbuffer_tofile()
    test_ndbuffer_tile()
