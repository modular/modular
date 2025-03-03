# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota, sqrt
from pathlib import Path
from sys import sizeof
from tempfile import NamedTemporaryFile

from max.tensor import Tensor, TensorShape, TensorSpec
from testing import assert_almost_equal, assert_equal, assert_false

from utils.index import Index, IndexList


# CHECK: test_tensor
fn test_tensor():
    print("== test_tensor")

    var tensor = Tensor[DType.float32](TensorShape(5, 2, 3))

    # CHECK: 5x2x3
    print(tensor.shape())

    # CHECK: 5x2x3xfloat32
    print(tensor.spec())

    # CHECK: 5
    print(tensor.spec()[0])

    # CHECK: 2
    print(tensor.spec()[1])

    # CHECK: 3
    print(tensor.spec()[2])

    # CHECK: 3
    print(tensor.dim(2))

    # CHECK: 5x2x3
    print(Tensor[DType.float32]((5, 2, 3)).shape())


# CHECK: test_tensor_init
fn test_tensor_init():
    print("== test_tensor_init")

    var tensor = Tensor[DType.float32](TensorShape(1, 2, 3))

    # CHECK: 0.0
    print(tensor[0, 0, 0])

    var t2 = Tensor[DType.float32](
        TensorShape(1, 2, 3), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    )

    # CHECK: 0.6
    print(t2[0, 1, 2])

    # CHECK: [3.0, 3.0, 3.0, 3.0, 3.0],
    # CHECK-NEXT: [3.0, 3.0, 3.0, 3.0, 3.0]], dtype=float32, shape=2x5
    print(Tensor[DType.float32](TensorShape(2, 5), 3))

    # CHECK: [3.0, 3.0, 3.0, 3.0, 3.0],
    # CHECK-NEXT: [3.0, 3.0, 3.0, 3.0, 3.0]], dtype=float32, shape=2x5
    print(Tensor[DType.float32]((2, 5), 3))


def test_tensor_copy():
    var tensor = Tensor[DType.float32](
        TensorShape(1, 2, 3), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    )
    var copy = Tensor(tensor)

    assert_almost_equal(tensor[0, 1, 2], 0.6)
    assert_almost_equal(copy[0, 1, 2], 0.6)
    assert_false(tensor._ptr == copy._ptr)


# CHECK-LABEL: test_tensor_equality
fn test_tensor_equality():
    print("== test_tensor_equality")

    var t1 = Tensor[DType.float32](TensorShape(2, 3))
    var t2 = Tensor[DType.float32](TensorShape(2, 2))
    var t3 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1
            t3[Index(i, j)] = 3

    for i in range(2):
        for j in range(2):
            t2[Index(i, j)] = 2

    # CHECK: True
    print(t1 == t1)
    # CHECK: True
    print(t2 == t2)

    # CHECK: False
    print(t1 == t2)
    # CHECK: True
    print(t1 != t2)

    # CHECK: False
    print(t1 == t3)
    # CHECK: True
    print(t1 != t3)

    for i in range(2):
        for j in range(3):
            t3[Index(i, j)] = 1

    # CHECK: True
    print(t1 == t3)
    # CHECK: False
    print(t1 != t3)


# CHECK: test_tensor_indexing
fn test_tensor_indexing():
    print("== test_tensor_indexing")

    var tensor = Tensor[DType.float32](TensorShape(5, 2, 3))

    # CHECK: 0
    print(tensor._compute_linear_offset(IndexList[3](0, 0, 0)))

    # CHECK: 0
    print(tensor._compute_linear_offset(0, 0, 0))

    # CHECK: 1
    print(tensor._compute_linear_offset(IndexList[3](0, 0, 1)))

    # CHECK: 1
    print(tensor._compute_linear_offset(0, 0, 1))

    # CHECK: 2
    print(tensor._compute_linear_offset(IndexList[3](0, 0, 2)))

    # CHECK: 2
    print(tensor._compute_linear_offset(0, 0, 2))

    # CHECK: 3
    print(tensor._compute_linear_offset(IndexList[3](0, 1, 0)))

    # CHECK: 3
    print(tensor._compute_linear_offset(0, 1, 0))

    # CHECK: 7
    print(tensor._compute_linear_offset(IndexList[3](0, 2, 1)))

    # CHECK: 7
    print(tensor._compute_linear_offset(0, 2, 1))

    # CHECK: 9
    print(tensor._compute_linear_offset(IndexList[3](1, 1, 0)))

    # CHECK: 9
    print(tensor._compute_linear_offset(1, 1, 0))

    # CHECK: 27
    print(tensor._compute_linear_offset(IndexList[3](4, 1, 0)))

    # CHECK: 27
    print(tensor._compute_linear_offset(4, 1, 0))

    # CHECK: 23
    print(tensor._compute_linear_offset(IndexList[3](3, 1, 2)))

    # CHECK: 23
    print(tensor._compute_linear_offset(3, 1, 2))


# CHECK: test_tensor_indexing2
fn test_tensor_indexing2():
    print("== test_tensor_indexing2")
    var tensor = Tensor[DType.float32](TensorShape(30, 20, 10))

    # CHECK: 223
    print(tensor._compute_linear_offset(1, 2, 3))


# CHECK: test_tensor_fill
fn test_tensor_fill():
    print("== test_tensor_fill")

    var tensor = Tensor[DType.float32](TensorShape(4, 5, 6))

    for i in range(4):
        for j in range(5):
            for k in range(6):
                tensor[Index(i, j, k)] = tensor._compute_linear_offset(i, j, k)

    # CHECK-NOT: ERROR
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if tensor[i, j, k] != (i * 5 + j) * 6 + k:
                    print("ERROR: invalid index fill")


# CHECK: test_tensor_astype
fn test_tensor_astype() raises:
    print("== test_tensor_astype")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = i + 2 * j

    # CHECK{LITERAL}: Tensor([[0, 2],
    # CHECK{LITERAL}: [1, 3]], dtype=int64, shape=2x2)
    var a_int64 = a.astype[DType.int64]()
    print(a_int64)

    assert_equal(String(a_int64._spec.dtype()), "int64")


# CHECK: test_euclidean_distance
fn test_euclidean_distance() raises:
    print("== test_euclidean_distance")

    var num_elements = 100

    var a = Tensor[DType.float64](num_elements)
    var b = Tensor[DType.float64](num_elements)

    for i in range(num_elements):
        a[i] = i
        b[i] = num_elements - i

    var accum = Float64(0)
    for i in range(num_elements):
        var dist = a[i] - b[i]
        accum += dist * dist

    # CHECK: 577.408
    print(sqrt(accum))


# CHECK: test_image_to_grayscale
# COM: This example comes from the change log.
fn test_image_to_grayscale() raises:
    print("== test_image_to_grayscale")

    var height = 256
    var width = 256
    var channels = 3

    # Create the tensor of dimensions height, width, channels and fill.
    var image = Tensor[DType.float32](height, width, channels)
    var num_elements = image.num_elements()
    for x in range(height):
        for y in range(width):
            for c in range(channels):
                image[Index(x, y, c)] = (
                    Float32((x * width + y) * channels + c) / num_elements
                )

    # Declare the grayscale image.
    var gray_scale_image = Tensor[DType.float32](height, width)

    # Perform the RGB to grayscale transform.
    for y in range(height):
        for x in range(width):
            var r = image[y, x, 0]
            var g = image[y, x, 1]
            var b = image[y, x, 2]
            gray_scale_image[Index(y, x)] = 0.299 * r + 0.587 * g + 0.114 * b

    # CHECK: 4.1453{{.*}}e-06
    print(gray_scale_image[0, 0])
    # CHECK: 0.0392192
    print(gray_scale_image[10, 10])
    # CHECK: 0.38676
    print(gray_scale_image[99, 3])
    # CHECK: 0.99998
    print(gray_scale_image[255, 255])


# CHECK-LABEL: test_add
fn test_add() raises:
    print("== test_add")
    var a = Tensor[DType.float32](2, 2)
    var b = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3
            b[Index(i, j)] = 2

    try:
        # CHECK{LITERAL}: Tensor([[5.0, 5.0],
        # CHECK{LITERAL}: [5.0, 5.0]], dtype=float32, shape=2x2)
        print(a + b)
    except:
        print("Error")


# CHECK-LABEL: test_add_scalar
fn test_add_scalar() raises:
    print("== test_add_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[5.0, 5.0],
    # CHECK{LITERAL}: [5.0, 5.0]], dtype=float32, shape=2x2)
    print(a + 2.0)


# CHECK-LABEL: test_radd_scalar
fn test_radd_scalar() raises:
    print("== test_radd_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[5.0, 5.0],
    # CHECK{LITERAL}: [5.0, 5.0]], dtype=float32, shape=2x2)
    print(2.0 + a)


# CHECK-LABEL: test_sub
fn test_sub() raises:
    print("== test_sub")
    var a = Tensor[DType.float32](2, 2)
    var b = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3
            b[Index(i, j)] = 2

    try:
        # CHECK{LITERAL}: Tensor([[1.0, 1.0],
        # CHECK{LITERAL}: [1.0, 1.0]], dtype=float32, shape=2x2)
        print(a - b)
    except:
        print("Error")


# CHECK-LABEL: test_sub_scalar
fn test_sub_scalar() raises:
    print("== test_sub_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[1.0, 1.0],
    # CHECK{LITERAL}: [1.0, 1.0]], dtype=float32, shape=2x2)
    print(a - 2.0)


# CHECK-LABEL: test_rsub_scalar
fn test_rsub_scalar() raises:
    print("== test_rsub_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[-1.0, -1.0],
    # CHECK{LITERAL}: [-1.0, -1.0]], dtype=float32, shape=2x2)
    print(2.0 - a)


# CHECK-LABEL: test_mul
fn test_mul() raises:
    print("== test_mul")
    var a = Tensor[DType.float32](2, 2)
    var b = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3
            b[Index(i, j)] = 2

    try:
        # CHECK{LITERAL}: Tensor([[6.0, 6.0],
        # CHECK{LITERAL}: [6.0, 6.0]], dtype=float32, shape=2x2)
        print(a * b)
    except:
        print("Error")


# CHECK-LABEL: test_mul_scalar
fn test_mul_scalar() raises:
    print("== test_mul_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[6.0, 6.0],
    # CHECK{LITERAL}: [6.0, 6.0]], dtype=float32, shape=2x2)
    print(a * 2.0)


# CHECK-LABEL: test_rmul_scalar
fn test_rmul_scalar() raises:
    print("== test_rmul_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[6.0, 6.0],
    # CHECK{LITERAL}: [6.0, 6.0]], dtype=float32, shape=2x2)
    print(2.0 * a)


# CHECK-LABEL: test_div
fn test_div() raises:
    print("== test_div")
    var a = Tensor[DType.float32](2, 2)
    var b = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3
            b[Index(i, j)] = 2

    try:
        # CHECK{LITERAL}: Tensor([[1.5, 1.5],
        # CHECK{LITERAL}: [1.5, 1.5]], dtype=float32, shape=2x2)
        print(a / b)
    except:
        print("Error")


# CHECK-LABEL: test_div_scalar
fn test_div_scalar() raises:
    print("== test_div_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[1.5, 1.5],
    # CHECK{LITERAL}: [1.5, 1.5]], dtype=float32, shape=2x2)
    print(a / 2.0)


# CHECK-LABEL: test_rdiv_scalar
fn test_rdiv_scalar() raises:
    print("== test_rdiv_scalar")
    var a = Tensor[DType.float32](2, 2)

    for i in range(2):
        for j in range(2):
            a[Index(i, j)] = 3

    # CHECK{LITERAL}: Tensor([[0.5, 0.5],
    # CHECK{LITERAL}: [0.5, 0.5]], dtype=float32, shape=2x2)
    print(1.5 / a)


# CHECK-LABEL: test_pow
fn test_pow() raises:
    print("== test_pow")
    var tensor = Tensor[DType.float32](2, 2)
    iota(tensor.unsafe_ptr(), tensor.num_elements())
    var tensor_pow_2 = tensor**2
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [2.0, 3.0]]
    print(tensor)
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [4.0, 9.0]]
    print(tensor_pow_2)

    var tensor_pow_0 = tensor**0
    # CHECK{LITERAL}: Tensor([[1.0, 1.0],
    # CHECK{LITERAL}: [1.0, 1.0]]
    print(tensor_pow_0)

    var tensor_copy = tensor
    tensor_copy **= 2
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [2.0, 3.0]]
    print(tensor)
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [4.0, 9.0]]
    print(tensor_copy)

    var tensor_pow_neg1 = tensor
    tensor_pow_neg1[Index(0, 0)] = 1.0
    tensor_pow_neg1[Index(0, 1)] = 1.0
    tensor_pow_neg1 **= -1
    # CHECK{LITERAL}: Tensor([[1.0, 1.0],
    # CHECK: [0.5, 0.333{{.*}}]]
    print(tensor_pow_neg1)

    var tensor_large = Tensor[DType.float32](65)
    iota(tensor_large.unsafe_ptr(), tensor_large.num_elements())
    # CHECK{LITERAL}: Tensor([[0.0, 1.0, 4.0, ..., 3844.0, 3969.0, 4096.0]]
    print(tensor_large**2)


# CHECK-LABEL: test_print
fn test_print():
    print("== test_print")
    # CHECK{LITERAL}: Tensor([[[0, 0, 0],
    # CHECK{LITERAL}: [0, 0, 0]],
    # CHECK{LITERAL}: [[0, 0, 0],
    # CHECK{LITERAL}: [0, 0, 0]]], dtype=index, shape=2x2x3)
    var tensor = Tensor[DType.index](2, 2, 3)
    print(tensor)


# CHECK-LABEL: test_print_small
fn test_print_small():
    print("== test_print_small")
    # CHECK{LITERAL}: Tensor([[[0, 1, 2],
    # CHECK{LITERAL}: [3, 4, 5]],
    # CHECK{LITERAL}: [[6, 7, 8],
    # CHECK{LITERAL}: [9, 10, 11]]], dtype=index, shape=2x2x3)
    var tensor = Tensor[DType.index](2, 2, 3)
    iota(tensor.unsafe_ptr(), tensor.num_elements())
    print(String(tensor))


# CHECK-LABEL: test_print_large
fn test_print_large():
    print("== test_print_large")
    # CHECK{LITERAL}: Tensor([[[0, 1, 2, ..., 330, 331, 332],
    # CHECK-NEXT: [333, 334, 335, ..., 663, 664, 665],
    # CHECK-NEXT: [666, 667, 668, ..., 996, 997, 998],
    # CHECK-NEXT: ...,
    # CHECK: 5994, 5995, 5996, ..., 6324, 6325, 6326],
    # CHECK{LITERAL}: [1482183, 1482184, 1482185, ..., 1482513, 1482514, 1482515]]], dtype=index, shape=212x21x333)
    var tensor = Tensor[DType.index](212, 21, 333)
    iota(tensor.unsafe_ptr(), tensor.num_elements())
    print(String(tensor))


# CHECK-LABEL: test_tensor
def test_tensor_from_file():
    print("== test_tensor")
    var tensor = Tensor[DType.int8](TensorShape(4), 1, 2, 3, 4)
    with NamedTemporaryFile(name=String("test_tensor_from_file")) as TEMP_FILE:
        tensor.tofile(TEMP_FILE.name)
        var tensor_from_file = Tensor[DType.int8].fromfile(TEMP_FILE.name)

        # CHECK-NOT: False
        for i in range(tensor.num_elements()):
            print(tensor[i] == tensor_from_file[i])


# CHECK-LABEL: test_tensor_load_save
fn test_tensor_load_save() raises:
    print("== test_tensor_load_save")
    var tensor = Tensor[DType.int8](2, 2, 3)
    tensor._to_buffer().fill(2)

    with NamedTemporaryFile(name=String("test_tensor_load_save")) as TEMP_FILE:
        tensor.save(TEMP_FILE.name)

        var tensor_loaded = Tensor[DType.int8].load(TEMP_FILE.name)

        # CHECK: True
        print(tensor == tensor_loaded)


def main():
    test_tensor()
    test_tensor_init()
    test_tensor_copy()
    test_tensor_equality()
    test_tensor_indexing()
    test_tensor_indexing2()
    test_tensor_fill()
    test_tensor_astype()
    test_euclidean_distance()
    test_image_to_grayscale()
    test_add()
    test_add_scalar()
    test_radd_scalar()
    test_sub()
    test_sub_scalar()
    test_rsub_scalar()
    test_mul()
    test_mul_scalar()
    test_rmul_scalar()
    test_div()
    test_div_scalar()
    test_rdiv_scalar()
    test_pow()
    test_print()
    test_print_small()
    test_print_large()

    # File-related tests
    test_tensor_from_file()
    test_tensor_load_save()
