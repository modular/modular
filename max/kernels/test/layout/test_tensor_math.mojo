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

from layout import (
    TileTensor,
    row_major,
    stack_allocation,
)
from layout._fillers import arange
from layout.tensor_engine import TensorOps
from layout.math import max, sum


def print_vector(tensor: TileTensor):
    for i in range(tensor.dim[0]()):
        print(tensor[i])


def print_matrix(tensor: TileTensor):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print(tensor[i, j], end=" ")
        print()


def binary_op[
    operation: StaticString
](
    dst: TileTensor,
    lhs: TileTensor[dst.dtype, ...],
    rhs: TileTensor[dst.dtype, ...],
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    comptime if operation == "add":
        type_of(dst).Engine.add(
            dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
            lhs=(lhs._storage, lhs.layout),
            rhs=(rhs._storage, rhs.layout),
        )
    elif operation == "sub":
        type_of(dst).Engine.sub(
            dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
            lhs=(lhs._storage, lhs.layout),
            rhs=(rhs._storage, rhs.layout),
        )
    elif operation == "mul":
        type_of(dst).Engine.mul(
            dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
            lhs=(lhs._storage, lhs.layout),
            rhs=(rhs._storage, rhs.layout),
        )
    else:
        comptime assert operation == "div"
        type_of(dst).Engine.truediv(
            dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
            lhs=(lhs._storage, lhs.layout),
            rhs=(rhs._storage, rhs.layout),
        )


def tensor_exp(
    dst: TileTensor, src: TileTensor[dst.dtype, ...]
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    type_of(dst).Engine.exp(
        dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
        src=(src._storage, src.layout),
    )


# CHECK-LABEL: test_reduce_sum
def test_reduce_sum():
    print("== test_reduce_sum")

    # Keep the parameter abstract to exercise generic reduction dispatch.
    def test_reduce_sum_impl(tensor: TileTensor[mut=True, ...]):
        arange(tensor)
        # CHECK: 6.0
        # CHECK: 22.0
        # CHECK: 38.0
        # CHECK: 54.0
        var tensor_4_1 = sum[axis=1](tensor)
        print_vector(tensor_4_1)
        # CHECK: 24.0
        # CHECK: 28.0
        # CHECK: 32.0
        # CHECK: 36.0
        var tensor_4_0 = sum[axis=0](tensor)
        print_vector(tensor_4_0)

    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = TileTensor(tensor_4x4_storage, row_major[4, 4]())
    test_reduce_sum_impl(tensor_4x4)


# CHECK-LABEL: test_reduce_max
def test_reduce_max():
    print("== test_reduce_max")

    def test_reduce_max_impl(tensor: TileTensor[mut=True, ...]):
        arange(tensor)
        var tensor_4_0 = max[axis=0](tensor)
        # CHECK: 12.0
        # CHECK: 13.0
        # CHECK: 14.0
        # CHECK: 15.0
        print_vector(tensor_4_0)

        var tensor_4_1 = max[axis=1](tensor)
        # CHECK: 3.0
        # CHECK: 7.0
        # CHECK: 11.0
        # CHECK: 15.0
        print_vector(tensor_4_1)

    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = TileTensor(tensor_4x4_storage, row_major[4, 4]())
    test_reduce_max_impl(tensor_4x4)


# CHECK-LABEL: test_reduce_res_allocated
def test_reduce_res_allocated():
    print("== test_reduce_res_allocated")
    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = TileTensor(tensor_4x4_storage, row_major[4, 4]())
    arange(tensor_4x4)
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    print_vector(max[axis=0](tensor_4x4))
    # CHECK: 6.0
    # CHECK: 22.0
    # CHECK: 38.0
    # CHECK: 54.0
    print_vector(sum[axis=1](tensor_4x4))


# CHECK-LABEL: test_exp
def test_exp():
    print("== test_exp")
    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = TileTensor(tensor_4x4_storage, row_major[4, 4]())
    arange(tensor_4x4)
    # CHECK: 1.0 2.7182817 7.389056 20.085537
    # CHECK: 54.59815 148.41316 403.42877 1096.6332
    # CHECK: 2980.958 8103.0835 22026.465 59874.14
    # CHECK: 162754.78 442413.37 1202604.2 3269017.2
    var result = stack_allocation[tensor_4x4.dtype](tensor_4x4.layout)
    tensor_exp(result, tensor_4x4)
    print_matrix(result)


# CHECK-LABEL: test_unary_scalar
def test_unary_scalar():
    print("== test_unary_scalar")
    var tensor_4x4_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4 = TileTensor(tensor_4x4_storage, row_major[4, 4]())
    arange(tensor_4x4)

    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    var result = stack_allocation[tensor_4x4.dtype](tensor_4x4.layout)
    var scalar = stack_allocation[tensor_4x4.dtype](tensor_4x4.layout).fill(2)
    binary_op["add"](result, tensor_4x4, scalar)
    print_matrix(result)

    # CHECK: -2.0 -1.0 0.0 1.0
    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    binary_op["sub"](result, tensor_4x4, scalar)
    print_matrix(result)

    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    _ = scalar.fill(10)
    binary_op["mul"](result, tensor_4x4, scalar)
    print_matrix(result)

    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    binary_op["mul"](result, tensor_4x4, scalar)
    print_matrix(result)

    var tensor_4x4_mul10_storage = Array[Float32, 4 * 4](fill={})
    var tensor_4x4_mul_by_10 = TileTensor(
        tensor_4x4_mul10_storage, row_major[4, 4]()
    )
    arange(tensor_4x4_mul_by_10, step=10.0)

    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    binary_op["div"](result, tensor_4x4_mul_by_10, scalar)
    print_matrix(result)

    # CHECK: 1.0 2.0 3.0 4.0
    # CHECK: 5.0 6.0 7.0 8.0
    # CHECK: 9.0 10.0 11.0 12.0
    # CHECK: 13.0 14.0 15.0 16.0
    _ = scalar.fill(1)
    tensor_4x4 += scalar
    print_matrix(tensor_4x4)

    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    tensor_4x4 -= scalar
    print_matrix(tensor_4x4)

    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    _ = scalar.fill(10)
    tensor_4x4 *= scalar
    print_matrix(tensor_4x4)

    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    tensor_4x4 /= scalar
    print_matrix(tensor_4x4)


# CHECK-LABEL: test_binary_same_rank
def test_binary_same_rank():
    print("== test_binary_same_rank")
    var tensor_4x5_storage = Array[Float32, 4 * 5](fill={})
    var tensor_4x5 = TileTensor(tensor_4x5_storage, row_major[4, 5]())
    arange(tensor_4x5)
    var tensor_4x5_2_storage = Array[Float32, 4 * 5](fill={})
    var tensor_4x5_2 = TileTensor(tensor_4x5_2_storage, row_major[4, 5]())
    arange(tensor_4x5_2)
    var scalar = stack_allocation[tensor_4x5_2.dtype](tensor_4x5_2.layout).fill(
        2
    )
    tensor_4x5_2 += scalar

    # CHECK: 2.0 4.0 6.0 8.0 10.0
    # CHECK: 12.0 14.0 16.0 18.0 20.0
    # CHECK: 22.0 24.0 26.0 28.0 30.0
    # CHECK: 32.0 34.0 36.0 38.0 40.0
    var result = stack_allocation[tensor_4x5.dtype](tensor_4x5.layout)
    binary_op["add"](result, tensor_4x5, tensor_4x5_2)
    print_matrix(result)

    # CHECK: 0.0 0.5 1.0 1.5 2.0
    # CHECK: 2.5 3.0 3.5 4.0 4.5
    # CHECK: 5.0 5.5 6.0 6.5 7.0
    # CHECK: 7.5 8.0 8.5 9.0 9.5
    binary_op["div"](result, tensor_4x5, scalar)
    print_matrix(result)

    # CHECK: 0.0 2.0 4.0 6.0 8.0
    # CHECK: 10.0 12.0 14.0 16.0 18.0
    # CHECK: 20.0 22.0 24.0 26.0 28.0
    # CHECK: 30.0 32.0 34.0 36.0 38.0
    var copy = stack_allocation[tensor_4x5.dtype](tensor_4x5.layout)
    copy.copy_from(tensor_4x5)
    tensor_4x5 += copy
    print_matrix(tensor_4x5)

    arange(tensor_4x5)
    _ = scalar.fill(1)
    tensor_4x5 += scalar

    copy.copy_from(tensor_4x5)
    tensor_4x5 /= copy
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    # CHECK: 1.0 1.0 1.0 1.0 1.0
    print_matrix(tensor_4x5)

    tensor_4x5 *= stack_allocation[tensor_4x5.dtype](tensor_4x5.layout).fill(
        10.0
    )
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    # CHECK: 10.0 10.0 10.0 10.0 10.0
    print_matrix(tensor_4x5)

    copy.copy_from(tensor_4x5)
    tensor_4x5 -= copy
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0
    print_matrix(tensor_4x5)


# CHECK-LABEL: test_binary_broadcast_inner
def test_binary_broadcast_inner():
    print("== test_binary_broadcast_inner")
    var tensor_4x5_storage = Array[Float32, 4 * 5](fill={})
    var tensor_4x5 = TileTensor(tensor_4x5_storage, row_major[4, 5]())
    arange(tensor_4x5)
    var tensor_4_storage = Array[Float32, 4](fill={})
    var tensor_4 = TileTensor(tensor_4_storage, row_major[4]())
    arange(tensor_4)
    tensor_4 += stack_allocation[tensor_4.dtype](tensor_4.layout).fill(1)
    # CHECK: -1.0 0.0 1.0 2.0 3.0
    # CHECK: 3.0 4.0 5.0 6.0 7.0
    # CHECK: 7.0 8.0 9.0 10.0 11.0
    # CHECK: 11.0 12.0 13.0 14.0 15.0
    var result = stack_allocation[tensor_4x5.dtype](tensor_4x5.layout)
    binary_op["sub"](result, tensor_4x5, tensor_4)
    print_matrix(result)

    # CHECK: 0.0 0.5 1.0 1.5 2.0
    # CHECK: 2.5 3.0 3.5 4.0 4.5
    # CHECK: 5.0 5.5 6.0 6.5 7.0
    # CHECK: 7.5 8.0 8.5 9.0 9.5
    var scalar = stack_allocation[tensor_4.dtype](tensor_4.layout).fill(2)
    binary_op["div"](result, tensor_4x5, scalar)
    print_matrix(result)


# CHECK-LABEL: test_softmax_math
def test_softmax_math():
    print("== test_softmax_math")
    var tensor_5x4 = stack_allocation[.float32](row_major[5, 4]())
    arange(tensor_5x4)

    var shifted = stack_allocation[tensor_5x4.dtype](tensor_5x4.layout)
    var row_max = max[axis=1](tensor_5x4)
    binary_op["sub"](shifted, tensor_5x4, row_max)
    var exp_norm = stack_allocation[shifted.dtype](shifted.layout)
    tensor_exp(exp_norm, shifted)
    var exp_norm_sum = sum[axis=1](exp_norm)
    var soft_max = stack_allocation[exp_norm.dtype](exp_norm.layout)
    binary_op["div"](soft_max, exp_norm, exp_norm_sum)
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    # CHECK: 0.032058604 0.08714432 0.23688284 0.6439143
    print_matrix(soft_max)


# CHECK: test_max_elemntwise
def test_max_elemntwise():
    print("== test_max_elemntwise")
    var tensor_4x4_a = stack_allocation[.float32](row_major[4, 4]())
    arange(tensor_4x4_a)

    var tensor_4x4_b = stack_allocation[.float32](row_major[4, 4]()).fill(5)

    # CHECK: 5.0 5.0 5.0 5.0
    # CHECK: 5.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    var result = max(tensor_4x4_a.as_imm(), tensor_4x4_b.as_imm())
    for i in range(4):
        for j in range(4):
            print(result[i, j], end=" ")
        print()


def main():
    test_reduce_sum()
    test_reduce_max()
    test_reduce_res_allocated()
    test_exp()
    test_unary_scalar()
    test_binary_same_rank()
    test_binary_broadcast_inner()
    test_softmax_math()
    test_max_elemntwise()
