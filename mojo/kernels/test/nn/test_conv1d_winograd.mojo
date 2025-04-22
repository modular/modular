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
# RUN: %mojo-no-debug %s | FileCheck %s

from math import isclose
from random import rand

from memory import UnsafePointer
from nn.conv import Naive2dConvolution

from utils.index import Index, IndexList


fn winograd_1d_convolution_3[
    type: DType, //, filter_len: Int
](
    input: UnsafePointer[Scalar[type]],
    filter: UnsafePointer[Scalar[type]],
    output: UnsafePointer[Scalar[type]],
    input_len: Int,
):
    # TODO: Current implementation requires input_len >= 4
    constrained[filter_len == 3]()
    # TODO
    # I expected to have to reverse the filter, but
    # internal conv seems not to do that
    var N = input_len - filter_len + 1

    var b0 = filter[0] + filter[2]
    var b1 = 0.5 * (b0 + filter[1])
    var b2 = 0.5 * (b0 - filter[1])

    for i in range(0, N - 1, 2):
        var a0 = (input[i + 1] + input[i + 2]) * b1
        var a1 = (input[i + 2] - input[i + 1]) * b2

        output[i + 0] = a0 + (a1 + (input[i + 0] - input[i + 2]) * filter[0])
        output[i + 1] = a0 - (a1 + (input[i + 1] - input[i + 3]) * filter[2])

    if N % 2 != 0:
        output[N - 1] = (
            filter[0] * input[N - 1]
            + filter[1] * input[N + 0]
            + filter[2] * input[N + 1]
        )


# CHECK-LABEL: test_conv1d_winograd
fn test[type: DType](C: Int):  # Input Len
    print("== test_conv1d_winograd")

    # TODO: make assert dynamic
    # constrained[C >= 4]()
    alias S: Int = 3  # Filter len

    var O: Int = C - S + 1  # Output len (method="same")
    var input_ptr = UnsafePointer[Scalar[type]].alloc(C)
    var filter_ptr = UnsafePointer[Scalar[type]].alloc(S)
    var output_ptr = UnsafePointer[Scalar[type]].alloc(O)
    var output_ref_ptr = UnsafePointer[Scalar[type]].alloc(O)

    rand[type](input_ptr, C)
    rand[type](filter_ptr, S)

    var output_shape = Index(1, 1, 1, O, 1)
    var input_shape = Index(1, 1, 1, C, 1)
    alias filter_shape = Index(1, 1, S, 1, 1)
    alias pad_d = Index(0, 0)
    alias pad_h = Index(0, 0)
    alias pad_w = Index(0, 0)
    alias stride = Index(1, 1, 1)
    alias dilation = Index(1, 1, 1)
    alias num_groups = 1

    Naive2dConvolution[
        type,
        type,
        type,
    ].run(
        output_ref_ptr,
        input_ptr,
        filter_ptr,
        output_shape,
        input_shape,
        filter_shape,
        pad_d,
        pad_h,
        pad_w,
        stride,
        dilation,
        1,
    )

    winograd_1d_convolution_3[S](input_ptr, filter_ptr, output_ptr, C)

    input_ptr.free()
    filter_ptr.free()

    for idx in range(O):
        if not isclose(
            output_ref_ptr[idx],
            output_ptr[idx],
            atol=1e-6,  # absolute error tolerance
            rtol=1e-6,  # relative error tolerance
        ):
            print(
                "diff naive-winograd: ", output_ref_ptr[idx] - output_ptr[idx]
            )
            print("Mismatch!")
            output_ptr.free()
            output_ref_ptr.free()
            return

    output_ptr.free()
    output_ref_ptr.free()

    # CHECK: Succeed
    print("Succeed")


def main():
    alias type = DType.float32

    # Make sure to test both even and odd
    test[type](7)
    test[type](128)
    test[type](129)
    test[type](256)
    test[type](16000)
    test[type](3199)
