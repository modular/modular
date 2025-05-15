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

from random import random_float64

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from nn.argsort import argsort
from testing import assert_equal


fn linear_filler(i: Int, n: Int) -> Float32:
    return i


fn reverse_filler(i: Int, n: Int) -> Float32:
    return n - i


fn test_argsort[
    type: DType = DType.float32,
    *,
    filler: fn (Int, Int) -> Float32,
    ascending: Bool = True,
](ctx: DeviceContext, N: Int) raises:
    var input = HostNDBuffer[type, 1](N)

    for i in range(N):
        input.tensor[i] = filler(i, N).cast[type]()

    var device_indices = DeviceNDBuffer[DType.int64, 1](N, ctx=ctx)

    var device_input = input.copy_to_device(ctx)

    argsort[ascending=ascending, target="gpu"](
        device_indices.to_layout_tensor(), device_input.to_layout_tensor(), ctx
    )

    var indices = device_indices.copy_from_device(ctx)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var expected_indices = HostNDBuffer[DType.int64, 1](N)
    argsort[ascending=ascending](
        expected_indices.to_layout_tensor(), input.to_layout_tensor()
    )

    for i in range(N):
        assert_equal(
            indices.tensor[i],
            expected_indices.tensor[i],
            msg=String(
                "indices[",
                i,
                "] = ",
                indices.tensor[i],
                " expected_indices[",
                i,
                "] = ",
                expected_indices.tensor[i],
                " N = ",
                N,
                " ascending = ",
                ascending,
                " at position ",
                i,
            ),
        )

    _ = device_indices^
    _ = device_input^
    _ = indices^
    _ = expected_indices^


fn test_argsort_helper[
    *,
    type: DType,
    filler: fn (Int, Int) -> Float32,
    ascending: Bool,
](ctx: DeviceContext) raises:
    test_argsort[type, filler=filler, ascending=ascending](ctx, N=3731)
    test_argsort[type, filler=filler, ascending=ascending](ctx, N=4096)
    test_argsort[type, filler=filler, ascending=ascending](ctx, N=102_400)
    test_argsort[type, filler=filler, ascending=ascending](ctx, N=16_384)
    test_argsort[type, filler=filler, ascending=ascending](ctx, N=1024)


def main():
    with DeviceContext() as ctx:  # argmax tests
        test_argsort_helper[
            type = DType.float32, filler=linear_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            type = DType.float32, filler=linear_filler, ascending=False
        ](ctx)
        test_argsort_helper[
            type = DType.float32, filler=reverse_filler, ascending=True
        ](ctx)
        test_argsort_helper[
            type = DType.float32, filler=reverse_filler, ascending=False
        ](ctx)
