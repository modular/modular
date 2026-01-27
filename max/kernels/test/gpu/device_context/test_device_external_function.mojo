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
"""Test loading and executing an external cubin binary."""

from gpu.host import DeviceContext
from gpu.host.device_context import DeviceExternalFunction
from os import getenv
from testing import assert_equal


def test_external_cubin_vec_add(ctx: DeviceContext):
    """Test loading and executing an external cubin for vector addition."""
    with open(getenv("CUBIN_PATH"), "r") as file:
        cubin_data = file.read_bytes()

    external_func = DeviceExternalFunction(
        ctx,
        function_name="vec_add",  # matches extern "C" name
        # DeviceExternalFunction takes a StringSlice, which is probably wrong.
        # The cubin is [very, very likely] invalid UTF8.
        asm=StringSlice(unsafe_from_utf8=cubin_data),
    )

    comptime length = 1024
    block_dim = 32
    grid_dim = length // block_dim

    in0 = ctx.enqueue_create_buffer[DType.float32](length)
    in1 = ctx.enqueue_create_buffer[DType.float32](length)
    out = ctx.enqueue_create_buffer[DType.float32](length)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = i
            in1_host[i] = 1

    ctx.enqueue_function(
        external_func,
        in0,
        in1,
        out,
        length,
        grid_dim=(grid_dim,),
        block_dim=(block_dim,),
    )

    ctx.synchronize()

    with out.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], i + 1)


def main():
    with DeviceContext() as ctx:
        test_external_cubin_vec_add(ctx)
