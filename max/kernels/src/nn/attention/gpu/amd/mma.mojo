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

from collections import OptionalReg
from math import ceildiv

from gpu import barrier
from layout.tensor_core import TiledTensorCore

from .buffers import KVBuffer, RegisterBuffer, RegisterMMABuffer


@always_inline
fn mma[
    c_register_buffer_type: RegisterBuffer,
    a_register_buffer_type: RegisterMMABuffer,
    b_buffer_type: KVBuffer,
    //,
    tensor_core_mma: TiledTensorCore,
    BK: Int,
    prefetch_function: OptionalReg[fn() capturing -> None],
    swap_a_b: Bool = False,
    beg_iter: Int = 0,
    num_iters: Int = 1,
    prefetched_b_tile: Bool = False,
](
    c: c_register_buffer_type,
    mut a_tile: a_register_buffer_type,
    mut b_tile: b_buffer_type,
):
    comptime assert (
        b_buffer_type._num_stages == 2
    ), "b_tile.num_stages must be 2"
    comptime num_k_mmas2 = ceildiv(
        BK, tensor_core_mma.shape[2] * tensor_core_mma.group_size
    )

    comptime if not prefetched_b_tile:
        b_tile.load_from_dram()

    comptime for i in range(beg_iter, beg_iter + num_iters):
        comptime if i < beg_iter + num_iters - 1:
            b_tile.load_from_dram()

            comptime if i == beg_iter + num_iters - 2:
                comptime if prefetch_function:
                    comptime prefetch_func = prefetch_function.value()
                    prefetch_func()

        b_tile.copy_to_shared[i % 2]()

        barrier()

        comptime for k_mma in range(num_k_mmas2):
            var a_reg_tile = a_tile.get_mma_tile[i, k_mma]()

            b_tile.load_from_shared[k_mma,]()

            tensor_core_mma.mma[swap_a_b=swap_a_b](
                a_reg_tile, b_tile.get_mma_tile(), c.get_reg_tile()
            )

        barrier()
