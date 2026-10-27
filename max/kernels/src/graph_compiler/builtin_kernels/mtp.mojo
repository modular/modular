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
"""Graph-op bindings for multi-token-prediction draft layers.

The kernel math lives in `nn.mtp_eh_norm`. Registration must be declared inside
the built-in kernel library; importing the kernel from `nn` does not add the op
to the graph compiler's registry.
"""

import extensibility
from extensibility import Input, Output, Tensor, TileTensorable
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu import WARP_SIZE

from nn.mtp_eh_norm import mtp_eh_norm_kernel


@extensibility.register("mo.mtp.eh_norm")
struct MTPEhNorm:
    """Registers the `mo.mtp.eh_norm` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        //,
        hidden_size: Int,
        block_threads: Int,
    ](
        out_buf: Some[Output & Tensor],
        embed: Some[Input & Tensor],
        prev: Some[Input & Tensor],
        enorm_weight: Some[Input & Tensor],
        hnorm_weight: Some[Input & Tensor],
        epsilon: Float32,
        ctx: DeviceContext,
    ) raises:
        """Normalizes both draft inputs and writes them side by side.

        Parameters:
            target: Compilation target.
            hidden_size: Channels per input; the output row is twice this.
            block_threads: Threads per block.

        Args:
            out_buf: Output `[num_tokens, 2 * hidden_size]`.
            embed: Token embeddings `[num_tokens, hidden_size]`.
            prev: Target hidden states `[num_tokens, hidden_size]`.
            enorm_weight: Embedding norm weight `[hidden_size]`.
            hnorm_weight: Hidden-state norm weight `[hidden_size]`.
            epsilon: Added inside the square root.
            ctx: Device context.

        Raises:
            Error: If the target is not a GPU.
        """
        comptime assert conforms_to(type_of(out_buf), TileTensorable)
        comptime assert conforms_to(type_of(embed), TileTensorable)
        comptime assert conforms_to(type_of(prev), TileTensorable)
        comptime assert conforms_to(type_of(enorm_weight), TileTensorable)
        comptime assert conforms_to(type_of(hnorm_weight), TileTensorable)
        comptime dtype = type_of(out_buf).dtype
        comptime assert type_of(out_buf).rank == 2
        comptime assert type_of(embed).rank == 2
        comptime assert type_of(prev).rank == 2
        comptime assert type_of(enorm_weight).rank == 1
        comptime assert type_of(hnorm_weight).rank == 1
        comptime assert is_gpu[target](), "mo.mtp.eh_norm is GPU only"
        comptime assert (
            block_threads % WARP_SIZE == 0
        ), "block_threads must be a whole number of warps"
        comptime assert (
            block_threads <= ctx.default_device_info.max_thread_block_size
        ), "block_threads exceeds the device's maximum block size"
        # `block_reduce_dual_sum` reduces over a power-of-two lane group, so
        # pass a power-of-two ceiling rather than this launch's warp count.
        # Slots past the real warp count hold zero.
        comptime max_warps = (
            ctx.default_device_info.max_thread_block_size // WARP_SIZE
        )

        var out_tt = out_buf.to_tile_tensor()
        var embed_tt = embed.to_tile_tensor[dtype]()
        var prev_tt = prev.to_tile_tensor[dtype]()
        var ew_tt = enorm_weight.to_tile_tensor[dtype]()
        var hw_tt = hnorm_weight.to_tile_tensor[dtype]()

        var num_tokens = Int(out_tt.dim[0]())
        if num_tokens == 0:
            return

        comptime kernel = mtp_eh_norm_kernel[
            dtype,
            out_tt.LayoutType,
            out_tt.origin,
            type_of(embed_tt.as_imm()).LayoutType,
            ImmOrigin(embed_tt.origin),
            type_of(prev_tt.as_imm()).LayoutType,
            ImmOrigin(prev_tt.origin),
            type_of(ew_tt.as_imm()).LayoutType,
            ImmOrigin(ew_tt.origin),
            type_of(hw_tt.as_imm()).LayoutType,
            ImmOrigin(hw_tt.origin),
            hidden_size,
            block_threads,
            max_warps,
            out_tt.Engine,
            type_of(embed_tt.as_imm()).Engine,
            type_of(prev_tt.as_imm()).Engine,
            type_of(ew_tt.as_imm()).Engine,
            type_of(hw_tt.as_imm()).Engine,
            pdl_level=PDLLevel.ON,
        ]
        ctx.enqueue_function[kernel](
            out_tt,
            embed_tt.as_imm(),
            prev_tt.as_imm(),
            ew_tt.as_imm(),
            hw_tt.as_imm(),
            epsilon,
            Int32(num_tokens),
            grid_dim=num_tokens,
            block_dim=block_threads,
            # The launch may start while the grid before it in the stream
            # drains; the kernel waits for that grid before its first read.
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
