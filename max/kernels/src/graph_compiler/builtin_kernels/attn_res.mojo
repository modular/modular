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
"""Graph-op binding for Kimi K3's attention-residual softmax mixture.

The kernel lives in `//Kernels/lib/attn_res` (`attn_res.mix`); only the
`@extensibility.register` wrapper lives here, mirroring the `msa.mojo`
binding -- registration has to be declared inside the built-in kernel
library itself, not the standalone lib, for a served graph to resolve
the op.
"""

import extensibility

from std.math import ceildiv

from extensibility import InputTensor, OutputTensor
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from max.gpu.host.info import is_gpu

from attn_res.mix import (
    ATTN_RES_SPLIT_CHUNK,
    attn_res_mix_from_partials_gpu,
    attn_res_mix_gpu,
    attn_res_score_partials_gpu,
)

comptime _PDL_LEVEL = PDLLevel.ON


@extensibility.register("attn_res_mix")
struct AttnResMix:
    """Kimi K3 attention-residual softmax mixture, in one or two fused kernels.

    Replaces the reference's `ops.stack` + RMS-normalize + score-reduce +
    softmax + weighted-reduce chain (6-7 separate kernel launches; see
    `Kernels/lib/attn_res/mix.mojo`'s module docstring for the profile and
    the reassociation this fuses on), starting AFTER the stack (the caller
    still builds `candidates` with its own `ops.stack`). At
    `hidden <= ATTN_RES_SPLIT_CHUNK` that is one kernel, one CTA per token.
    Above it, `attn_res_score_partials_gpu` and
    `attn_res_mix_from_partials_gpu` split the hidden axis across CTAs, so a
    batch-1 decode fills more than one workgroup.

    Tensor shapes:
        - output      : [tokens, hidden]              (OUT)
        - candidates  : [tokens, num_candidates, hidden]
        - proj_weight : [1, hidden]
        - norm_weight : [hidden]
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        eps: StaticString = "1e-6",
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        candidates: InputTensor[dtype=dtype, rank=3, ...],
        proj_weight: InputTensor[dtype=dtype, rank=2, ...],
        norm_weight: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert is_gpu[
            target
        ](), "attn_res_mix is only supported on GPU."

        # `ops.custom`'s extensibility bridge only accepts bool/int/str/DType
        # parameters (no float), so `eps` -- a host-known constant at every
        # call site -- arrives string-encoded; `atof` is prelude, no import.
        var eps_f32 = Float32(atof(eps))

        var tokens = candidates.dim_size(0)
        var num_candidates = candidates.dim_size(1)
        var hidden = candidates.dim_size(2)

        debug_assert(
            proj_weight.dim_size(1) == hidden,
            "attn_res_mix: proj_weight width must match hidden",
        )
        debug_assert(
            norm_weight.dim_size(0) == hidden,
            "attn_res_mix: norm_weight width must match hidden",
        )

        var output_tt = output.to_tile_tensor[.int64]()
        var candidates_tt = candidates.to_tile_tensor[.int64]()
        var proj_tt = proj_weight.to_tile_tensor[.int64]()
        var norm_tt = norm_weight.to_tile_tensor[.int64]()

        comptime BLOCK_SIZE = 256

        # Candidate count is known at every Python call site, so it is a
        # comptime kernel parameter -- dispatch on the runtime value.
        #
        # The bound is set by DEPTH, not by the residual block width. A
        # sublayer at `layer_idx` mixes `layer_idx // attn_res_block_size + 1`
        # block residuals plus the running prefix sum, so the deepest layer
        # needs `(num_layers - 1) // attn_res_block_size + 2`. Published Kimi
        # K3 is 93 layers at `attn_res_block_size` 12, which is 9 -- one past
        # the 8 this used to compile, and a runtime failure rather than a
        # build one. 16 covers K3 with room for a deeper model or a narrower
        # block; raise it if that formula exceeds it.
        comptime MAX_CANDIDATES = 16
        if num_candidates < 1 or num_candidates > MAX_CANDIDATES:
            raise Error(
                "attn_res_mix: unsupported candidate count "
                + String(num_candidates)
                + " (compiled: 1.."
                + String(MAX_CANDIDATES)
                + "). A model of `L` layers at `attn_res_block_size` `B`"
                " needs `(L - 1) // B + 2`; raise MAX_CANDIDATES to cover it."
            )

        # One CTA per token only fills the machine when there are tokens to
        # fill it with; at batch-1 decode it is a single workgroup. Split the
        # hidden axis across CTAs whenever it holds more than one chunk; at
        # one chunk the split has no parallelism left to win and would only
        # add a launch and a round trip through scratch. The two paths round
        # differently on either side of that threshold, by ~2.6e-7 relative
        # in fp32 -- see the module docstring on `Kernels/lib/attn_res`.
        var splits = ceildiv(hidden, ATTN_RES_SPLIT_CHUNK)
        if splits > 1:
            # Per-call fp32 chunk sums: the capture-safe workspace pattern
            # this directory uses elsewhere (`msa.mojo`, `mega_ffn.mojo`).
            # The score kernel writes every slot before the mix kernel reads
            # it, so it needs no initialization, and `DeviceBuffer` frees are
            # stream-ordered, so it outlives both launches.
            var partials_buf = ctx.enqueue_create_buffer[DType.float32](
                tokens * splits * num_candidates * 2
            )
            var partials_tt = TileTensor(
                partials_buf, row_major(tokens, splits, num_candidates, 2)
            )

            # The candidate index is a block index here, not a comptime
            # parameter, so this half of the split compiles ONCE rather than
            # once per entry of the ladder below.
            ctx.enqueue_function[
                attn_res_score_partials_gpu[
                    dtype,
                    partials_tt.LayoutType,
                    partials_tt.Engine,
                    candidates_tt.LayoutType,
                    candidates_tt.Engine,
                    proj_tt.LayoutType,
                    proj_tt.Engine,
                    norm_tt.LayoutType,
                    norm_tt.Engine,
                    BLOCK_SIZE,
                ]
            ](
                partials_tt,
                candidates_tt,
                proj_tt,
                norm_tt,
                Int32(hidden),
                grid_dim=(splits, num_candidates, tokens),
                block_dim=(BLOCK_SIZE,),
                attributes=pdl_launch_attributes(_PDL_LEVEL),
            )

            comptime for c in range(1, MAX_CANDIDATES + 1):
                if num_candidates == c:
                    ctx.enqueue_function[
                        attn_res_mix_from_partials_gpu[
                            dtype,
                            output_tt.LayoutType,
                            output_tt.Engine,
                            candidates_tt.LayoutType,
                            candidates_tt.Engine,
                            partials_tt.LayoutType,
                            partials_tt.Engine,
                            c,
                            BLOCK_SIZE,
                        ]
                    ](
                        output_tt,
                        candidates_tt,
                        partials_tt,
                        eps_f32,
                        Int32(hidden),
                        Int32(splits),
                        grid_dim=(ceildiv(hidden, BLOCK_SIZE), tokens),
                        block_dim=(BLOCK_SIZE,),
                        attributes=pdl_launch_attributes(_PDL_LEVEL),
                    )

            _ = partials_buf^
        else:
            comptime for c in range(1, MAX_CANDIDATES + 1):
                if num_candidates == c:
                    ctx.enqueue_function[
                        attn_res_mix_gpu[
                            dtype,
                            output_tt.LayoutType,
                            output_tt.Engine,
                            candidates_tt.LayoutType,
                            candidates_tt.Engine,
                            proj_tt.LayoutType,
                            proj_tt.Engine,
                            norm_tt.LayoutType,
                            norm_tt.Engine,
                            c,
                            BLOCK_SIZE,
                        ]
                    ](
                        output_tt,
                        candidates_tt,
                        proj_tt,
                        norm_tt,
                        eps_f32,
                        Int32(hidden),
                        grid_dim=(tokens,),
                        block_dim=(BLOCK_SIZE,),
                        attributes=pdl_launch_attributes(_PDL_LEVEL),
                    )
