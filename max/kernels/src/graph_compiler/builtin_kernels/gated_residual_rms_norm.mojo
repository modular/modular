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
"""Registers Muse Spark's gated residual boundary as
`gated_residual_rms_norm`.

From SM-count rows up it runs the persistent `nn.gated_residual_rms_norm`.
Below, a CTA per row would leave most SMs idle on a latency-bound row, so
`nn.gated_residual_rms_norm_split` spreads each row over a CTA cluster.
Both match the unfused ops bit for bit.
"""

import extensibility
from extensibility import Input, Output, Tensor, TileTensorable
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu

from nn.gated_residual_rms_norm import (
    HIDDEN,
    GatedResidualArgs,
    In,
    Out,
    gated_residual_rms_norm,
    gated_residual_rms_norm_split,
)


@always_inline
def _in[T: DType](t: Some[Input & Tensor]) -> In[T]:
    comptime assert conforms_to(type_of(t), TileTensorable)
    return (
        t.to_tile_tensor[T]()
        .ptr.unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmUntrackedOrigin]()
    )


@always_inline
def _out[T: DType](t: Some[Output & Tensor]) -> Out[T]:
    comptime assert conforms_to(type_of(t), TileTensorable)
    return t.to_tile_tensor[T]().ptr.unsafe_origin_cast[MutUntrackedOrigin]()


@extensibility.register("gated_residual_rms_norm")
struct GatedResidualRMSNorm:
    """Registers the `gated_residual_rms_norm` graph op."""

    @staticmethod
    def execute[
        scale_branch: Bool,
    ](
        stream_out: Some[Output & Tensor],
        normed_out: Some[Output & Tensor],
        branch: Some[Input & Tensor],
        residual: Some[Input & Tensor],
        post_weight: Some[Input & Tensor],
        gate: Some[Input & Tensor],
        norm_weight: Some[Input & Tensor],
        post_eps: Float32,
        norm_eps: Float32,
        temperature: Float32,
        gate_eps: Float32,
        ctx: DeviceContext,
    ) raises:
        """Writes the gated residual stream and its next pre-norm.

        Parameters:
            scale_branch: Whether the branch is scaled by `1 + post_weight`
                before its norm; otherwise `post_weight` is not read.

        Args:
            stream_out: The new residual stream `[rows, 8192]`.
            normed_out: The stream after the next pre-norm `[rows, 8192]`.
            branch: The branch output `[rows, 8192]`.
            residual: The incoming residual stream `[rows, 8192]`.
            post_weight: The branch's input scale weight `[8192]`.
            gate: The residual gate's logits `[8192]`.
            norm_weight: The next pre-norm's weight `[8192]`.
            post_eps: The branch norm's epsilon.
            norm_eps: The next pre-norm's epsilon.
            temperature: The residual gate's temperature.
            gate_eps: The floor under the residual coefficient's root.
            ctx: Device context.

        Raises:
            Error: If the launch fails.
        """
        comptime assert (
            ctx.target.is_gpu()
        ), "gated_residual_rms_norm is GPU only"
        comptime assert type_of(stream_out).rank == 2
        comptime assert type_of(normed_out).rank == 2
        comptime assert type_of(branch).rank == 2
        comptime assert type_of(residual).rank == 2
        comptime assert type_of(post_weight).rank == 1
        comptime assert type_of(gate).rank == 1
        comptime assert type_of(norm_weight).rank == 1
        comptime assert (
            type_of(branch)._shape_types[1].static_value == HIDDEN
        ), "gated_residual_rms_norm takes 8192 channels"
        comptime bf16 = DType.bfloat16
        var rows = Int(branch.shape()[0].value())
        var args = GatedResidualArgs(
            _in[bf16](branch),
            _in[bf16](residual),
            _in[bf16](post_weight),
            _in[bf16](gate),
            _in[bf16](norm_weight),
            _out[bf16](stream_out),
            _out[bf16](normed_out),
            Int64(rows),
            post_eps,
            norm_eps,
            temperature,
            gate_eps,
        )
        if rows < ctx.default_device_info.sm_count:
            gated_residual_rms_norm_split[scale_branch](args, ctx)
        else:
            gated_residual_rms_norm[scale_branch](args, ctx)
