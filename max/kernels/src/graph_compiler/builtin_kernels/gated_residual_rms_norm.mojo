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
from extensibility import InputTensor, OutputTensor
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
def _in[T: DType](t: InputTensor[dtype=T, ...]) -> In[T]:
    return (
        t.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmUntrackedOrigin]()
    )


@always_inline
def _out[T: DType](t: OutputTensor[dtype=T, ...]) -> Out[T]:
    return t.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


@extensibility.register("gated_residual_rms_norm")
struct GatedResidualRMSNorm:
    """Registers the `gated_residual_rms_norm` graph op."""

    @staticmethod
    def execute[
        target: StaticString,
        //,
        scale_branch: Bool,
    ](
        stream_out: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        normed_out: OutputTensor[dtype=DType.bfloat16, rank=2, ...],
        branch: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        residual: InputTensor[dtype=DType.bfloat16, rank=2, ...],
        post_weight: InputTensor[dtype=DType.bfloat16, rank=1, ...],
        gate: InputTensor[dtype=DType.bfloat16, rank=1, ...],
        norm_weight: InputTensor[dtype=DType.bfloat16, rank=1, ...],
        post_eps: Float32,
        norm_eps: Float32,
        temperature: Float32,
        gate_eps: Float32,
        ctx: DeviceContext,
    ) raises:
        """Writes the gated residual stream and its next pre-norm.

        Parameters:
            target: Compilation target.
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
        comptime assert is_gpu[target](), "gated_residual_rms_norm is GPU only"
        comptime assert (
            Int(branch.static_spec.shape_tuple[1]) == HIDDEN
        ), "gated_residual_rms_norm takes 8192 channels"
        var args = GatedResidualArgs(
            _in(branch),
            _in(residual),
            _in(post_weight),
            _in(gate),
            _in(norm_weight),
            _out(stream_out),
            _out(normed_out),
            Int64(branch.dim_size[0]()),
            post_eps,
            norm_eps,
            temperature,
            gate_eps,
        )
        if branch.dim_size[0]() < ctx.default_device_info.sm_count:
            gated_residual_rms_norm_split[scale_branch](args, ctx)
        else:
            gated_residual_rms_norm[scale_branch](args, ctx)
