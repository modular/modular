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
"""GLM's clamped SwiGLU: clamp first, then activate.

GLM-5.3-Flash computes ``silu(min(gate, limit)) * clamp(up, -limit, limit)``.
Neither existing MAX path computes that, and both fail silently:

* ``max/python/max/nn/linear.py:1072`` applies the activation and *then*
  clamps, so it returns ``min(silu(20), 10) = 10`` where GLM needs
  ``silu(min(20, 10)) = 9.9995``.
* ``max/python/max/nn/moe/moe.py:99`` clamps first, but adds GPT-OSS's ``+1``
  bias on ``up`` and an ``alpha`` scale inside the sigmoid that GLM does not
  have.

The divergence only shows up for pre-activation values above roughly the
limit, which a random-weight smoke test never produces and a real checkpoint
does -- so it survives every gate that does not compare tensors.

Used by the dense MLP, the shared expert, the routed experts and the vision
tower, so one fix covers all four.

TODO(GLM53-CORE): move this beside the other activations in
``max/python/max/nn/`` once the vision lane's tower lands, so it is shared
rather than imported across architecture packages. It lives here for now to
keep the parallel lanes off one shared file.
"""

from __future__ import annotations

from max.graph import TensorValue, ops

__all__ = [
    "clamped_swiglu",
    "clamped_swiglu_interleaved",
    "clamped_swiglu_split",
]


def clamped_swiglu(
    gate: TensorValue, up: TensorValue, limit: float
) -> TensorValue:
    """Applies GLM's clamped SwiGLU to already-separated halves.

    Args:
        gate: The gate projection's output, clamped above only.
        up: The up projection's output, clamped on both sides.
        limit: The clamp bound, ``swiglu_limit`` (10.0 for GLM-5.3-Flash).

    Returns:
        ``silu(min(gate, limit)) * clamp(up, -limit, limit)``.
    """
    gate_upper = ops.constant(limit, gate.dtype, device=gate.device)
    up_upper = ops.constant(limit, up.dtype, device=up.device)
    up_lower = ops.constant(-limit, up.dtype, device=up.device)
    # Clamp before the activation. Reversing these two lines is the bug this
    # module exists to avoid.
    gated = ops.silu(ops.min(gate, gate_upper))
    return gated * ops.min(ops.max(up, up_lower), up_upper)


def clamped_swiglu_split(
    gate_up: TensorValue, moe_dim: int, limit: float
) -> TensorValue:
    """Applies :func:`clamped_swiglu` to a ``[gate | up]`` concatenation."""
    return clamped_swiglu(gate_up[:, :moe_dim], gate_up[:, moe_dim:], limit)


def clamped_swiglu_interleaved(
    gate_up: TensorValue, limit: float
) -> TensorValue:
    """Applies :func:`clamped_swiglu` where the halves are strided, not split.

    Matches the fused grouped-GEMM output layout, where gate and up alternate
    along the last axis rather than occupying contiguous halves.
    """
    return clamped_swiglu(gate_up[:, 0::2], gate_up[:, 1::2], limit)
