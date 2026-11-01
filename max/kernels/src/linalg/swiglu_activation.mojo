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
"""Gated activations for the fused SwiGLU epilogues.

Kernels that fuse a gated activation take it as a comptime thin function, so
each activation compiles to its own specialization. `get_swiglu_activation`
builds one from a name and its f32 scalar constants.
"""

from std.math import clamp, exp, recip, tanh


comptime swiglu_activation_type = def(Float32, Float32) thin -> Float32
"""Gated activation `(gate, up) -> out`.

Thin so it can ride as a comptime parameter; any constants it uses must be
comptime values."""


# The stock activations use `recip()` (`rcp.approx.ftz.f32` on NVIDIA) rather
# than fp32 `/`: the fused grouped-matmul epilogue runs 4 warps (1 per
# scheduler), where IEEE `div.f32` latency does not hide.
@inline(.always)
def swiglu[limit: Float32](gate: Float32, up: Float32) -> Float32:
    """Computes the SwiGLU activation `silu(gate) * up`.

    A non-zero `limit` first clamps the inputs to `min(gate, limit)` and
    `clamp(up, -limit, limit)`.

    Parameters:
        limit: Clamp bound, or 0 for no clamp.

    Args:
        gate: Gate projection value.
        up: Up projection value.

    Returns:
        The activated value.
    """
    var g = gate
    var u = up
    comptime if limit != 0.0:
        g = min(g, limit)
        u = clamp(u, -limit, limit)
    return g * recip(Float32(1.0) + exp(-g)) * u


@inline(.always)
def swigluoai[
    alpha: Float32, limit: Float32
](gate: Float32, up: Float32) -> Float32:
    """Computes the clamped `swigluoai` activation.

    With `g = min(gate, limit)` and `u = clamp(up, -limit, limit)`, returns
    `g * sigmoid(alpha * g) * (u + 1)`.

    Parameters:
        alpha: Sigmoid scale.
        limit: Clamp bound.

    Args:
        gate: Gate projection value.
        up: Up projection value.

    Returns:
        The activated value.
    """
    var g = min(gate, limit)
    var u = clamp(up, -limit, limit)
    # `g` is unbounded below. Scaling it by its sigmoid first keeps a huge
    # negative `g` at `g * 0`; `(u + 1) * g` first overflows to -inf, and
    # -inf * 0 is NaN.
    return g * recip(Float32(1.0) + exp(-(g * alpha))) * (u + Float32(1.0))


@inline(.always)
def situ[alpha: Float32, limit: Float32](gate: Float32, up: Float32) -> Float32:
    """Computes the SiTU gated activation.

    With ``beta = alpha`` and ``linear_beta = limit``, returns
    ``beta * tanh(gate / beta) * sigmoid(gate) * u``, where ``u`` is
    ``linear_beta * tanh(up / linear_beta)`` when ``limit != 0`` and ``up``
    otherwise. Unlike the stock activations this uses IEEE fp32 division
    rather than ``recip``, like the graph's elementwise SiTU chain: the
    quantize consumes the value's bfloat16 rounding, so it must track the
    chain within its transcendental ulps, which recip would exceed.

    Parameters:
        alpha: The gate's tanh scale (``beta``); non-zero, since the gate
            is divided by it.
        limit: The up half's soft saturation (``linear_beta``); 0 passes
            ``up`` through unchanged.

    Args:
        gate: Gate projection value.
        up: Up projection value.

    Returns:
        The activated value.
    """
    comptime assert alpha != 0.0, "situ needs a non-zero alpha (beta)"
    var s = alpha * tanh(gate / alpha)
    s = s * (Float32(1.0) / (Float32(1.0) + exp(-gate)))
    var u = up
    comptime if limit != 0.0:
        u = limit * tanh(up / limit)
    return s * u


@inline(.always)
def geglu_tanh(gate: Float32, up: Float32) -> Float32:
    """Computes GeGLU with the tanh approximation of GELU.

    Returns `gelu(gate) * up`, where
    `gelu(x) = 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))`
    in the operation order of `nn.activations.gelu_tanh`.

    Args:
        gate: Gate projection value.
        up: Up projection value.

    Returns:
        The activated value.
    """
    var gelu = (
        gate
        * 0.5
        * (
            1.0
            + tanh(
                Float32(0.7978845608028654)
                * (gate + Float32(0.044715) * gate * gate * gate)
            )
        )
    )
    return gelu * up


@inline(.always)
def _named_swiglu_activation[
    name: StaticString, alpha: Float32, limit: Float32
](gate: Float32, up: Float32) -> Float32:
    comptime match name:
        case "silu":
            return swiglu[limit](gate, up)
        case "swigluoai":
            return swigluoai[alpha, limit](gate, up)
        case "gelu_tanh":
            return geglu_tanh(gate, up)
        case "situ":
            return situ[alpha, limit](gate, up)
        case _:
            comptime assert False, String(
                "unsupported fused SwiGLU activation: ", name
            )


# Binds every parameter of `_named_swiglu_activation` rather than returning the
# matched activation from a function: a parametric instance such as
# `swigluoai[alpha, limit]` returned from a comptime call is dropped when the
# GPU module is sliced out ("Unknown exported symbol").
comptime get_swiglu_activation[
    name: StaticString,
    alpha: Float32 = Float32(0.0),
    limit: Float32 = Float32(0.0),
]: swiglu_activation_type = _named_swiglu_activation[name, alpha, limit]
"""The activation for a name, for a kernel's `activation` parameter. Unknown
names fail to compile.

Parameters:
    name: Activation name, as the Python graph API spells it.
    alpha: Constant for activations that take one; ignored otherwise.
    limit: Constant for activations that take one; ignored otherwise.
"""


@inline(.always)
def apply_swiglu_activation[
    dtype: DType, width: SIMDLength, //, activation: swiglu_activation_type
](gate: SIMD[dtype, width], up: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Applies a gated activation lane by lane.

    Parameters:
        dtype: Element type of the operands.
        width: SIMD width of the operands.
        activation: The activation to apply.

    Args:
        gate: Gate projection values.
        up: Up projection values.

    Returns:
        The activated values.
    """
    var out = SIMD[dtype, width]()
    comptime for i in range(width):
        out[i] = activation(
            gate[i].cast[.float32](), up[i].cast[.float32]()
        ).cast[dtype]()
    return out
