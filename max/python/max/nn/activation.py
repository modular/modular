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

"""Activation function utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from max.graph import TensorValue, ops


def activation_function_from_name(
    name: str,
) -> Callable[[TensorValue], TensorValue]:
    activations: dict[str, Callable[[TensorValue], TensorValue]] = {
        "silu": ops.silu,
        "gelu": ops.gelu,
        "gelu_tanh": partial(ops.gelu, approximate="tanh"),
        "relu": ops.relu,
        "tanh": ops.tanh,
        "sigmoid": ops.sigmoid,
    }
    activation_function = activations.get(name)
    if activation_function is None:
        raise ValueError(f"Unrecognized activation function name ({name})")
    return activation_function


# The constants each fused activation reads; the others must stay 0.
_FUSED_ACTIVATION_CONSTANTS: dict[str, tuple[str, ...]] = {
    "silu": ("limit",),
    "swigluoai": ("alpha", "limit"),
    "gelu_tanh": (),
}


@dataclass(frozen=True)
class GatedActivation:
    """The gated activation between an MoE's gate-up and down projections.

    Kernels that fuse the activation compute ``fused_activation``, named like
    the HF config ``hidden_act``:

    - ``"silu"``: ``silu(gate) * up``. A non-zero ``limit`` first clamps the
      inputs to ``min(gate, limit)`` and ``clamp(up, -limit, limit)``.
    - ``"swigluoai"``: ``g * sigmoid(alpha * g) * (u + 1)`` with
      ``g = min(gate, limit)`` and ``u = clamp(up, -limit, limit)``.
    - ``"gelu_tanh"``: ``gelu(gate) * up`` with the tanh approximation
      ``gelu(x) = 0.5 * x * (1 + tanh(0.7978845608028654 *
      (x + 0.044715 * x**3)))``.

    ``gated_activation_fn`` replaces the activation with graph ops, for
    activations no kernel fuses. It takes the ``[gate | up]`` projection and
    ``moe_dim``; see :func:`~max.nn.moe.make_concatenated_gated_activation_fn`.
    A layer that has one runs it instead of any fused kernel, which costs
    performance: the activation runs as separate graph ops over the whole
    gate-up output, and the fused grouped-matmul and quantize paths are
    skipped. Prefer a ``fused_activation`` whenever one matches.
    """

    fused_activation: str = "silu"
    """The activation fused kernels compute."""

    alpha: float = 0.0
    """The sigmoid scale of ``"swigluoai"``; must be 0 for the others."""

    limit: float = 0.0
    """The clamp bound of ``"swigluoai"`` and ``"silu"`` (0 leaves ``"silu"``
    unclamped); must be 0 for ``"gelu_tanh"``."""

    gated_activation_fn: Callable[[TensorValue, int], TensorValue] | None = None
    """Graph-op replacement for the activation, or ``None`` to fuse it."""

    def __post_init__(self) -> None:
        name = self.fused_activation
        if name not in _FUSED_ACTIVATION_CONSTANTS:
            raise ValueError(
                "fused_activation must be one of"
                f" {tuple(_FUSED_ACTIVATION_CONSTANTS)}, got {name!r}"
            )
        takes = _FUSED_ACTIVATION_CONSTANTS[name]
        for constant in ("alpha", "limit"):
            if getattr(self, constant) != 0.0 and constant not in takes:
                raise ValueError(f"{name} takes no {constant}")
        if self.limit < 0.0:
            raise ValueError(f"limit must be non-negative, got {self.limit}")
        if name == "swigluoai" and (self.alpha == 0.0 or self.limit == 0.0):
            raise ValueError("swigluoai needs a non-zero alpha and limit")
        if self.gated_activation_fn is not None and (
            name != "silu" or self.limit != 0.0
        ):
            raise ValueError(
                "a gated_activation_fn replaces the fused activation; leave"
                " fused_activation, alpha, and limit at their defaults"
            )

    def kernel_parameters(self) -> dict[str, float | str]:
        """Returns ``fused_activation`` encoded as kernel parameters.

        Raises:
            ValueError: If ``gated_activation_fn`` is set; a kernel can only
                fuse ``fused_activation``.
        """
        if self.gated_activation_fn is not None:
            raise ValueError(
                "a gated_activation_fn runs as graph ops and cannot be fused"
                " into a kernel"
            )
        return {
            "activation": self.fused_activation,
            "swiglu_alpha": float(self.alpha),
            "swiglu_limit": float(self.limit),
        }
