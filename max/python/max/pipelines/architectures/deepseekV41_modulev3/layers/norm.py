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
"""RMSNorm matching the V4.1 reference, for the ModuleV3 attention stack.

The reference computes in float32 and multiplies by the weight *before*
casting back::

    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + self.eps)
    return (self.weight * x).to(dtype)

with ``self.eps = rms_norm_eps = 1e-20``. ``eps`` sits inside the square
root's argument, so an all-zero row normalizes to 0 (not NaN):
``0 * rsqrt(0 + 1e-20) == 0``.

``ops.rms_norm`` with ``multiply_before_cast=True`` is exactly this order:
the reduction runs in float32, the weight multiply happens in the
reduction dtype, and only the product is cast back to the input dtype.
The V3 experimental :class:`~max.nn.norm.RMSNorm` instead
applies the Llama-style default (normalize, cast, then scale), which
differs by one rounding -- small, but this stack's KV path feeds a
bit-exact FP8 grid check, so the norm is pinned here rather than
inherited.

The weight is checkpoint bf16; the multiply runs in fp32 against the
fp32-cast weight, matching the reference's ``self.weight * x`` with a
bf16 ``self.weight`` promoted by type promotion.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module import Module

# ``rms_norm_eps`` from the V4.1 ``config.json``.
EPS = 1e-20


class RMSNorm(Module[[Tensor], Tensor]):
    """Reference-order RMSNorm: fp32 math, ``(w * x).to(dtype)``, eps 1e-20."""

    weight: Tensor
    """The learned per-element scale, bf16 as stored in the checkpoint."""

    def __init__(self, dim: int, eps: float = EPS) -> None:
        self.weight = Tensor.ones([dim], dtype=DType.bfloat16)
        self.eps = eps

    @property
    def dim(self) -> int:
        """The size of the last dimension normalized over."""
        return int(self.weight.shape[0])

    def __rich_repr__(self):
        """Yields fields for the rich debug repr."""
        yield "dim", self.dim
        yield "eps", self.eps, EPS

    def forward(self, x: Tensor) -> Tensor:
        """Normalizes ``x`` over its last axis in the reference's order.

        Args:
            x: Tensor whose last axis is ``dim`` wide.

        Returns:
            ``x``'s dtype and shape, with each row scaled to unit RMS and
            multiplied by the weight in float32 before the cast back. An
            all-zero row returns zeros, never NaN.
        """
        # The weight is bf16 and the reduction dtype is fp32, so the
        # multiply runs after the graph promotes the weight -- the same
        # promotion ``self.weight * x`` performs in the reference.
        return F.rms_norm(
            x, self.weight, epsilon=self.eps, multiply_before_cast=True
        ).cast(x.dtype)
