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
"""Selection and validation for Kimi K3's AMD CDNA4 W4A8 compute path.

Copied from ``kimi_k3.layers.moe_latent`` (V2) rather than imported, so this
ModuleV3 port has no build dependency on the V2 package.
"""

from __future__ import annotations

import functools
import os

from max.driver import accelerator_architecture_name
from max.dtype import DType

# Opt-in for the pre-shuffled gate/up weights. Off by default because it is
# unmeasured on this model and because a producer/consumer layout disagreement
# here is the failure that runs, reads fluently and scores zero -- so it wants
# an accuracy gate, not a smoke test, before it becomes the default.
_PRESHUFFLE_B_ENV = "MAX_KIMI_K3_PRESHUFFLE_MOE_B"

# K elements the CDNA 4 block-scaled GEMM tiles at: it reads 64 packed
# MXFP4 bytes per BK step, so a K below this leaves a trailing chunk the
# kernel will not compute.
_AMD_K_GRANULE = 128


@functools.cache
def preshuffled_gate_up_requested() -> bool:
    """Whether the operator opted the gate/up experts into the CDNA 4 preb layout.

    Cached because the env var and the accelerator are both fixed for the
    process.
    """
    requested = os.environ.get(_PRESHUFFLE_B_ENV, "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not requested:
        return False
    try:
        # The permutation targets the gfx95x scaled-MFMA fragment layout; the
        # same checkpoint also serves SM100, which consumes these weights
        # unpermuted.
        return accelerator_architecture_name().startswith("gfx95")
    except Exception:
        return False


def validate_amd_tp_w4a8(
    *,
    mxfp4: bool,
    expert_parallel: bool,
    dtype: DType,
    requested: bool | None = None,
) -> bool:
    """Resolves whether the routed experts run as AMD CDNA4 W4A8.

    It is this port's only MXFP4 expert path, so an MXFP4 checkpoint selects
    it and a configuration that cannot run it fails here.

    Args:
        mxfp4: Whether the routed experts ship MXFP4, which is what W4A8
            dequantizes.
        expert_parallel: Whether the layer runs under expert parallelism,
            which needs its own dispatch transport.
        dtype: The weight dtype the graph runs at.
        requested: ``None`` selects W4A8 for an MXFP4 checkpoint and requires
            a CDNA 4 target. An explicit value skips the target check, so a
            graph builds anywhere.

    Raises:
        ValueError: If W4A8 is selected for a layout or target it cannot run.
    """
    if requested is False or (requested is None and not mxfp4):
        return False
    if expert_parallel:
        raise ValueError("Kimi K3 AMD TP W4A8 supports tensor parallelism only")
    if preshuffled_gate_up_requested():
        raise ValueError(
            "Kimi K3 AMD TP W4A8 requires row-major weights and scales;"
            f" unset {_PRESHUFFLE_B_ENV}"
        )
    if not mxfp4 or dtype != DType.bfloat16:
        raise ValueError(
            "Kimi K3 AMD TP W4A8 requires MXFP4 weights and BF16 activations"
        )
    if requested:
        return True
    try:
        architecture = accelerator_architecture_name()
    except Exception:
        architecture = "unknown"
    if not architecture.startswith("gfx95"):
        raise ValueError(
            "Kimi K3's MXFP4 routed experts need an AMD CDNA4 target;"
            f" got {architecture!r}"
        )
    return True


def validate_amd_packed_k(*, latent_size: int, per_device_ffn: int) -> None:
    """Rejects K widths the CDNA 4 block-scaled GEMM cannot tile.

    The latent is replicated under every split, so no split clears it; the
    expert FFN slice clears under another split.
    """
    for name, k_elems, recovery in (
        ("routed latent", latent_size, "no split clears it"),
        ("per-device expert FFN", per_device_ffn, "change the split"),
    ):
        if k_elems % _AMD_K_GRANULE:
            raise ValueError(
                f"Kimi K3 MXFP4 MoE: {name} {k_elems} must be a multiple of"
                f" {_AMD_K_GRANULE} for the CDNA 4 block-scaled GEMM;"
                f" {recovery}"
            )
