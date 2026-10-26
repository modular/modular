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
"""Validation for Kimi K3's AMD CDNA4 W4A8 compute path.

Copied from ``kimi_k3.layers.moe_latent`` (V2) rather than imported, so this
ModuleV3 port has no build dependency on the V2 package.
"""

from __future__ import annotations

import functools
import os

from max.driver import accelerator_architecture_name
from max.dtype import DType

# Opt-in for the W4A4 measurement path. Named for what it costs rather than for
# the kernel it reaches, so setting it without reading the docs still says so.
_LOSSY_W4A4_ENV = "MAX_KIMI_K3_LOSSY_MXFP4_ACTIVATIONS"

# Opt-in for the pre-shuffled gate/up weights. Off by default because it is
# unmeasured on this model and because a producer/consumer layout disagreement
# here is the failure that runs, reads fluently and scores zero -- so it wants
# an accuracy gate, not a smoke test, before it becomes the default.
_PRESHUFFLE_B_ENV = "MAX_KIMI_K3_PRESHUFFLE_MOE_B"

_AMD_W4A8_ENV = "MAX_KIMI_K3_AMD_W4A8"

# K elements the CDNA 4 block-scaled GEMM tiles at: it reads 64 packed
# MXFP4 bytes per BK step, so a K below this leaves a trailing chunk the
# kernel will not compute.
_AMD_K_GRANULE = 128


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def amd_tp_w4a8_requested() -> bool:
    """Whether the operator selected the dense AMD TP W4A8 path."""
    return _env_flag(_AMD_W4A8_ENV)


@functools.cache
def preshuffled_gate_up_requested() -> bool:
    """Whether the operator opted the gate/up experts into the CDNA 4 preb layout.

    Gated the same way :func:`amd_tp_w4a8_requested` is, and cached for the
    same reasons: the env var and the accelerator are both fixed for the
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
    """Resolve the opt-in before pricing memory or mutating weight layouts.

    This compute eligibility is separate from NVIDIA's quantized EP transport.
    Unsupported requested combinations fail rather than changing dispatch.

    Args:
        requested: ``None`` reads the opt-in env var and requires a CDNA 4
            host. An explicit value skips both, so a graph builds anywhere,
            but still rejects layouts the dispatch cannot consume.
    """
    if requested is None:
        if not amd_tp_w4a8_requested():
            return False
        # Read raw rather than through `lossy_mxfp4_activations_available`,
        # which is cached and warns that W4A4 is about to run.
        if _env_flag(_LOSSY_W4A4_ENV):
            raise ValueError(
                f"{_AMD_W4A8_ENV} and {_LOSSY_W4A4_ENV} select different"
                " activation formats; set only one"
            )
    elif not requested:
        return False
    if expert_parallel:
        raise ValueError(f"{_AMD_W4A8_ENV} supports tensor parallelism only")
    if preshuffled_gate_up_requested():
        raise ValueError(
            f"{_AMD_W4A8_ENV} requires row-major weights and scales"
        )
    if not mxfp4 or dtype != DType.bfloat16:
        raise ValueError(
            f"{_AMD_W4A8_ENV} requires MXFP4 weights and BF16 activations"
        )
    if requested is not None:
        return True
    try:
        architecture = accelerator_architecture_name()
    except Exception as error:
        raise ValueError(f"{_AMD_W4A8_ENV} requires AMD CDNA4") from error
    if not architecture.startswith("gfx95"):
        raise ValueError(
            f"{_AMD_W4A8_ENV} requires AMD CDNA4, got {architecture}"
        )
    return True


def validate_amd_packed_k(
    *, latent_size: int, per_device_ffn: int, opt_in_env: str
) -> None:
    """Rejects K widths the CDNA 4 block-scaled GEMM cannot tile.

    The latent is replicated under every split, so only turning the opt-in
    off clears it; the expert FFN slice also clears under another split.
    """
    for name, k_elems, recovery in (
        ("routed latent", latent_size, f"unset {opt_in_env}"),
        (
            "per-device expert FFN",
            per_device_ffn,
            f"unset {opt_in_env} or change the split",
        ),
    ):
        if k_elems % _AMD_K_GRANULE:
            raise ValueError(
                f"Kimi K3 MXFP4 MoE: {name} {k_elems} must be a multiple of"
                f" {_AMD_K_GRANULE} for the CDNA 4 block-scaled GEMM;"
                f" {recovery}"
            )
