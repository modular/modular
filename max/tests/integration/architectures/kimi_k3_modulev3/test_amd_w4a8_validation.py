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
"""Selection and refusal for Kimi K3's AMD CDNA4 W4A8 routed experts.

CPU-only: every value here is resolved from the checkpoint's quantization
format, the layer layout and the accelerator's name, long before a graph is
built. The kernels take the same operands as the dense fallback, so a wrong
selection produces a model that runs and computes something else rather than
an error -- which is why the selection itself is pinned here.

Ported from the graph-API K3's ``test_amd_w4a8_config_graph.py``: that file
also built graphs, which the ModuleV3 tests cover elsewhere
(``test_moe_gather_quantize_wiring.py``); what it checked about the selection
is what this file keeps.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers import (
    amd_w4a8_validation,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.amd_w4a8_validation import (
    preshuffled_gate_up_requested,
    validate_amd_packed_k,
    validate_amd_tp_w4a8,
)


def _validate(
    *,
    mxfp4: bool = True,
    expert_parallel: bool = False,
    dtype: DType = DType.bfloat16,
    requested: bool | None = None,
) -> bool:
    """The production pair's arguments, overridable one at a time."""
    return validate_amd_tp_w4a8(
        mxfp4=mxfp4,
        expert_parallel=expert_parallel,
        dtype=dtype,
        requested=requested,
    )


def _no_architecture() -> str:
    raise RuntimeError("no accelerator")


@pytest.fixture(autouse=True)
def _cdna4_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Puts the process on the one architecture W4A8 is selected for.

    ``preshuffled_gate_up_requested`` is cached for the process, and both it
    and the selection read the architecture, so both carry the fixture's
    answer.
    """
    preshuffled_gate_up_requested.cache_clear()
    monkeypatch.delenv("MAX_KIMI_K3_PRESHUFFLE_MOE_B", raising=False)
    monkeypatch.setattr(
        amd_w4a8_validation,
        "accelerator_architecture_name",
        lambda: "gfx950",
    )
    yield
    preshuffled_gate_up_requested.cache_clear()


def test_selected_by_default_for_an_mxfp4_checkpoint_on_cdna4() -> None:
    """The checkpoint's own format selects the path; nothing else has to."""
    assert _validate()


def test_a_dense_checkpoint_keeps_the_dense_path() -> None:
    """``None`` selects W4A8 only where there is something packed to keep
    packed; a bf16 checkpoint has no reason to take it."""
    assert not _validate(mxfp4=False)


def test_a_non_cdna4_host_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selection is the only thing standing between an MXFP4 checkpoint
    and a GEMM the accelerator does not have."""
    monkeypatch.setattr(
        amd_w4a8_validation, "accelerator_architecture_name", lambda: "gfx942"
    )

    with pytest.raises(ValueError, match="CDNA4"):
        _validate()


def test_an_unreadable_host_is_refused() -> None:
    """A driver that cannot name its architecture cannot be trusted to have
    the kernel either."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            amd_w4a8_validation,
            "accelerator_architecture_name",
            _no_architecture,
        )
        with pytest.raises(ValueError, match="CDNA4"):
            _validate()


@pytest.mark.parametrize(
    ("expert_parallel", "mxfp4", "dtype", "match"),
    [
        (True, True, DType.bfloat16, "tensor parallelism only"),
        (False, False, DType.bfloat16, "MXFP4"),
        (False, True, DType.float32, "BF16"),
    ],
)
def test_an_explicit_request_fails_closed(
    expert_parallel: bool, mxfp4: bool, dtype: DType, match: str
) -> None:
    """``requested=True`` skips the host check but not the layout's: a caller
    that asked for W4A8 and cannot have it is told, rather than silently
    handed the dense path's numbers."""
    with pytest.raises(ValueError, match=match):
        _validate(
            mxfp4=mxfp4,
            expert_parallel=expert_parallel,
            dtype=dtype,
            requested=True,
        )


def test_an_explicit_override_ignores_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit arm builds a graph anywhere, so a test or a trace can
    stage the path on a host that does not have the kernel."""
    monkeypatch.setattr(
        amd_w4a8_validation, "accelerator_architecture_name", lambda: "sm_100"
    )

    assert _validate(requested=True)
    assert not _validate(requested=False)


def test_the_preshuffle_opt_in_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block-scaled GEMM takes the checkpoint's row-major weight scales;
    the pre-shuffled ``b`` layout is the W4A16 gate/up arm's, so asking for
    both is refused rather than one of them silently winning."""
    monkeypatch.setenv("MAX_KIMI_K3_PRESHUFFLE_MOE_B", "1")
    preshuffled_gate_up_requested.cache_clear()
    assert preshuffled_gate_up_requested()

    assert not _validate(requested=False)
    with pytest.raises(ValueError, match="row-major"):
        _validate()
    with pytest.raises(ValueError, match="row-major"):
        _validate(requested=True)


def test_the_preshuffle_opt_in_needs_cdna4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The permutation is the gfx95x scaled-MFMA fragment layout; the same
    checkpoint serves SM100 unpermuted."""
    monkeypatch.setenv("MAX_KIMI_K3_PRESHUFFLE_MOE_B", "1")
    monkeypatch.setattr(
        amd_w4a8_validation, "accelerator_architecture_name", lambda: "sm_100"
    )
    preshuffled_gate_up_requested.cache_clear()

    assert not preshuffled_gate_up_requested()


def test_the_preshuffle_opt_in_is_off_by_default() -> None:
    assert not preshuffled_gate_up_requested()


def test_the_packed_k_granule_admits_a_tileable_width() -> None:
    """The production pair: the latent is replicated, and one device's expert
    FFN slice has to be a whole number of 128-element blocks."""
    validate_amd_packed_k(latent_size=3584, per_device_ffn=256)


@pytest.mark.parametrize(
    ("latent_size", "per_device_ffn", "match"),
    [
        # No split narrows the latent, so a K the GEMM cannot tile is fatal.
        (3000, 192, "routed latent"),
        # The FFN slice is the caller's split to change.
        (3584, 200, "per-device expert FFN"),
    ],
)
def test_an_untileable_k_is_refused(
    latent_size: int, per_device_ffn: int, match: str
) -> None:
    """A K the block-scaled GEMM cannot tile would otherwise surface from the
    graph compiler as a constraint on ``BK_BYTES``, naming neither the
    projection nor the split that produced it."""
    with pytest.raises(ValueError, match=match):
        validate_amd_packed_k(
            latent_size=latent_size, per_device_ffn=per_device_ffn
        )
