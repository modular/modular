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
"""Runs the AMD W4A8 TP2 layer end to end to grade its window handoff.

The routing glue rounds the shared expert's gate/up window to bfloat16 while
it gathers the routed experts' rows, and the exit collective reads that
window where the glue wrote it -- one tensor the glue produced and a
distributed collective consumed. Nothing on a single device exercises that
hop, and the wiring target only stages the TP2 graph.

So this runs the layer twice on two MI355s, over one fixture:

* as it ships: the glue's window goes straight into the exit collective;
* with the exit collective replaced by the chain it fuses -- the
  ``allreduce_rms_norm`` on the latent, this rank's latent column shard,
  the shared expert's own ``situ_and_mul`` of that same window, and the
  concatenation -- which is the exit the layer ran before the fusion, when
  the window reached it narrowed by a cast of its own.

Both arms compute the same function and differ only in the exit's
rounding, so their outputs must agree to within bfloat16 rounding: the
fused op's own multi-GPU test
(``max/tests/integration/graph/multi_gpu/test_allreduce_rms_norm_exit_concat.py``)
bounds that substitution at one ULP on the normed shard and the
activation, and the exit GEMM then accumulates those one-ULP differences
over its K. What that comparison cannot see -- both arms read the same
window -- is whether the window carries the right numbers, so the test
also grades the window the exit received against the shared expert's own
``gate_up_proj`` of the same input, which is what the merged entry GEMM
computed those columns from.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
from _moe_test_utils import assign_experts
from max import functional as F
from max.driver import (
    Accelerator,
    Buffer,
    Device,
    accelerator_api,
    accelerator_architecture_name,
)
from max.dtype import DType
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
    tensor_parallel_latent_moe,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh, NamedMapping

# One whole fused router-GEMV K tile (512 on gfx950), so the gate runs the
# mixed GEMV Kimi K3 opts into there rather than the cast + matmul fallback.
HIDDEN = 512
LATENT = 128
# The routed expert's FFN width. 128 per rank at TP2 is the CDNA4
# block-scaled GEMM's K granule, which the packed scales need un-split.
MOE_INTER = 256
NUM_SHARED_EXPERTS = 2
SHARED_INTER = MOE_INTER * NUM_SHARED_EXPERTS
NUM_EXPERTS = 64
TOPK = 4
# K3's decode step, and a row count the exit collective keeps on its fused
# arm.
ROWS = 32
EPS = 1e-5
BETA = 1.3
LINEAR_BETA = 0.9
ROUTED_SCALING = 1.7
TP_AXIS = "tp"

# Every bound is a largest-absolute-difference as a fraction of the largest
# value, because near-zero outputs have no meaningful bfloat16 ULP distance
# to a value of the array's own scale. One bfloat16 ULP is 2**-7 (7.8e-3)
# of a value's magnitude.
#
# Measured on this fixture: the window is bit-identical to the shared
# expert's own gate/up (the merged entry GEMM runs the same products as
# `gate_up_proj` over the same rows), and the two exits' outputs differ by
# 8.7e-3 -- one ULP at the output's magnitude, from the exit chain's own
# rounding. The bounds leave the window room for a rounding flip if a
# kernel's tiling changes the accumulation order, and the exit ~2x.
MAX_EXIT_REL_ERROR = 2e-2
MAX_WINDOW_REL_ERROR = 1e-2


def _mesh() -> DeviceMesh:
    return DeviceMesh(
        devices=(Accelerator(0), Accelerator(1)),
        mesh_shape=(2,),
        axis_names=(TP_AXIS,),
    )


def _packed_tensor(
    array: npt.NDArray[np.uint8], dtype: DType, device: Device
) -> Tensor:
    """Relabels raw bytes as ``dtype`` without converting them.

    ``Tensor(array, dtype=...)`` rejects a dtype that disagrees with the
    array's own, and these are packed E2M1 codes, not small integers.
    """
    buf = (
        Buffer.from_dlpack(array)
        .to(device)
        .view(dtype=dtype, shape=tuple(int(d) for d in array.shape))
    )
    return Tensor(storage=buf)


_SF_VECTOR_SIZE = 32

_E2M1_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float64,
)


def _encode_mxfp4(
    values: np.ndarray, sf_vector_size: int = _SF_VECTOR_SIZE
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Quantizes realistic weights to packed E2M1 codes and E8M0 scales.

    Same encoder as ``test_moe_latent_mxfp4_gpu.py``'s: per block it picks
    the power-of-two scale that puts the block's peak at E2M1's top code,
    then rounds to the nearest of the eight magnitudes. Realistic weights
    (not random bytes) keep the routed expert's contribution the size the
    layer is built for.
    """
    *lead, k = values.shape
    blocks = values.reshape(*lead, k // sf_vector_size, sf_vector_size)
    peak = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1e-30)
    # The highest E2M1 magnitude is 6; one exponent higher would clip it.
    exponent = np.clip(np.floor(np.log2(peak / 6.0)), -127, 127)
    scale_bytes = (exponent + 127).astype(np.uint8)
    normalized = blocks / np.exp2(exponent)
    magnitudes = np.abs(normalized)
    code_idx = np.argmin(
        np.abs(magnitudes[..., None] - _E2M1_VALUES[:8]), axis=-1
    )
    nibble = code_idx.astype(np.uint8) | np.where(
        normalized < 0, 0x8, 0x0
    ).astype(np.uint8)
    nibble = nibble.reshape(*lead, k // 2, 2)
    packed_bytes = (nibble[..., 0] | (nibble[..., 1] << 4)).astype(np.uint8)
    return packed_bytes, scale_bytes.reshape(*lead, k // sf_vector_size)


def _random_fixture(
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, npt.NDArray[np.uint8]]]:
    """The layer's dense weights, plus its routed experts' packed bytes."""
    rng = np.random.default_rng(seed)

    def gaussian(*shape: int, scale: float) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return (
        {
            "gate_score": gaussian(NUM_EXPERTS, HIDDEN, scale=0.1),
            "correction_bias": gaussian(NUM_EXPERTS, scale=0.1),
            "entry": gaussian(LATENT, HIDDEN, scale=1 / np.sqrt(HIDDEN)),
            "exit": gaussian(HIDDEN, LATENT, scale=1 / np.sqrt(LATENT)),
            "norm": np.ones(LATENT, dtype=np.float32),
            "shared_gate_up": gaussian(
                2 * SHARED_INTER, HIDDEN, scale=1 / np.sqrt(HIDDEN)
            ),
            "shared_down": gaussian(
                HIDDEN, SHARED_INTER, scale=1 / np.sqrt(SHARED_INTER)
            ),
            "x": rng.standard_normal((ROWS, HIDDEN)).astype(np.float32),
        },
        _packed_experts(
            gaussian(
                NUM_EXPERTS,
                2 * MOE_INTER,
                LATENT,
                scale=1 / np.sqrt(LATENT),
            ),
            gaussian(
                NUM_EXPERTS,
                LATENT,
                MOE_INTER,
                scale=1 / np.sqrt(MOE_INTER),
            ),
        ),
    )


def _packed_experts(
    gate_up: np.ndarray, down: np.ndarray
) -> dict[str, npt.NDArray[np.uint8]]:
    """Packs the stacked expert weights the way the checkpoint stores them."""
    packed: dict[str, npt.NDArray[np.uint8]] = {}
    for name, values in (("gate_up", gate_up), ("down", down)):
        bytes_, scales = _encode_mxfp4(values)
        packed[name] = bytes_
        packed[f"{name}_scale"] = scales
    return packed


def _build_layer(
    weights: dict[str, np.ndarray], packed: dict[str, npt.NDArray[np.uint8]]
) -> KimiK3LatentMoE:
    """Builds the layer over one fixture, before any sharding happens."""
    layer = KimiK3LatentMoE(
        hidden_size=HIDDEN,
        latent_size=LATENT,
        moe_intermediate_size=MOE_INTER,
        shared_intermediate_size=SHARED_INTER,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOPK,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        rms_norm_eps=EPS,
        routed_scaling_factor=ROUTED_SCALING,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        dtype=DType.bfloat16,
        quant_config=QuantConfig.mxfp4(mlp_quantized_layers=[0]),
        # Explicit: this is the CDNA4 W4A8 path, whatever device the test
        # process happens to see.
        amd_w4a8=True,
        # The chain under test. The MegaFFN is a different routed-expert
        # implementation, covered against this one elsewhere.
        amd_mega_ffn=False,
    )

    def bf16(array: np.ndarray) -> Tensor:
        return Tensor(array, dtype=DType.float32).cast(DType.bfloat16)

    # The gate weight keeps the checkpoint's bfloat16 storage, so it shares
    # the entry GEMM's dtype and rides the merged entry GEMM: that GEMM then
    # writes float32 (the logits keep their accumulation), which is where the
    # window starts life. A float32 gate weight would take the other branch
    # and leave the window bfloat16 throughout -- no fold to test.
    layer.gate.gate_score.weight = bf16(weights["gate_score"])
    layer.gate.e_score_correction_bias = Tensor(
        weights["correction_bias"], dtype=DType.float32
    )

    layer.routed_expert_down_proj.weight = bf16(weights["entry"])
    layer.routed_expert_up_proj.weight = bf16(weights["exit"])
    layer.routed_expert_norm.weight = bf16(weights["norm"])
    layer.shared_experts.gate_up_proj.weight = bf16(weights["shared_gate_up"])
    layer.shared_experts.down_proj.weight = bf16(weights["shared_down"])
    assign_experts(
        layer,
        packed["gate_up"],
        packed["down"],
        lambda a, device: _packed_tensor(a, DType.uint8, device),
    )
    assign_experts(
        layer,
        packed["gate_up_scale"],
        packed["down_scale"],
        lambda a, device: _packed_tensor(a, DType.float8_e8m0fnu, device),
        leaf="weight_scale",
    )
    return layer


def _standalone_exit(
    layer: KimiK3LatentMoE, record: Callable[[Tensor], None] | None = None
) -> Callable[..., Tensor]:
    """The exit chain the fused collective replaces, as a functional.

    ``allreduce_rms_norm`` on the latent, this rank's column shard of it
    cast to bfloat16, and the shared expert's own activation beside it --
    the layer's exit before the fusion, when the window reached it narrowed
    by a cast instead of by the routing glue.
    """

    def unfused(
        t: Tensor,
        gate_up: Tensor,
        gamma: Tensor,
        *,
        epsilon: float,
        beta: float,
        linear_beta: float,
        mesh_axis: int | str | Sequence[int | str] = 0,
    ) -> Tensor:
        if record is not None:
            record(gate_up)
        normed = F.allreduce_rms_norm(
            t,
            gamma,
            epsilon,
            weight_offset=0.0,
            multiply_before_cast=False,
            mesh_axis=mesh_axis,
        )
        dtype = layer.routed_expert_up_proj.weight.dtype
        latent_slice = normed.to(NamedMapping(t.mesh, (None, TP_AXIS))).cast(
            dtype
        )
        activated = layer.shared_experts.situ_and_mul(gate_up).cast(dtype)
        return F.concat([latent_slice, activated], axis=-1)

    return unfused


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    """Largest absolute difference, as a fraction of the largest value."""
    return float(np.abs(got - want).max()) / max(
        float(np.abs(want).max()), 1e-12
    )


class _Result(NamedTuple):
    """One arm's outcome: the layer output, plus what its exit received."""

    out: np.ndarray
    # The window the exit collective (or its replacement) was handed.
    window: np.ndarray | None
    # The shared expert's own gate/up of the same input: what the merged
    # entry GEMM computed the window's columns from, at TP2.
    reference: np.ndarray | None


def _run(
    weights: dict[str, np.ndarray],
    packed: dict[str, npt.NDArray[np.uint8]],
    *,
    fold: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> _Result:
    """Runs the layer at TP2, fused or with the standalone exit chain."""
    mesh = _mesh()
    captured: list[Tensor] = []
    with default_device(mesh):
        layer = tensor_parallel_latent_moe(_build_layer(weights, packed), mesh)
        x = Tensor(weights["x"], dtype=DType.float32).cast(DType.bfloat16)

        def record(gate_up: Tensor) -> None:
            # A window's reported width is the unreduced one (the glue
            # divides it by the mesh size for its kernel-visible width), so
            # the assertions below read its dtype and values, not its shape.
            captured.append(gate_up)

        if fold:
            fused = F.allreduce_rms_norm_exit_concat

            def spy(
                t: Tensor,
                gate_up: Tensor,
                gamma: Tensor,
                *,
                epsilon: float,
                beta: float,
                linear_beta: float,
                mesh_axis: int | str | Sequence[int | str] = 0,
            ) -> Tensor:
                record(gate_up)
                return fused(
                    t,
                    gate_up,
                    gamma,
                    epsilon=epsilon,
                    beta=beta,
                    linear_beta=linear_beta,
                    mesh_axis=mesh_axis,
                )

            with monkeypatch.context() as patched:
                patched.setattr(F, "allreduce_rms_norm_exit_concat", spy)
                out = layer(x)
        else:
            with monkeypatch.context() as patched:
                patched.setattr(
                    F,
                    "allreduce_rms_norm_exit_concat",
                    _standalone_exit(layer, record),
                )
                out = layer(x)

        assert len(captured) == 1, "the exit ran once per forward"
        window = captured[0].cast(DType.float32)
        # The shared expert's own gate/up projection, over the same input
        # and this rank's own weight shard: the numbers the merged entry
        # GEMM computed the window's columns from.
        reference = layer.shared_experts.gate_up_proj(x).cast(DType.float32)
        # numpy has no bfloat16; widening it to float32 is exact.
        return _Result(
            out=out.cast(DType.float32).to_numpy(),
            window=window.to_numpy(),
            reference=reference.to_numpy(),
        )


@pytest.mark.skipif(
    accelerator_api() != "hip"
    or not accelerator_architecture_name().startswith("gfx95"),
    reason="the MXFP4 W4A8 chain is CDNA4-only",
)
@pytest.mark.compile_on_gpu(
    "moe.single.group.router and the routing glue are staged as undeclared"
    " custom ops the interpreter can't bind"
)
def test_w4a8_tp2_window_fold_matches_the_standalone_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The glued window reaches the exit and matches the exit chain's run.

    Both arms run the production layer at TP2 on two devices over one
    fixture; only the exit differs, so their outputs must agree to within
    the rounding the fused op's own multi-GPU test bounds. The window the
    exit received must also be the shared expert's own gate/up for this
    rank, narrowed to bfloat16 the way the glue narrows it.
    """
    weights, packed = _random_fixture(seed=5)

    folded = _run(weights, packed, fold=True, monkeypatch=monkeypatch)
    chained = _run(weights, packed, fold=False, monkeypatch=monkeypatch)

    assert folded.out.shape == chained.out.shape == (ROWS, HIDDEN)
    assert np.all(np.isfinite(folded.out))
    assert folded.window is not None and folded.reference is not None
    assert chained.window is not None and chained.reference is not None
    assert folded.window.shape == chained.window.shape == folded.reference.shape

    # The fold's window is the shared expert's own gate/up: every column of
    # it must be, not just the ones a per-column write bound would reach.
    window_error = _rel_error(folded.window, folded.reference)
    # Both exits narrow the same float32 window to bfloat16 -- the glue in
    # place, the chain by a cast of its own -- so they read the same numbers.
    assert _rel_error(folded.window, chained.window) == 0.0, (
        "the fused and standalone exits read different windows"
    )
    exit_error = _rel_error(folded.out, chained.out)
    print(
        f"window vs the shared expert's own gate/up: {window_error:.2e}"
        f" relative; fold vs standalone exit: {exit_error:.2e} relative"
    )
    assert window_error <= MAX_WINDOW_REL_ERROR, (
        f"the window the exit collective received is {window_error:.2e}"
        " relative off the shared expert's own gate/up projection"
    )
    assert exit_error <= MAX_EXIT_REL_ERROR, (
        f"the fused window handoff is {exit_error:.2e} relative off the"
        " standalone exit chain that narrows the same window"
    )
