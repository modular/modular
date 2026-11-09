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
"""In-graph NVFP4 quantization-error measurement on a single linear weight.

``NVFP4TensorQuantizer.report_error`` adds measuring ops next to the
quantization and prints ``[sum_sq_error, sum_sq_source, max_abs_error]``.
This compiles a one-weight module, runs it once, and checks the printed
numbers against a host recomputation from the *same* packed weight and
scales the graph returned, so the report describes the served weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from max import driver
from max import functional as F
from max._entrypoints.cli.quantize import (
    insert_quantized_linears,
    on_the_fly_config,
)
from max.driver import Accelerator, accelerator_count
from max.driver.buffer import load_max_buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.nn.common_layers import quant_ops
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.quant_tensor import (
    MAX_E2M1,
    MAX_E4M3,
    NVFP4TensorQuantizer,
)
from max.nn.kernels import nvfp4_dequant
from max.nn.module import Module, module_dataclass
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    dequantize_nvfp4,
)
from max.tensor import Tensor
from max.tensor.tensor import TensorType

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU"
)

_N, _K = 128, 256
# The dump is named after the weight, which ``compile`` names after its path.
_LABEL = "weight"


@module_dataclass
class _OneWeight(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizer = NVFP4TensorQuantizer(self.weight, report_error=True)
        quantized = quantizer.quant_tensor
        # Returning the leaves keeps them live and lets the test recompute
        # the error from exactly what the graph quantized.
        return (
            quantized.weight,
            quantized.weight_scale,
            quantized.weight_scale_2,
        )


def _find_dump(directory: Path, label: str = _LABEL) -> Path:
    matches = sorted(directory.glob(f"*quant_error*{label}*.max"))
    assert matches, sorted(p.name for p in directory.iterdir())
    return matches[0]


def test_in_graph_error_matches_host_dequantize(tmp_path: Path) -> None:
    device = Accelerator()
    rng = np.random.default_rng(7)
    w_f32 = rng.standard_normal((_N, _K)).astype(np.float32)
    weight = Tensor(storage=driver.Buffer.from_numpy(w_f32).to(device)).cast(
        DType.bfloat16
    )
    source = weight.cast(DType.float32).to_numpy()

    with F.lazy():
        model = _OneWeight(weight=Tensor.zeros([_N, _K], dtype=DType.bfloat16))

    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = model.compile(
        TensorType(DType.bfloat16, [1, _K], device=device),
        weights={"weight": weight.driver_tensor},
        session=session,
    )
    x = Tensor.zeros([1, _K], dtype=DType.bfloat16, device=device)
    packed, scales, decode_scale = compiled(x)

    packed_np = packed.to_numpy()
    scales_np = scales.driver_tensor.view(DType.uint8).to_numpy()
    # The host helper broadcasts scales against ``[N, K/16, 16]`` blocks, so
    # it takes them with a trailing singleton axis.
    restored = dequantize_nvfp4(
        packed_np,
        scales_np[..., None],
        float(decode_scale.to_numpy()),
        FP4Format.NVFP4,
    )
    error = restored.astype(np.float64) - source.astype(np.float64)
    expected = np.array(
        [
            (error**2).sum(),
            (source.astype(np.float64) ** 2).sum(),
            np.abs(error).max(),
        ]
    )

    got = load_max_buffer(_find_dump(tmp_path)).to_numpy().astype(np.float64)
    assert got.shape == (3,)
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    # Quantization is lossy but small: relative energy of the error.
    assert 0 < got[0] / got[1] < 0.05


def _bf16_weight(device: driver.Device, values: np.ndarray) -> Tensor:
    return Tensor(storage=driver.Buffer.from_numpy(values).to(device)).cast(
        DType.bfloat16
    )


def _host_dequant(
    packed: Tensor, scales: Tensor, decode_scale: Tensor
) -> np.ndarray:
    scales_np = scales.driver_tensor.view(DType.uint8).to_numpy()
    return dequantize_nvfp4(
        packed.to_numpy(),
        scales_np[..., None],
        float(decode_scale.to_numpy()),
        FP4Format.NVFP4,
    )


# --- Dequant op vs. the matmul's own decode ---------------------------------

_M = 32
_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _fp4_exact_activations(rng: np.random.Generator) -> np.ndarray:
    """Activations the matmul's dynamic FP4 quantization reproduces exactly.

    Each 16-block holds E2M1 values times a power of two and contains a 6.0,
    so its block scale (``amax / 6``) is that power of two, exactly
    representable in E4M3. The matmul then adds no activation error, and
    any remaining difference is the weight decode.
    """
    blocks = _K // 16
    mags = rng.choice(_E2M1_MAGNITUDES, size=(_M, blocks, 16))
    mags[..., 0] = 6.0
    exponent = rng.integers(-2, 2, size=(_M, blocks, 1))
    signs = rng.choice([-1.0, 1.0], size=(_M, blocks, 16))
    return (mags * signs * (2.0**exponent)).reshape(_M, _K).astype(np.float32)


@module_dataclass
class _MatmulAndDequant(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizer = NVFP4TensorQuantizer(self.weight)
        quantized = quantizer.quant_tensor
        restored = nvfp4_dequant(
            quantized.weight,
            quantized.weight_scale,
            quantized.weight_scale_2,
            out_type=DType.float32,
        )
        return (
            quant_ops.matmul(x, quantizer),
            restored,
            quantized.weight,
            quantized.weight_scale,
            quantized.weight_scale_2,
        )


def test_dequant_op_matches_matmul_decode() -> None:
    """``x @ dequant(packed).T`` equals the quantized matmul on the same weight.

    The measurement decodes with ``nvfp4_dequant``; serving decodes inside
    the block-scaled matmul. If they disagreed, the report would describe a
    weight other than the one served.
    """
    device = Accelerator()
    rng = np.random.default_rng(11)
    w_f32 = rng.standard_normal((_N, _K)).astype(np.float32)
    x_np = _fp4_exact_activations(rng)

    with F.lazy():
        model = _MatmulAndDequant(
            weight=Tensor.zeros([_N, _K], dtype=DType.bfloat16)
        )
    compiled = model.compile(
        TensorType(DType.bfloat16, [_M, _K], device=device),
        weights={"weight": _bf16_weight(device, w_f32).driver_tensor},
        session=InferenceSession(devices=[device]),
    )
    out, restored, packed, scales, decode_scale = compiled(
        _bf16_weight(device, x_np)
    )

    restored_np = restored.to_numpy().astype(np.float64)
    # The op and the host reference decode the same bytes identically.
    np.testing.assert_array_equal(
        restored_np.astype(np.float32),
        _host_dequant(packed, scales, decode_scale),
    )

    expected = x_np.astype(np.float64) @ restored_np.T
    got = out.cast(DType.float32).to_numpy().astype(np.float64)
    # Output is bf16 (8 bits of mantissa); accumulation order differs.
    atol = 2e-2 * np.abs(expected).mean()
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=atol)


# --- Fused group -------------------------------------------------------------


@module_dataclass
class _FusedPair(Module[[Tensor], tuple[Tensor, ...]]):
    gate: Tensor
    up: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        members = []
        for weight in (self.gate, self.up):
            members.append(NVFP4TensorQuantizer(weight, report_error=True))
        fused = quant_ops.concat_nvfp4(*members)
        return fused.weight, fused.weight_scale, fused.weight_scale_2


def test_fused_group_reports_error_under_the_shared_scale(
    tmp_path: Path,
) -> None:
    """Each member's report uses the group's pinned scale, not its own.

    ``up`` is four times larger than ``gate``, so the shared amax is
    ``up``'s. The fused tensor keeps one ``weight_scale_2``; the report for
    ``gate`` must be its error under that scale, i.e. against the row slice
    of the fused leaves that is actually served.
    """
    device = Accelerator()
    rng = np.random.default_rng(3)
    gate_f32 = rng.standard_normal((_N, _K)).astype(np.float32)
    up_f32 = 4.0 * rng.standard_normal((_N, _K)).astype(np.float32)
    gate = _bf16_weight(device, gate_f32)
    up = _bf16_weight(device, up_f32)

    with F.lazy():
        model = _FusedPair(
            gate=Tensor.zeros([_N, _K], dtype=DType.bfloat16),
            up=Tensor.zeros([_N, _K], dtype=DType.bfloat16),
        )
    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = model.compile(
        TensorType(DType.bfloat16, [1, _K], device=device),
        weights={
            "gate": gate.driver_tensor,
            "up": up.driver_tensor,
        },
        session=session,
    )
    packed, scales, decode_scale = compiled(
        Tensor.zeros([1, _K], dtype=DType.bfloat16, device=device)
    )

    sources = {
        "gate": gate.cast(DType.float32).to_numpy().astype(np.float64),
        "up": up.cast(DType.float32).to_numpy().astype(np.float64),
    }
    shared_amax = max(np.abs(a).max() for a in sources.values())
    np.testing.assert_allclose(
        float(decode_scale.to_numpy()),
        shared_amax / (MAX_E2M1 * MAX_E4M3),
        rtol=1e-6,
    )

    restored = _host_dequant(packed, scales, decode_scale).astype(np.float64)
    assert restored.shape == (2 * _N, _K)
    rows = {"gate": slice(0, _N), "up": slice(_N, 2 * _N)}
    sse = {}
    for name, source in sources.items():
        error = restored[rows[name]] - source
        expected = np.array(
            [(error**2).sum(), (source**2).sum(), np.abs(error).max()]
        )
        got = (
            load_max_buffer(_find_dump(tmp_path, name))
            .to_numpy()
            .astype(np.float64)
        )
        np.testing.assert_allclose(got, expected, rtol=1e-4, err_msg=name)
        sse[name] = got[0] / got[1]

    # Both members report nonzero error; the equality checks above are what
    # pin each report to the served (group-scaled) leaves.
    assert sse["gate"] > 0 and sse["up"] > 0


# --- Flag on vs. off ---------------------------------------------------------


@module_dataclass
class _TwoLinears(Module[[Tensor], tuple[Tensor, ...]]):
    """``x -> quantized linear -> quantized linear``, returning leaves too."""

    w0: Tensor
    w1: Tensor
    report: ClassVar[bool] = False

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizers = [
            NVFP4TensorQuantizer(w, report_error=self.report)
            for w in (self.w0, self.w1)
        ]
        hidden = quant_ops.matmul(x, quantizers[0])
        out = quant_ops.matmul(hidden, quantizers[1])
        leaves: list[Tensor] = []
        for quantizer in quantizers:
            quantized = quantizer.quant_tensor
            leaves += [
                quantized.weight,
                quantized.weight_scale,
                quantized.weight_scale_2,
            ]
        return (out, *leaves)


@module_dataclass
class _TwoLinearsReporting(_TwoLinears):
    report: ClassVar[bool] = True


def _run_two_linears(
    model_cls: type[_TwoLinears],
    device: driver.Device,
    weights: dict[str, Tensor],
    x: Tensor,
    session: InferenceSession,
) -> list[np.ndarray]:
    with F.lazy():
        model = model_cls(
            w0=Tensor.zeros([_K, _K], dtype=DType.bfloat16),
            w1=Tensor.zeros([_N, _K], dtype=DType.bfloat16),
        )
    compiled = model.compile(
        TensorType(DType.bfloat16, [_M, _K], device=device),
        weights={name: w.driver_tensor for name, w in weights.items()},
        session=session,
    )
    results = compiled(x)
    arrays = []
    for result in results:
        if result.rank == 0:
            # A rank-0 buffer cannot be re-viewed at another element size,
            # and the float32 decode scale is exact under numpy anyway.
            arrays.append(np.asarray(result.to_numpy()).copy())
            continue
        # Compare bytes, not values: float8 scales have no numpy dtype, and
        # bit-identity is the claim anyway.
        view = result.driver_tensor.view(
            DType.uint8 if result.dtype != DType.bfloat16 else DType.uint16
        )
        arrays.append(view.to_numpy().copy())
    return arrays


def test_flag_does_not_change_served_output(tmp_path: Path) -> None:
    """Logits and quantized leaves are bit-identical with the flag on or off.

    The measuring ops are extra consumers of the cached quantization, so
    turning the flag on must not move a single served bit.
    """
    device = Accelerator()
    rng = np.random.default_rng(5)
    weights = {
        "w0": _bf16_weight(
            device, rng.standard_normal((_K, _K)).astype(np.float32)
        ),
        "w1": _bf16_weight(
            device, rng.standard_normal((_N, _K)).astype(np.float32)
        ),
    }
    x = _bf16_weight(device, rng.standard_normal((_M, _K)).astype(np.float32))

    plain = _run_two_linears(
        _TwoLinears, device, weights, x, InferenceSession(devices=[device])
    )

    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    flagged = _run_two_linears(
        _TwoLinearsReporting, device, weights, x, session
    )

    # The flag actually did something: both layers reported.
    for label in ("w0", "w1"):
        _find_dump(tmp_path, label)

    assert len(plain) == len(flagged) == 7
    for index, (a, b) in enumerate(zip(plain, flagged, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"output {index}")


# --- Labels survive insert -> compile ---------------------------------------


def test_inserted_layers_report_under_their_tree_paths(tmp_path: Path) -> None:
    """A config with ``report_error=True`` reaches the dumps.

    ``compile`` rebuilds the declared module before tracing, so this is the
    path where a flag stored on the quantizer would be lost. Gate and up
    are fused into one matmul and still get one entry each, named after
    their weights.
    """
    device = Accelerator()
    rng = np.random.default_rng(13)
    hidden = ff = _K
    with F.lazy():
        mlp = MLP(hidden, ff)
        insert_quantized_linears(
            mlp,
            [r"^gate_proj$", r"^up_proj$", r"^down_proj$"],
            on_the_fly_config("nvfp4", report_error=True),
        )

    weights = {
        f"{name}.weight": _bf16_weight(
            device, rng.standard_normal((out, inp)).astype(np.float32)
        ).driver_tensor
        for name, (out, inp) in {
            "gate_proj": (ff, hidden),
            "up_proj": (ff, hidden),
            "down_proj": (hidden, ff),
        }.items()
    }
    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = mlp.compile(
        TensorType(DType.bfloat16, [_M, hidden], device=device),
        weights=weights,
        session=session,
    )
    compiled(
        _bf16_weight(
            device, rng.standard_normal((_M, hidden)).astype(np.float32)
        )
    )

    for label in ("gate_proj", "up_proj", "down_proj"):
        stats = load_max_buffer(_find_dump(tmp_path, label)).to_numpy()
        assert stats.shape == (3,)
        assert stats[0] > 0 and stats[1] > 0, label
