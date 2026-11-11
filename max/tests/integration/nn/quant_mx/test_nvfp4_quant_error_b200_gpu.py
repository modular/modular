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
"""In-graph NVFP4 quantization-error measurement.

``NVFP4TensorQuantizer.report_error`` adds measuring ops next to the
quantization and prints ``[sum_sq_error, sum_sq_source, max_abs_error]``.
Every graph compile here costs tens of seconds, so the checks share graphs:

* one graph holds a single weight and a fused gate/up pair, and checks the
  printed numbers against a host recomputation from the *same* packed
  weight and scales the graph returned, the dequant op against the
  matmul's decode, and the fused group's shared scale;
* one MLP is compiled twice, with and without the flag, through
  ``insert_quantized_linears``: outputs must match bit for bit, and only
  the flagged one reports, under the weights' tree paths.
"""

from __future__ import annotations

from pathlib import Path

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
_M = 32
_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


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


def _find_dump(directory: Path, label: str) -> Path:
    matches = sorted(directory.glob(f"*quant_error*{label}*.max"))
    assert matches, sorted(p.name for p in directory.iterdir())
    return matches[0]


def _expected_stats(restored: np.ndarray, source: np.ndarray) -> np.ndarray:
    error = restored.astype(np.float64) - source.astype(np.float64)
    return np.array(
        [
            (error**2).sum(),
            (source.astype(np.float64) ** 2).sum(),
            np.abs(error).max(),
        ]
    )


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
class _SingleAndFused(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor
    gate: Tensor
    up: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        single = NVFP4TensorQuantizer(self.weight, report_error=True)
        quantized = single.quant_tensor
        restored = nvfp4_dequant(
            quantized.weight,
            quantized.weight_scale,
            quantized.weight_scale_2,
            out_type=DType.float32,
        )
        members = [
            NVFP4TensorQuantizer(w, report_error=True)
            for w in (self.gate, self.up)
        ]
        fused = quant_ops.concat_nvfp4(*members)
        # Returning the leaves keeps them live and lets the test recompute
        # the error from exactly what the graph quantized.
        return (
            quant_ops.matmul(x, single),
            restored,
            quantized.weight,
            quantized.weight_scale,
            quantized.weight_scale_2,
            fused.weight,
            fused.weight_scale,
            fused.weight_scale_2,
        )


def test_single_weight_and_fused_group(tmp_path: Path) -> None:
    """Reports match the served leaves; the dequant op matches the matmul.

    Single weight: the printed stats equal a host recomputation from the
    returned packed weight and scales, and ``x @ dequant(packed).T`` equals
    the quantized matmul, so the measurement decodes what serving decodes.

    Fused group: ``up`` is four times larger than ``gate``, so the shared
    amax is ``up``'s. The fused tensor keeps one ``weight_scale_2`` and each
    member's report is its error under that scale, against its row slice of
    the fused leaves.
    """
    device = Accelerator()
    rng = np.random.default_rng(7)
    w_f32 = rng.standard_normal((_N, _K)).astype(np.float32)
    gate_f32 = rng.standard_normal((_N, _K)).astype(np.float32)
    up_f32 = 4.0 * rng.standard_normal((_N, _K)).astype(np.float32)
    x_np = _fp4_exact_activations(rng)
    weight = _bf16_weight(device, w_f32)
    gate = _bf16_weight(device, gate_f32)
    up = _bf16_weight(device, up_f32)

    with F.lazy():
        model = _SingleAndFused(
            weight=Tensor.zeros([_N, _K], dtype=DType.bfloat16),
            gate=Tensor.zeros([_N, _K], dtype=DType.bfloat16),
            up=Tensor.zeros([_N, _K], dtype=DType.bfloat16),
        )
    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = model.compile(
        TensorType(DType.bfloat16, [_M, _K], device=device),
        weights={
            "weight": weight.driver_tensor,
            "gate": gate.driver_tensor,
            "up": up.driver_tensor,
        },
        session=session,
    )
    (
        out,
        restored,
        packed,
        scales,
        decode_scale,
        fused_packed,
        fused_scales,
        fused_decode_scale,
    ) = compiled(_bf16_weight(device, x_np))

    # Single weight: in-graph stats vs host recomputation.
    source = weight.cast(DType.float32).to_numpy()
    host_restored = _host_dequant(packed, scales, decode_scale)
    got = load_max_buffer(_find_dump(tmp_path, "weight")).to_numpy()
    got = got.astype(np.float64)
    assert got.shape == (3,)
    np.testing.assert_allclose(
        got, _expected_stats(host_restored, source), rtol=1e-4
    )
    # Quantization is lossy but small: relative energy of the error.
    assert 0 < got[0] / got[1] < 0.05

    # The dequant op and the host reference decode the same bytes identically.
    restored_np = restored.to_numpy().astype(np.float64)
    np.testing.assert_array_equal(restored_np.astype(np.float32), host_restored)
    # The matmul agrees with the dequantized weight. Output is bf16 (8 bits
    # of mantissa) and accumulation order differs.
    expected = x_np.astype(np.float64) @ restored_np.T
    matmul_out = out.cast(DType.float32).to_numpy().astype(np.float64)
    atol = 2e-2 * np.abs(expected).mean()
    np.testing.assert_allclose(matmul_out, expected, rtol=2e-2, atol=atol)

    # Fused group: shared scale, per-member report under it.
    sources = {
        "gate": gate.cast(DType.float32).to_numpy().astype(np.float64),
        "up": up.cast(DType.float32).to_numpy().astype(np.float64),
    }
    shared_amax = max(np.abs(a).max() for a in sources.values())
    np.testing.assert_allclose(
        float(fused_decode_scale.to_numpy()),
        shared_amax / (MAX_E2M1 * MAX_E4M3),
        rtol=1e-6,
    )
    fused_restored = _host_dequant(
        fused_packed, fused_scales, fused_decode_scale
    ).astype(np.float64)
    assert fused_restored.shape == (2 * _N, _K)
    rows = {"gate": slice(0, _N), "up": slice(_N, 2 * _N)}
    for name, member_source in sources.items():
        member_got = (
            load_max_buffer(_find_dump(tmp_path, name))
            .to_numpy()
            .astype(np.float64)
        )
        np.testing.assert_allclose(
            member_got,
            _expected_stats(fused_restored[rows[name]], member_source),
            rtol=1e-4,
            err_msg=name,
        )
        # Nonzero error; the equality above pins each report to the served
        # (group-scaled) leaves.
        assert member_got[0] > 0 and member_got[1] > 0, name


def test_flag_leaves_output_unchanged_and_names_layers(tmp_path: Path) -> None:
    """A flagged MLP matches an unflagged one bit for bit, and reports.

    The measuring ops are extra consumers of the cached quantization, so
    turning the flag on must not move a single served bit. A config with
    ``report_error=True`` goes through ``insert_quantized_linears`` and
    ``compile``, which rebuilds the declared module before tracing: the path
    where a flag stored on the quantizer would be lost. Gate and up are fused
    into one matmul and still get one dump each, named after their weights.
    """
    device = Accelerator()
    rng = np.random.default_rng(13)
    hidden = ff = _K
    shapes = {
        "gate_proj": (ff, hidden),
        "up_proj": (ff, hidden),
        "down_proj": (hidden, ff),
    }
    weights = {
        f"{name}.weight": _bf16_weight(
            device, rng.standard_normal(shape).astype(np.float32)
        ).driver_tensor
        for name, shape in shapes.items()
    }
    x = _bf16_weight(
        device, rng.standard_normal((_M, hidden)).astype(np.float32)
    )

    def run(report_error: bool, session: InferenceSession) -> np.ndarray:
        with F.lazy():
            mlp = MLP(hidden, ff)
            insert_quantized_linears(
                mlp,
                [r"^gate_proj$", r"^up_proj$", r"^down_proj$"],
                on_the_fly_config("nvfp4", report_error=report_error),
            )
        compiled = mlp.compile(
            TensorType(DType.bfloat16, [_M, hidden], device=device),
            weights=weights,
            session=session,
        )
        out = compiled(x)
        # Compare bytes: bit-identity is the claim.
        return out.driver_tensor.view(DType.uint16).to_numpy().copy()

    plain = run(False, InferenceSession(devices=[device]))
    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    flagged = run(True, session)

    np.testing.assert_array_equal(plain, flagged)
    for label in shapes:
        stats = load_max_buffer(_find_dump(tmp_path, label)).to_numpy()
        assert stats.shape == (3,)
        assert stats[0] > 0 and stats[1] > 0, label
