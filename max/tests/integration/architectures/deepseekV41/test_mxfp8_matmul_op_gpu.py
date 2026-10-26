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
"""Tests ``mo.matmul.mxfp8.cluster_splitk`` in a graph against the MXFP8
matmul it replaces, bit for bit, at every DeepSeek-V4.1 weight shape.

:func:`mxfp8_linear` builds the op as the model does: the dynamic MXFP8
quantize of a bf16 activation, the weight scales interleaved, and the custom
op. On B200 the op runs cluster split-K where its dispatch takes the shape,
else the swap-AB decode GEMM at the decode M of its table's rows, the
pair-CTA GEMM from the prefill M its other table gives each weight shape,
and the block-scaled matmul otherwise. :func:`_matmul_float8_mxfp8` on the
same graph inputs is the reference. The M values reach each of the four:
one row for split-K; the decode M 33, 48, 65, 96, 144, 161 and 255 for the
decode GEMM in its rows (every config it picks: one token tile and several,
one pair a tile, two and three pairs, the persistent kernel) and for
split-K where it keeps ``wq_a_wkv`` to M 160 and ``gate_up`` to M 64; the
row before each prefill entry, and ``wq_b`` at 255, for the block-scaled
matmul; the entry itself and 8312 (the serving prefill chunk, 8168 tokens
and 144 decode rows) for the pair-CTA GEMM; and 2053, a ragged M on either
side of the entries.

Every activation block is ``q * 2**t`` with ``q`` in ``{0, +-64, +-128,
+-448}``, at least one ``+-448``, so the quantize's scale is exactly ``2**t``
and the E4M3 values are ``q`` exactly. Weights are integers in ``[-2, 2]``
with power-of-two scales. Every product and partial sum is then an exact
float32, so each kernel's bits equal the reference's whatever order it adds
K in.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer, float32_array_to_buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn.linear import Linear
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.quant_ops import _matmul_float8_mxfp8
from max.pipelines.architectures.deepseekV41.layers.mxfp8_matmul import (
    mxfp8_linear,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    MXFP8_SPLITK_ENV,
)

_BLOCK = 32
"""Elements per E8M0 scale."""

_SHAPES = {
    "wq_a_wkv": (1792, 5120),
    "wq_b": (32768, 1280),
    "index_wq_b": (4096, 1280),
    "wo_b": (5120, 8192),
    "gate_up": (4608, 5120),
    "down": (5120, 2304),
}
"""The weight shapes ``(N, K)``."""

_FIRST_M = {
    "wq_a_wkv": 8168,
    "wq_b": 512,
    "index_wq_b": 3072,
    "wo_b": 3072,
    "gate_up": 2048,
    "down": 3072,
}
"""The first M at which the op takes the pair-CTA prefill GEMM, per weight
shape: ``B200_PREFILL_PICKS`` in ``Kernels/lib/mxfp8_gemm/mxfp8_gemm.mojo``.
Its decode rows are ``B200_DECODE_PICKS`` there."""

_MS = (1, 33, 48, 65, 96, 144, 161, 255, 2053, 8312)

_E4M3 = np.array([0xC0, 0xB8, 0x00, 0x38, 0x40], dtype=np.uint8)
"""E4M3 encodings of the weights' integers, -2 to 2."""


def _cases() -> Iterator[tuple[str, int]]:
    for name in _SHAPES:
        ms = set(_MS) | {_FIRST_M[name] - 1, _FIRST_M[name]}
        for m in sorted(ms):
            yield name, m


def _quant_config() -> QuantConfig:
    """The model's MXFP8 config: a dynamic activation scale and a static
    weight scale per 32 elements."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _BLOCK),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, _BLOCK),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.MXFP8,
    )


def _activations(
    rng: np.random.Generator, m: int, k: int
) -> npt.NDArray[np.float32]:
    """``[m, k]`` activations whose MXFP8 quantization is exact."""
    blocks = (m, k // _BLOCK)
    q = rng.choice(
        np.array([0, 64, -64, 128, -128, 448, -448], dtype=np.float32),
        size=(*blocks, _BLOCK),
    )
    q[..., 0] = rng.choice(np.array([448, -448], dtype=np.float32), size=blocks)
    t = rng.integers(-8, -5, size=(*blocks, 1)).astype(np.float32)
    return (q * np.exp2(t)).reshape(m, k)


def _weight_data(
    rng: np.random.Generator, n: int, k: int
) -> dict[str, WeightData]:
    """The layer's weight: integers in ``[-2, 2]`` as E4M3, and its scales,
    powers of two in ``[1, 4]`` as E8M0."""
    codes = _E4M3[rng.integers(-2, 3, size=(n, k)) + 2]
    scales = (127 + rng.integers(0, 3, size=(n, k // _BLOCK))).astype(np.uint8)
    return {
        "weight": WeightData(
            Buffer.from_numpy(codes).view(DType.float8_e4m3fn, [n, k]),
            "weight",
            DType.float8_e4m3fn,
            Shape([n, k]),
        ),
        "weight_scale": WeightData(
            Buffer.from_numpy(scales).view(
                DType.float8_e8m0fnu, [n, k // _BLOCK]
            ),
            "weight_scale",
            DType.float8_e8m0fnu,
            Shape([n, k // _BLOCK]),
        ),
    }


@dataclass
class _Compiled:
    model: Model
    k: int


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


@pytest.fixture(scope="module")
def compiled(session: InferenceSession) -> dict[str, _Compiled]:
    """One graph per weight shape, compiled once: the op's output and the
    reference's."""
    out = {}
    with pytest.MonkeyPatch.context() as env:
        # The op is on by default; pinned, so the compare cannot pass with
        # the reference on both sides.
        env.setenv(MXFP8_SPLITK_ENV, "1")
        for i, (name, (n, k)) in enumerate(_SHAPES.items()):
            layer = Linear(
                in_dim=k,
                out_dim=n,
                dtype=DType.float8_e4m3fn,
                device=DeviceRef.GPU(),
                quant_config=_quant_config(),
            )
            layer.load_state_dict(
                _weight_data(np.random.default_rng(100 + i), n, k)
            )
            with Graph(
                f"mxfp8_matmul_op_{name}",
                input_types=[
                    TensorType(DType.bfloat16, ["m", k], device=DeviceRef.GPU())
                ],
            ) as graph:
                x = graph.inputs[0].tensor
                assert layer.weight_scale is not None
                graph.output(
                    mxfp8_linear(layer, x),
                    _matmul_float8_mxfp8(
                        x, layer.weight.to(x.device), layer.weight_scale
                    ),
                )
            assert "mo.matmul.mxfp8.cluster_splitk" in str(graph)
            out[name] = _Compiled(
                session.load(graph, weights_registry=layer.state_dict()), k
            )
    return out


@pytest.fixture(scope="module")
def activations() -> dict[int, npt.NDArray[np.float32]]:
    """Activations for the largest M at each K; a smaller M takes the first
    rows."""
    top = max(m for _, m in _cases())
    ks = sorted({k for _, k in _SHAPES.values()})
    rng = np.random.default_rng(7)
    return {k: _activations(rng, top, k) for k in ks}


@pytest.mark.parametrize("name, m", list(_cases()))
def test_op_matches_reference(
    compiled: dict[str, _Compiled],
    activations: dict[int, npt.NDArray[np.float32]],
    name: str,
    m: int,
) -> None:
    case = compiled[name]
    device = case.model.input_devices[0]
    x = float32_array_to_buffer(
        activations[case.k][:m], dtype=DType.bfloat16, device=device
    )
    op, ref = (
        np.from_dlpack(out.to(CPU()).view(DType.uint16))
        for out in case.model.execute(x)
    )
    assert op.shape == ref.shape == (m, _SHAPES[name][0])
    # The data must reach the outputs, or the compare proves nothing.
    assert np.count_nonzero(ref) > ref.size // 2
    differ = np.flatnonzero(op != ref)
    assert differ.size == 0, (
        f"{name} M={m}: {differ.size} of {op.size} outputs differ, first at"
        f" {np.unravel_index(differ[0], op.shape)}"
    )
