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
"""Executes the FP8 grouped wo_a and FP8 wo_b path on one and two GPUs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from max import functional as F
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layer_resolver import (
    resolve,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    Attention,
    RaggedRows,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.block import (
    DecoderLayer,
    LayerPlan,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import (
    mhc_identity_writeback,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.norm import RMSNorm
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
    V41RotaryEmbedding,
    apply_rope_tail,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded
from runtime_config import text_config


class _AttentionOutput(Module[..., object]):
    """An attention block built for its output projection alone."""

    def __init__(self, mesh: DeviceMesh) -> None:
        super().__init__()
        text = text_config()
        text.num_attention_heads = 64
        text.o_groups = 8
        # A group rank that is a whole 128-row scale tile, like the
        # checkpoint's own 1024: the grouped wo_a interleaves its B
        # scales over whole 128-row blocks.
        text.o_lora_rank = 128
        interface = V41Config.from_hf({"text_config": vars(text)})
        kv = build_kv_params(
            interface,
            KVCacheConfig(kv_cache_page_size=128),
            [DeviceRef.GPU(i) for i in range(mesh.num_devices)],
        )
        config = DeepseekV41Config(
            dtype=DType.bfloat16,
            kv_params=kv,
            devices=[DeviceRef.GPU(i) for i in range(mesh.num_devices)],
            max_seq_len=256,
            **DeepseekV41Config.resolve_text_fields(text),
        )
        self.mesh = mesh
        self.attn = Attention(
            config,
            resolve(config)[0],
            mesh,
            rope=RopeParams(64, 32.0, 1.0, 4096),
        )


class _Projection(_AttentionOutput):
    def forward(self, values: Tensor) -> Tensor:
        values = F.distributed_broadcast(values, self.mesh)
        values = values.to(DeviceMapping(self.mesh, (Sharded(1),)))
        output = self.attn._output_projection(values)
        return output.local_shards[0].cast(DType.float32)


class _FoldedProjection(_AttentionOutput):
    """The projection twice: rotation folded into ``wo_a``'s quantize, and
    rotated first by the chain."""

    def forward(self, values: Tensor, cos_sin: Tensor) -> tuple[Tensor, Tensor]:
        values = F.distributed_broadcast(values, self.mesh)
        cos_sin = F.distributed_broadcast(cos_sin, self.mesh)
        values = values.to(DeviceMapping(self.mesh, (Sharded(1),)))
        folded = self.attn._output_projection(values, (cos_sin, 512))
        chain = self.attn._output_projection(
            apply_rope_tail(values, cos_sin, 64, inverse=True)
        )
        return (
            folded.local_shards[0].cast(DType.float32),
            chain.local_shards[0].cast(DType.float32),
        )


def _buffer(raw: np.ndarray, dtype: DType) -> Buffer:
    return Buffer.from_numpy(np.ascontiguousarray(raw)).view(dtype)


def _mesh(degree: int) -> DeviceMesh:
    assert accelerator_count() >= 2, (
        "This target requires two distinct visible GPUs"
    )
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(degree)), (degree,), ("tp",)
    )


def _weights(
    module: Module[..., object], dense_woa: bool = False
) -> dict[str, Buffer]:
    weights: dict[str, Buffer] = {}
    for name, parameter in module.parameters:
        shape = tuple(int(dim) for dim in parameter.shape)
        if parameter.dtype == DType.bfloat16:
            raw = np.zeros(shape, np.uint16)
        elif parameter.dtype == DType.float32:
            raw = np.zeros(shape, np.float32)
        else:
            raw = np.zeros(shape, np.uint8)
        weights[name] = _buffer(raw, parameter.dtype)
    # Sparse one-hot E4M3 rows keep this independent reference exact
    # through the MXFP8 quantize (every activation block's amax is a
    # power of two, so the ceil-pow2 scale and the codes are exact),
    # while selecting different local groups on each rank.
    woa = np.zeros((8 * 128, 4096), np.uint8)
    for group in range(8):
        for row in range(128):
            woa[group * 128 + row, (row * 37 + group * 11) % 4096] = np.uint8(
                0xB8 if row % 2 else 0x38
            )
    if dense_woa:
        # Every column of every group is read, so a wrong rotation anywhere
        # in a head's tail moves the output (finite codes only).
        rng = np.random.default_rng(0)
        woa = (
            rng.integers(0, 0x70, woa.shape)
            | (rng.integers(0, 2, woa.shape) << 7)
        ).astype(np.uint8)
    weights["attn.wo_a.weight"] = _buffer(woa, DType.float8_e4m3fn)
    weights["attn.wo_a.scale"] = _buffer(
        np.full((8 * 128 // 32, 4096 // 32), 127, np.uint8),
        DType.float8_e8m0fnu,
    )
    wob = np.zeros((64, 8 * 128), np.uint8)
    wob[np.arange(64), (np.arange(64) * 13 + 7) % (8 * 128)] = np.uint8(0x38)
    weights["attn.wo_b.weight"] = _buffer(wob, DType.float8_e4m3fn)
    weights["attn.wo_b.scale"] = _buffer(
        np.full((2, 8 * 128 // 32), 127, np.uint8), DType.float8_e8m0fnu
    )
    return weights


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def projection(
    request: pytest.FixtureRequest,
) -> tuple[Callable[..., Any], int]:
    degree = int(request.param)
    with F.lazy():
        module = _Projection(_mesh(degree))
    run = module.compile(
        TensorType(DType.bfloat16, ["rows", 64, 512], DeviceRef.GPU(0)),
        weights=_weights(module),
    )
    return run, degree


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def folded_projection(request: pytest.FixtureRequest) -> Callable[..., Any]:
    with F.lazy():
        module = _FoldedProjection(_mesh(int(request.param)))
    return module.compile(
        TensorType(DType.bfloat16, ["rows", 64, 512], DeviceRef.GPU(0)),
        TensorType(DType.float32, ["rows", 32, 2], DeviceRef.GPU(0)),
        weights=_weights(module, dense_woa=True),
    )


@pytest.mark.parametrize("rows", [1, 17])
def test_output_projection_tp(
    projection: tuple[Callable[..., Any], int], rows: int
) -> None:
    run, _ = projection
    cols = np.arange(64 * 512)
    base = np.power(2.0, (cols // 7) % 3).astype(np.float32)
    base *= np.where((cols // 13) % 2, -1.0, 1.0)
    values = np.stack([base * (1 + (row % 2)) for row in range(rows)]).reshape(
        rows, 64, 512
    )
    flat = values.reshape(rows, 8, 4096)
    grouped = np.empty((rows, 8, 128), np.float32)
    for group in range(8):
        for row in range(128):
            grouped[:, group, row] = flat[
                :, group, (row * 37 + group * 11) % 4096
            ] * (-1.0 if row % 2 else 1.0)
    expected = grouped.reshape(rows, 8 * 128)[
        :, (np.arange(64) * 13 + 7) % (8 * 128)
    ]
    values_bf16 = _buffer(
        (values.view(np.uint32) >> 16).astype(np.uint16), DType.bfloat16
    )
    actual = run(Tensor.from_dlpack(values_bf16).to(Accelerator(0)))
    np.testing.assert_array_equal(actual.to_numpy(), expected)


@pytest.mark.parametrize("rows", [1, 17])
def test_folded_inverse_rope_matches_rotate_first(
    folded_projection: Callable[..., Any], rows: int
) -> None:
    """The rope-quantize fold reads what the chain's rotated rows quantize to."""
    rng = np.random.default_rng(rows)
    values = rng.standard_normal((rows, 64, 512)).astype(np.float32)
    values_bf16 = _buffer(
        (values.view(np.uint32) >> 16).astype(np.uint16), DType.bfloat16
    )
    inv_freqs = V41RotaryEmbedding(
        RopeParams(64, 32.0, 1.0, 4096), 10000.0, 1.0
    ).inv_freqs()
    angles = np.outer(np.arange(rows, dtype=np.float32) * 97, inv_freqs)
    cos_sin = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    folded, chain = folded_projection(
        Tensor.from_dlpack(values_bf16).to(Accelerator(0)),
        Tensor.from_dlpack(
            _buffer(cos_sin.astype(np.float32), DType.float32)
        ).to(Accelerator(0)),
    )
    assert np.any(chain.to_numpy() != 0)
    np.testing.assert_array_equal(folded.to_numpy(), chain.to_numpy())


class _PassFFN(Module[..., Tensor]):
    """The boundary's feed-forward stand-in: passes its rows through.

    Takes the same arguments the real feed-forward's forward does, the EP
    communication buffers and the host row count among them, and ignores
    every one but the rows.
    """

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        comm_buffers: object = None,
        total_rows: Tensor | None = None,
    ) -> Tensor:
        del mask, comm_buffers, total_rows
        return x


# The fused mHC boundary needs a hidden width in whole 1024s.
_BOUNDARY_HIDDEN = 1024


class _Boundary(Module[[Tensor], tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        layer = object.__new__(DecoderLayer)
        Module.__init__(layer)
        layer.hc_mult = 4
        layer.norm_eps = 1e-20
        layer.plan = LayerPlan((), {}, "")
        layer.attn_norm = RMSNorm(_BOUNDARY_HIDDEN)
        layer.ffn_norm = RMSNorm(_BOUNDARY_HIDDEN)
        layer.ffn_norm.weight = F.full(
            [_BOUNDARY_HIDDEN], 2.0, dtype=DType.bfloat16, device=Accelerator(0)
        )
        monkeypatch.setattr(layer, "ffn", _PassFFN(), raising=False)
        monkeypatch.setattr(layer, "_attention", lambda x, site: (x, []))
        for role in ("attn", "ffn"):
            setattr(
                layer,
                f"hc_{role}_fn",
                Tensor.zeros([24, 4 * _BOUNDARY_HIDDEN], dtype=DType.float32),
            )
            setattr(
                layer,
                f"hc_{role}_base",
                Tensor.zeros([24], dtype=DType.float32),
            )
            setattr(
                layer,
                f"hc_{role}_scale",
                Tensor.zeros([3], dtype=DType.float32),
            )
        self.layer = layer

    def forward(self, stream: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        total = stream.shape[0]
        pre = F.broadcast_to(
            F.constant(
                np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
                DType.float32,
                device=stream.mapping,
            ),
            [total, 4],
        )
        y, post, comb = mhc_identity_writeback(stream)
        zero = F.full([total], 0, dtype=DType.int32, device=stream.mapping)
        offsets = F.shape_to_tensor([0, total]).cast(DType.uint32)
        rows = RaggedRows(
            offsets,
            zero,
            zero,
            zero,
            F.shape_to_tensor([total]).cast(DType.int32),
        )
        mixed, pre, result, _, _, _ = self.layer(
            stream, pre, y, post, comb, rows, zero, zero, zero, [], [], []
        )
        return mixed, pre, result


def test_compiled_decoder_boundary_normalizes_ffn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with default_device(Accelerator(0)):
        module = _Boundary(monkeypatch)
    run = module.compile(
        TensorType(
            DType.bfloat16, ["rows", 4, _BOUNDARY_HIDDEN], DeviceRef.GPU(0)
        )
    )
    values = (
        np.random.default_rng(73)
        .normal(size=(3, 4, _BOUNDARY_HIDDEN))
        .astype(np.float32)
    )
    bits = values.view(np.uint32)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    inputs = Tensor.from_dlpack(Buffer.from_numpy(bits).view(DType.bfloat16))
    mixed, pre, result = run(inputs.to(Accelerator(0)))
    stream = mixed.cast(DType.float32).to_numpy()
    collapse = np.sum(stream * pre.to_numpy()[:, :, None], axis=1)
    bits = collapse.astype(np.float32).view(np.uint32)
    collapse = (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16).view(
        np.float32
    )
    expected = (
        2.0
        * collapse
        / np.sqrt(np.mean(collapse * collapse, axis=-1, keepdims=True) + 1e-20)
    )
    # One BF16 rounding at the normalization output.
    bits = expected.astype(np.float32).view(np.uint32)
    expected = (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16).view(
        np.float32
    )
    np.testing.assert_allclose(
        result.cast(DType.float32).to_numpy(),
        expected,
        rtol=1.0 / 128,
        atol=1e-5,
    )
