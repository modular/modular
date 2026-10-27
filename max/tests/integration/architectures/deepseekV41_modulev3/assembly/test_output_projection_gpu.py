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
"""Executes the BF16 grouped wo_a and FP8 wo_b path on one and two GPUs."""

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
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded
from runtime_config import text_config


class _Projection(Module[[Tensor], Tensor]):
    def __init__(self, mesh: DeviceMesh) -> None:
        super().__init__()
        text = text_config()
        text.num_attention_heads = 64
        text.o_groups = 8
        text.o_lora_rank = 32
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

    def forward(self, values: Tensor) -> Tensor:
        values = F.distributed_broadcast(values, self.mesh)
        values = values.to(DeviceMapping(self.mesh, (Sharded(1),)))
        output = self.attn._output_projection(values)
        return output.local_shards[0].cast(DType.float32)


def _buffer(raw: np.ndarray, dtype: DType) -> Buffer:
    return Buffer.from_numpy(np.ascontiguousarray(raw)).view(dtype)


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def projection(
    request: pytest.FixtureRequest,
) -> tuple[Callable[..., Any], int]:
    assert accelerator_count() >= 2, (
        "This target requires two distinct visible GPUs"
    )
    degree = int(request.param)
    mesh = DeviceMesh(
        tuple(Accelerator(i) for i in range(degree)), (degree,), ("tp",)
    )
    with F.lazy():
        module = _Projection(mesh)
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
    # Sparse one-hot rows keep this independent reference exact in BF16
    # and FP8, while selecting different local groups on each rank.
    woa = np.zeros((8 * 32, 4096), np.float32)
    for group in range(8):
        for row in range(32):
            woa[group * 32 + row, (row * 37 + group * 11) % 4096] = (
                -1.0 if row % 2 else 1.0
            )
    weights["attn.wo_a.weight"] = _buffer(
        (woa.view(np.uint32) >> 16).astype(np.uint16), DType.bfloat16
    )
    wob = np.zeros((64, 256), np.uint8)
    wob[np.arange(64), (np.arange(64) * 13 + 7) % 256] = np.uint8(0x38)
    weights["attn.wo_b.weight"] = _buffer(wob, DType.float8_e4m3fn)
    weights["attn.wo_b.scale"] = _buffer(
        np.full((2, 8), 127, np.uint8), DType.float8_e8m0fnu
    )
    run = module.compile(
        TensorType(DType.bfloat16, ["rows", 64, 512], DeviceRef.GPU(0)),
        weights=weights,
    )
    return run, degree


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
    grouped = np.empty((rows, 8, 32), np.float32)
    for group in range(8):
        for row in range(32):
            grouped[:, group, row] = flat[
                :, group, (row * 37 + group * 11) % 4096
            ] * (-1.0 if row % 2 else 1.0)
    expected = grouped.reshape(rows, 256)[:, (np.arange(64) * 13 + 7) % 256]
    values_bf16 = _buffer(
        (values.view(np.uint32) >> 16).astype(np.uint16), DType.bfloat16
    )
    actual = run(Tensor.from_dlpack(values_bf16).to(Accelerator(0)))
    np.testing.assert_array_equal(actual.to_numpy(), expected)


class _PassFFN(Module[..., Tensor]):
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        del mask
        return x


class _Boundary(Module[[Tensor], tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        layer = object.__new__(DecoderLayer)
        Module.__init__(layer)
        layer.hc_mult = 4
        layer.norm_eps = 1e-20
        layer.plan = LayerPlan((), {}, "")
        layer.attn_norm = RMSNorm(64)
        layer.ffn_norm = RMSNorm(64)
        layer.ffn_norm.weight = F.full(
            [64], 2.0, dtype=DType.bfloat16, device=Accelerator(0)
        )
        monkeypatch.setattr(layer, "ffn", _PassFFN(), raising=False)
        monkeypatch.setattr(layer, "_attention", lambda x, site: (x, []))
        for role in ("attn", "ffn"):
            setattr(
                layer,
                f"hc_{role}_fn",
                Tensor.zeros([24, 256], dtype=DType.float32),
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
        TensorType(DType.bfloat16, ["rows", 4, 64], DeviceRef.GPU(0))
    )
    values = (
        np.random.default_rng(73).normal(size=(3, 4, 64)).astype(np.float32)
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
