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
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.sharding import DeviceMapping, DeviceMesh, Sharded
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layer_resolver import (
    resolve,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    Attention,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
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
