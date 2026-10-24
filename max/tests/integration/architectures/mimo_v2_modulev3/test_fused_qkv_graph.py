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
"""The fused chunked QKV op reaches the base graph only where it is asked for.

Staged on virtual B200s over the tiny fixture at TP2 and TP4 and never
compiled. Attention runs inline, once per device, so every layer of a fused
attention type is one op site per device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tiny_fixture
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from mimo_v3_runner import kv_params
from transformers.configuration_utils import PretrainedConfig

FUSED_QKV = "mo.chunked_qkv_rope_store.ragged.paged"
SLIDING_LAYERS = sum(sliding for sliding, _ in tiny_fixture.LAYERS)
FULL_LAYERS = len(tiny_fixture.LAYERS) - SLIDING_LAYERS


@pytest.mark.parametrize("n_devices", [2, 4])
@pytest.mark.parametrize(
    ("sliding", "full"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["off", "sliding", "full", "both"],
)
def test_fused_qkv_sites(
    tmp_path: Path,
    n_devices: int,
    sliding: bool,
    full: bool,
) -> None:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)

    tiny_fixture.write(tmp_path, tiny_fixture.tensors())
    hf = PretrainedConfig(**json.loads((tmp_path / "config.json").read_text()))
    devices = [Accelerator(i) for i in range(n_devices)]
    refs = [DeviceRef.GPU(i) for i in range(n_devices)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    config.return_logits = ReturnLogits.ALL
    config.fused_sliding_qkv = sliding
    config.fused_full_qkv = full
    weights = SafetensorWeights(sorted(tmp_path.glob("*.safetensors")))
    adapted = convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=hf
    )

    mesh = DeviceMesh(tuple(devices), (n_devices,), ("tp",))
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config)
        inputs = modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params, device_refs=refs
        )
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = model.stage(*inputs, weights=adapted)
        text = str(staged)
    assert text.count(FUSED_QKV) == n_devices * (
        SLIDING_LAYERS * sliding + FULL_LAYERS * full
    )
