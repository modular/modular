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
"""The routed experts' tensor-parallel stacks on a CPU mesh, byte for byte,
and the staged TP4 graph's device-resident expert weights.

The grouped matmuls read each expert's gate rows followed by its up rows,
and the scale interleave reads the scales in the same order, so every
device's stacks are checked against that layout built in numpy from the
checkpoint's per-expert tensors. The expert-parallel counterparts live in
``test_ep_construction_cpu.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tiny_fixture
from expert_stacks import (
    PROJECTIONS,
    Array,
    check_staged_expert_stacks,
    gate_up,
    load_random_experts,
    realize,
)
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.legacy import kernels
from max.nn.legacy.quant_config import QuantConfig
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import MiMoV2MoE
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
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from mimo_v3_runner import kv_params
from tiny_fixture import EXPERTS, HIDDEN, MOE_DIM, TOP_K
from transformers.configuration_utils import PretrainedConfig


def _tp_moe() -> MiMoV2MoE:
    # The E8M0 zeros the experts start with would compile eagerly.
    with F.lazy(), default_dtype(DType.bfloat16):
        return MiMoV2MoE(
            hidden_dim=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            moe_dim=MOE_DIM,
            norm_topk_prob=True,
            quant_config=QuantConfig.mxfp4(mlp_quantized_layers={1, 2, 3}),
        )


@pytest.mark.parametrize("n_devices", [1, 2, 4])
def test_tp_stacks_split_every_experts_channels(
    monkeypatch: pytest.MonkeyPatch, n_devices: int
) -> None:
    """Device ``d`` holds rows ``d`` of ``n`` of every expert's gate and up
    projections and the same columns of its down projection."""
    layer = _tp_moe()
    ref = load_random_experts(layer, monkeypatch)
    mesh = DeviceMesh((CPU(),) * n_devices, (n_devices,), ("tp",))
    got = realize(layer, mesh)
    assert len(got) == n_devices
    for d, (gate_up_w, gate_up_scales, down, down_scales) in enumerate(got):
        rows = np.s_[
            :, d * MOE_DIM // n_devices : (d + 1) * MOE_DIM // n_devices
        ]

        def cols(a: Array, d: int = d) -> Array:
            width = a.shape[2] // n_devices
            return a[:, :, d * width : (d + 1) * width]

        (gate, gate_s), (up, up_s), (down_ref, down_s) = (
            ref[name] for name in PROJECTIONS
        )
        np.testing.assert_array_equal(gate_up_w, gate_up(gate[rows], up[rows]))
        np.testing.assert_array_equal(
            gate_up_scales, gate_up(gate_s[rows], up_s[rows])
        )
        np.testing.assert_array_equal(down, cols(down_ref))
        np.testing.assert_array_equal(down_scales, cols(down_s))


def test_tp4_staged_graph_holds_only_final_expert_stacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each device holds only its final expert weight stacks.

    Each device receives its share of the host-built gate/up stack in one
    transfer. A separate gate or up stack would be a device temporary freed
    around the persistent stacks, which strands allocator granules and
    shrinks the KV cache.
    """
    n_devices = 4
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)

    tiny_fixture.write(tmp_path, tiny_fixture.tensors())
    hf = PretrainedConfig(**json.loads((tmp_path / "config.json").read_text()))
    refs = [DeviceRef.GPU(i) for i in range(n_devices)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    inputs = modulev3_ragged_kv_symbolic_inputs(
        kv_params=config.kv_params, device_refs=refs
    )
    weights = SafetensorWeights(sorted(tmp_path.glob("*.safetensors")))
    adapted = convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=hf
    )
    devices = [Accelerator(i) for i in range(n_devices)]
    mesh = DeviceMesh(tuple(devices), (n_devices,), ("tp",))
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config, None)
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = model.stage(*inputs, weights=adapted)
        text = str(staged)
    assert "mo.call @mimo_v2_moe" in text
    check_staged_expert_stacks(text, n_devices, expert_parallel=False)
