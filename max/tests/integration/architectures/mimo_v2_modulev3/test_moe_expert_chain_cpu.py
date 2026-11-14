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
"""The routed experts' chain, staged on a virtual B200 and never compiled.

The gate/up matmul's epilogue applies the gated activation and the down
projection's re-quantize, so the experts are two launches; the
``MAX_DISABLE_FUSED_SWIGLU_NVFP4`` escape hatch restores the chained four on
the same sigma-permuted weights. Both arms are wired here; their numerics are
the GPU gates' business.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tiny_fixture
from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn import _graph_ops as kernels
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2Decoder
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.lib.module_registration import (
    _DecoderGraph,
    decoder_input_types,
)
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from mimo_v3_runner import kv_params
from transformers.configuration_utils import PretrainedConfig

FUSED_GATE_UP = "grouped_matmul_swiglu_nvfp4"
CHAINED_ACTIVATION = "ep.fused_silu"


@pytest.mark.parametrize("fused", [True, False])
def test_fused_swiglu_switch_picks_the_expert_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fused: bool
) -> None:
    """Two launches by default, four under the switch."""
    if fused:
        # The default arm, whatever the caller's environment says.
        monkeypatch.delenv("MAX_DISABLE_FUSED_SWIGLU_NVFP4", raising=False)
    else:
        monkeypatch.setenv("MAX_DISABLE_FUSED_SWIGLU_NVFP4", "1")
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)

    tiny_fixture.write(tmp_path, tiny_fixture.tensors())
    hf = PretrainedConfig(**json.loads((tmp_path / "config.json").read_text()))
    refs = [DeviceRef.GPU(0)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    weights = SafetensorWeights(sorted(tmp_path.glob("*.safetensors")))
    adapted = convert_safetensor_state_dict(dict(weights.items()))
    mesh = DeviceMesh((Accelerator(0),), (1,), ("tp",))
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2Decoder(config)
        inputs = decoder_input_types(
            config.kv_params,
            refs[0],
            input_layout=MiMoV2Decoder.input_layout,
        )
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = _DecoderGraph(model).stage(
            inputs,
            weights=adapted,
            name=MiMoV2Decoder.graph_name,
        )
        text = str(staged)
    fused_present = FUSED_GATE_UP in text
    chained_present = CHAINED_ACTIVATION in text
    assert fused_present == fused
    assert chained_present == (not fused)
    # The fused epilogue rounds its SwiGLU through bfloat16, which the chained
    # pair of launches reproduces by writing bfloat16; the fused arm's graph
    # carries that as the activation it fuses, so a switch that stopped short
    # of the epilogue, or one that changed its rounding, shows up here.
    assert ("silu_bf16" in text) == fused, text
