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
"""Nemotron-H's Mamba state on the KV cache, and its weight loading."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import BufferValue, DeviceRef, TensorValue
from max.nn.kv_cache import RecurrentLeafInputs
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    NemotronHConfig,
)
from max.pipelines.architectures.nemotron_h_modulev3.nemotron_h import (
    NemotronH,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    NemotronHQuantScheme,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    plan_jenga_geometry,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from transformers import NemotronHConfig as HFNemotronHConfig

MIB = 1024**2

_TINY_LAYERS = ["mamba", "moe", "mamba", "attention", "mamba", "mlp"]
_TINY = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    mamba_num_heads=4,
    mamba_head_dim=8,
    n_groups=2,
    ssm_state_size=8,
    conv_kernel=4,
)


def _config(layers: list[str], dims: dict[str, int]) -> NemotronHConfig:
    hf = HFNemotronHConfig(
        layers_block_type=layers,
        vocab_size=64,
        intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=24,
        routed_scaling_factor=2.5,
        **dims,
    )
    pipeline = Mock()
    pipeline.model.data_parallel_degree = 1
    devices = [DeviceRef.CPU()]
    kv_params = NemotronHConfig.construct_kv_params(
        huggingface_config=hf,
        pipeline_config=pipeline,
        devices=devices,
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    return NemotronHConfig.from_huggingface(
        hf,
        kv_params=kv_params,
        devices=devices,
        max_seq_len=256,
    )


def _model(config: NemotronHConfig) -> NemotronH:
    with F.lazy(), default_dtype(DType.bfloat16):
        model = NemotronH(config)
        model.to(CPU())
    return model


def test_lightning_state_tiles_one_huge_page_exactly() -> None:
    """23 Mamba layers is prime, so the huge page is 414 MiB and unpadded.

    Jenga addresses a state's layers as consecutive rows of its page, which
    holds only while no page is padded.
    """
    # Nemotron-3.5-Lightning's layer schedule and mixer shapes.
    layers = [
        {"M": "mamba", "E": "moe", "*": "attention"}[c]
        for c in "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    ]
    config = _config(
        layers,
        dict(
            hidden_size=2688,
            num_attention_heads=32,
            num_key_value_heads=2,
            head_dim=128,
            mamba_num_heads=64,
            mamba_head_dim=64,
            n_groups=8,
            ssm_state_size=128,
            conv_kernel=4,
        ),
    )
    leaves = config.kv_params.leaves()
    page_bytes = {key: leaf.bytes_per_page for key, leaf in leaves.items()}
    assert page_bytes == {
        "attn.full_group": 768 * 1024,
        "mamba/conv": 23 * 6144 * 3 * 2,
        "mamba/ssm": 23 * 64 * 64 * 128 * 4,
    }

    geometry = plan_jenga_geometry(
        100 * 1024 * MIB,
        page_bytes,
        {key: leaf.row_bytes for key, leaf in leaves.items()},
    )

    assert geometry.huge_page_bytes == 414 * MIB
    assert geometry.ratios == {
        "attn.full_group": 552,
        "mamba/conv": 512,
        "mamba/ssm": 9,
    }
    assert geometry.padded_sizes == page_bytes


def test_each_mamba_layer_reads_its_own_state_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharing a row would load, run and return fluent nonsense.

    The Mamba layers share one subgraph, so a row selected inside it would
    be recorded once instead of once per layer.
    """
    config = _config(_TINY_LAYERS, _TINY)
    model = _model(config)
    selected: list[int] = []
    original = RecurrentLeafInputs.live_row_id

    def record(
        self: RecurrentLeafInputs[TensorValue, BufferValue], layer: int
    ) -> TensorValue:
        selected.append(layer)
        return original(self, layer)

    monkeypatch.setattr(RecurrentLeafInputs, "live_row_id", record)
    model.trace(
        *modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params, device_refs=config.devices
        )
    )

    # The conv then the SSM row of each of the three Mamba layers.
    assert selected == [0, 0, 1, 1, 2, 2]


def test_loading_is_strict() -> None:
    """A missing weight or a stray scale fails the load by name."""
    config = _config(_TINY_LAYERS, _TINY)
    model = _model(config)
    weights = dict(model.parameters)
    names = set(weights)

    missing = "backbone.layers.1.mixer.experts.3.down_proj.weight"
    del weights[missing]
    with pytest.raises(KeyError, match=missing):
        model.compile(weights=weights)
    # A scale beside a BF16 weight.
    stray = "backbone.layers.0.mixer.in_proj"
    with pytest.raises(ValueError, match=stray):
        config.quant_scheme.check_weights(names | {f"{stray}.weight_scale"})


def test_w4a4_selects_the_moe_mixers_with_nvfp4_experts() -> None:
    config = _config(_TINY_LAYERS, _TINY)
    nvfp4 = {
        f"backbone.layers.1.mixer.experts.{e}.{proj}": (
            ModuleFormat.NVFP4_WEIGHT_ONLY
        )
        for e in range(config.num_experts)
        for proj in ("up_proj", "down_proj")
    }
    config = replace(
        config, w4a4_experts=True, quant_scheme=NemotronHQuantScheme(nvfp4)
    )
    assert config.w4a4_mixers() == {"backbone.layers.1.mixer"}
