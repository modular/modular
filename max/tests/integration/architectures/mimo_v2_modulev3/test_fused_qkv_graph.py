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
attention type is one op site per device, each writing the split cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tiny_fixture
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef, Shape
from max.graph.weights import SafetensorWeights
from max.nn.kernels.quant import (
    _rms_norm_quantize_dual_rule,
    _rms_norm_quantize_rule,
    rms_norm_quantize_dynamic_scaled_float8,
    rms_norm_quantize_dynamic_scaled_float8_dual,
)
from max.nn.outputs import ReturnLogits
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
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
from max.tensor.sharding import DeviceMesh, Sharded, TensorLayout
from max.tensor.sharding.rules import rms_norm_rule
from mimo_v3_runner import kv_params
from transformers.configuration_utils import PretrainedConfig

FUSED_QKV = "kv_cache.chunked_qkv_rope_store.ragged.paged"
SLIDING_LAYERS = sum(sliding for sliding, _ in tiny_fixture.LAYERS)
FULL_LAYERS = len(tiny_fixture.LAYERS) - SLIDING_LAYERS
RMS_NORM_QUANTIZE = "quantization.rms_norm_quantize_dynamic_scaled_float8"
STANDALONE_RMS_NORM = "mo.composite.reduce.rms_norm"


def _staged(
    tmp_path: Path,
    n_devices: int,
    *,
    sliding: bool = False,
    full: bool = False,
) -> str:
    """Returns the base graph of the tiny fixture, staged and never compiled."""
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
    adapted = convert_safetensor_state_dict(dict(weights.items()))

    mesh = DeviceMesh(tuple(devices), (n_devices,), ("tp",))
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
        return text


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
    text = _staged(tmp_path, n_devices, sliding=sliding, full=full)
    sites = n_devices * (SLIDING_LAYERS * sliding + FULL_LAYERS * full)
    assert text.count(FUSED_QKV) == sites
    # Every site writes the split cache's packed rows, not padded planes.
    assert text.count("split_cache = true") == sites


@pytest.mark.parametrize("n_devices", [2, 4])
def test_qkv_input_norm_fuses_into_the_quantize(
    tmp_path: Path, n_devices: int
) -> None:
    """Each attention layer normalizes its input inside the FP8 quantize.

    Attention runs inline, so the fused op is one site per layer per device.
    Every other norm keeps a standalone site: the post-attention norm of each
    layer plus the model's final one.
    """
    text = _staged(tmp_path, n_devices)
    layers = len(tiny_fixture.LAYERS)
    assert text.count(RMS_NORM_QUANTIZE) == n_devices * layers
    assert text.count(STANDALONE_RMS_NORM) == n_devices * (layers + 1)


def test_fused_norm_rule_offers_the_rms_norm_rows() -> None:
    """The fused norm + quantize can't take a contraction-sharded input.

    The kernel reduces the sum of squares over the whole row, so a device
    holding a K shard would normalize by a partial one. The rule offers
    exactly the rows ``rms_norm_rule`` does, which the cost model reshards
    from rather than quantizing a partial norm.

    Both forms of the op take the same placements, so only the number of
    results tells their rules apart: the fused op returns the FP8 data and its
    block scales, the dual form the normalized row as well. Each op is bound to
    the rule that places its own results, which a swapped binding would break.
    """
    mesh = DeviceMesh((CPU(), CPU()), (2,), ("tp",))
    x = TensorLayout(DType.bfloat16, Shape([4, 256]), mesh)
    gamma = TensorLayout(DType.bfloat16, Shape([256]), mesh)
    fused = _rms_norm_quantize_rule(x, gamma)
    assert [a.needed_inputs for a in fused] == [
        a.needed_inputs for a in rms_norm_rule(x, gamma, 1e-6)
    ]
    assert all(
        placement != Sharded(1) for a in fused for placement in a.needed_inputs
    )
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 128),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float32,
        block_size=(128, 128),
    )
    dual_op = rms_norm_quantize_dynamic_scaled_float8_dual
    dual_rule = dual_op.rule  # type: ignore[attr-defined]
    assert dual_rule is _rms_norm_quantize_dual_rule
    assert {len(row.outputs) for row in fused} == {2}
    assert {
        len(row.outputs)
        for row in dual_rule(x, gamma, 1e-6, input_spec, weight_spec)
    } == {3}
    # Each op keeps the rule that places its own results, by name, so a swapped
    # binding fails here rather than as a mis-sharded result count.
    fused_op = rms_norm_quantize_dynamic_scaled_float8
    assert fused_op.rule is _rms_norm_quantize_rule  # type: ignore[attr-defined]
