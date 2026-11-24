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
"""The rules map every checkpoint tensor onto the module tree's names.

The module declares each tensor as the checkpoint stores it, so the rules
only drop what no graph reads. The checks read the tensor manifest of each
published Nemotron-3.5-Lightning checkpoint, BF16 and NVFP4, in
``testdata/lightning_checkpoints.json``: the names, dtypes and shapes of
their safetensors headers and their ``config.json`` at a pinned revision. The
manifest keeps every tensor outside the decoder layers, the MTP head
included, and decoder layers 0-5, which hold one of each layer kind, with
the config cut to the same six layers. ``experts.*`` stands for each routed
expert, which all store the same dtype and shape. The manifest comes from the
checkpoints, not from the module, so a misspelled parameter or a wrong stored
shape or dtype fails here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest
from _tiny_config import TINY, TINY_LAYERS, config_from_hf, hf_config
from max import functional as F
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.nemotron_h_modulev3.nemotron_h import (
    NemotronH,
)
from max.pipelines.architectures.nemotron_h_modulev3.weight_adapters import (
    nemotron_h_weight_rules,
)
from max.pipelines.lib.weight_transform import Ignore, WeightTransform
from max.tensor import default_dtype
from transformers import NemotronHConfig as HFNemotronHConfig

_MANIFEST = Path(__file__).parent / "testdata" / "lightning_checkpoints.json"

_SAFETENSORS_DTYPES = {
    "BF16": DType.bfloat16,
    "F32": DType.float32,
    "F8_E4M3": DType.float8_e4m3fn,
    "U8": DType.uint8,
}

_T = TypeVar("_T")


def _zeros(shape: tuple[int, ...], dtype: DType) -> WeightData:
    """Returns a zero weight of ``shape`` and ``dtype``, as a checkpoint
    stores it: numpy has no BF16 or FP8 dtype, so the zeros are raw bytes."""
    array = np.zeros(shape, f"u{dtype.size_in_bytes}")
    buffer = Buffer.from_numpy(np.ascontiguousarray(array)).view(dtype, shape)
    return WeightData(
        data=buffer, name="", dtype=dtype, shape=Shape(list(shape))
    )


def _header_only(dtype: str, shape: Sequence[int]) -> WeightData:
    """Returns a weight with a header's dtype and shape and no data: the
    rules read names only."""
    return WeightData(
        data=np.empty(0, np.uint8),
        name="",
        dtype=_SAFETENSORS_DTYPES[dtype],
        shape=Shape(list(shape)),
    )


def _each_expert(entries: Mapping[str, _T], num_experts: int) -> dict[str, _T]:
    """Returns ``entries`` with each ``experts.*`` name repeated per expert."""
    expanded: dict[str, _T] = {}
    for name, value in entries.items():
        if ".experts.*." not in name:
            expanded[name] = value
            continue
        for e in range(num_experts):
            expanded[name.replace(".experts.*.", f".experts.{e}.")] = value
    return expanded


def _published(
    variant: str,
) -> tuple[HFNemotronHConfig, dict[str, WeightData]]:
    """Returns a published checkpoint's config and its header-only tensors."""
    entry = json.loads(_MANIFEST.read_text())[variant]
    raw = dict(entry["config"])
    num_experts = raw["n_routed_experts"]
    if quant := raw.get("quantization_config"):
        raw["quantization_config"] = quant | {
            "quantized_layers": _each_expert(
                quant["quantized_layers"], num_experts
            )
        }
    checkpoint = {
        name: _header_only(dtype, shape)
        for name, (dtype, shape) in _each_expert(
            entry["tensors"], num_experts
        ).items()
    }
    return HFNemotronHConfig.from_dict(raw), checkpoint


def _dtype_and_shape(
    tensor: WeightData,
) -> tuple[DType, tuple[int, ...]]:
    return tensor.dtype, tuple(int(dim) for dim in tensor.shape)


@pytest.mark.parametrize("variant", ["bf16", "nvfp4"])
def test_the_published_checkpoint_maps_onto_the_module_tree(
    variant: str,
) -> None:
    """Every published tensor the rules keep maps to a name the module
    declares, with the declared dtype and shape, and nothing the module
    declares is left unbound."""
    hf, checkpoint = _published(variant)
    adapted = WeightTransform(nemotron_h_weight_rules(hf)).apply_data(
        checkpoint
    )
    config = config_from_hf(hf)
    config.finalize(hf, adapted)
    with F.lazy(), default_dtype(DType.bfloat16):
        module = NemotronH(config)
    declared = {
        name: (param.dtype, tuple(int(dim) for dim in param.shape))
        for name, param in module.parameters
    }
    stored = {name: _dtype_and_shape(w) for name, w in adapted.items()}

    assert sorted(declared.keys() - stored.keys()) == []
    assert sorted(stored.keys() - declared.keys()) == []
    assert {
        name: (stored[name], declared[name])
        for name in stored
        if stored[name] != declared[name]
    } == {}


def test_the_rules_only_drop_weights() -> None:
    """No rule renames or relabels: the module tree mirrors the checkpoint,
    so the rules are Ignore rules alone."""
    rules = nemotron_h_weight_rules(hf_config(TINY_LAYERS, TINY))
    assert rules and all(isinstance(rule, Ignore) for rule in rules)
    assert [rule.pattern for rule in rules] == [
        r"^mtp\.",
        r"\.k_proj\.k_scale$",
        r"\.v_proj\.v_scale$",
    ]


@pytest.mark.parametrize(
    "name",
    ["mtp.norm.weight", "backbone.layers.3.mixer.k_proj.k_scale"],
)
def test_the_rules_drop_only_the_head_and_the_scales(name: str) -> None:
    """The MTP head and the KV-cache scales drop; a tensor the graph loads
    keeps its own name."""
    adapted = WeightTransform(
        nemotron_h_weight_rules(hf_config(TINY_LAYERS, TINY))
    ).apply_data(
        {
            name: _zeros((1,), DType.float32),
            "backbone.layers.1.mixer.gate.weight": _zeros(
                (1, 1), DType.bfloat16
            ),
        }
    )
    assert set(adapted) == {"backbone.layers.1.mixer.gate.weight"}
