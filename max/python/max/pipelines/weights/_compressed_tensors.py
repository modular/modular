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

"""Shared compressed-tensors checkpoint normalization helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData


def is_compressed_tensors_nvfp4_config(
    config: Mapping[str, Any] | None,
) -> bool:
    """Whether a compressed-tensors config describes NVFP4 weights."""
    if (
        not isinstance(config, Mapping)
        or config.get("quant_method") != "compressed-tensors"
    ):
        return False

    config_groups = config.get("config_groups")
    if not isinstance(config_groups, Mapping):
        return False
    group = config_groups.get("group_0")
    if not isinstance(group, Mapping):
        return False

    weights = group.get("weights")
    inputs = group.get("input_activations")
    if not isinstance(weights, Mapping) or not isinstance(inputs, Mapping):
        return False

    format_name = config.get("format") or group.get("format")
    return bool(
        format_name == "nvfp4-pack-quantized"
        and weights.get("type") == "float"
        and weights.get("num_bits") == 4
        and weights.get("strategy") == "tensor_group"
        and weights.get("group_size") == 16
        and weights.get("dynamic") is False
        and inputs.get("type") == "float"
        and inputs.get("num_bits") == 4
        and inputs.get("strategy") == "tensor_group"
        and inputs.get("group_size") == 16
    )


def convert_compressed_tensors_nvfp4_weight(
    name: str, data: WeightData
) -> tuple[str, WeightData]:
    """Normalize one compressed-tensors NVFP4 weight for MAX.

    Global scales use the reciprocal of MAX's multiplicative convention, so
    they are inverted once while their raw compressed-tensors suffix is still
    present. Per-group E4M3 scales have identical values in both conventions;
    only exporters that store their bytes as ``uint8`` need a dtype
    reinterpretation.
    """
    if name.endswith(".weight_packed"):
        name = name.removesuffix(".weight_packed") + ".weight"
    elif name.endswith(".weight_global_scale"):
        name = name.removesuffix(".weight_global_scale") + ".weight_scale_2"
        arr = np.from_dlpack(data)
        reciprocal = np.asarray(
            np.reciprocal(arr.astype(np.float32)), dtype=np.float32
        ).reshape(arr.shape)
        data = WeightData.from_numpy(reciprocal, name)
    elif name.endswith(".input_global_scale"):
        name = name.removesuffix(".input_global_scale") + ".input_scale"
        arr = np.from_dlpack(data)
        reciprocal = np.asarray(
            np.reciprocal(arr.astype(np.float32)), dtype=np.float32
        ).reshape(arr.shape)
        data = WeightData.from_numpy(reciprocal, name)
    elif name.endswith(".weight_scale") and data.dtype == DType.uint8:
        data = dataclasses.replace(data, dtype=DType.float8_e4m3fn)

    return name, data
