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
"""Checkpoint precision policy shared by GLM-5.3 config and weight adaptation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from max.driver import Device, accelerator_api
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.legacy.kernels import mhc_pre_supports
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
    parse_quant_scheme,
)

_HC_FN_SUFFIX = ".hc_fn"
_V3_SCALE = re.compile(
    r"^language_model\.(layers\.\d+\.(?:mlp|self_attn)\..+?)\.weight_scale_inv$"
)


def _stored_hc_fn_dtype(state_dict: Mapping[str, WeightData]) -> DType:
    """Returns the checkpoint dtype shared by every mHC stream projection.

    One dtype is chosen for all sites, so a mix would narrow the float32
    projections to bfloat16; it is rejected instead.
    """
    stored = {
        weight.dtype
        for name, weight in state_dict.items()
        if name.endswith(_HC_FN_SUFFIX)
    }
    if len(stored) > 1:
        raise ValueError(
            "GLM-5.3-Flash expects every mHC projection (*.hc_fn) in one"
            f" dtype, but the checkpoint mixes {sorted(map(str, stored))}"
        )
    return stored.pop() if stored else DType.float32


def _hc_fn_dtype(
    stored: DType,
    hc_mult: int,
    hidden_size: int,
    dtype: DType,
    device: Device,
    *,
    api: str | None = None,
) -> DType:
    """Returns the dtype the mHC sites hold their stream projections in.

    The fused pre-site op reads a bfloat16 projection in one pass rather
    than three, so it keeps the checkpoint's bfloat16. The op chain computes
    in float32, and widening there is exact.
    """
    if (
        stored == DType.bfloat16
        and (api if api is not None else accelerator_api()) == "hip"
        and mhc_pre_supports(hc_mult, hidden_size, dtype, stored, device)
    ):
        return DType.bfloat16
    return DType.float32


def checkpoint_quant_scheme(
    hf_quantization_config: dict[str, Any] | None,
    declared_dtype: DType | None,
    state_dict: Mapping[str, Any],
) -> Glm5NextQuantScheme | None:
    """Resolves mixed precision from the adapted projection scales."""
    renamed = {
        f"{match.group(1)}.weight_scale": value
        for name, value in state_dict.items()
        if (match := _V3_SCALE.match(name)) is not None
    }
    return parse_quant_scheme(
        hf_quantization_config, renamed, declared_dtype or DType.bfloat16
    )


def cast_hc_fn_weights(
    state_dict: dict[str, Any], dtype: DType
) -> dict[str, Any]:
    """Casts mHC projections while preserving other checkpoint entries."""
    return {
        name: weight.astype(dtype)
        if name.endswith(_HC_FN_SUFFIX)
        and isinstance(weight, WeightData)
        and weight.dtype != dtype
        else weight
        for name, weight in state_dict.items()
    }
