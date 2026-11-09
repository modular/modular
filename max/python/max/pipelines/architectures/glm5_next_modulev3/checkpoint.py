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

from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.glm5_next.quantization import (
    Glm5NextQuantScheme,
    parse_quant_scheme,
)

_HC_FN_SUFFIX = ".hc_fn"
_V3_SCALE = re.compile(
    r"^language_model\.(layers\.\d+\.(?:mlp|self_attn)\..+?)\.weight_scale_inv$"
)
_MLA_PROJECTION = re.compile(
    r"^(language_model\.layers\.\d+\.self_attn\."
    r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj))(\.weight)?$"
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


def check_raw_projections(state_dict: Mapping[str, Any]) -> None:
    """Checks each MLA projection's name against its scale.

    The weight rules name a projection bare when the checkpoint's
    ``modules_to_not_convert`` lists it, while the precision map reads which
    projections carry a scale. The two have to agree.

    Raises:
        ValueError: If ``modules_to_not_convert`` lists a projection the
            checkpoint quantizes, or omits one it leaves unquantized.
    """
    for name in state_dict:
        if (match := _MLA_PROJECTION.match(name)) is None:
            continue
        projection, weight = match.groups()
        quantized = f"{projection}.weight_scale_inv" in state_dict
        if quantized != (weight is not None):
            raise ValueError(
                f"GLM-5.3-Flash ships {projection}"
                f" {'with' if quantized else 'without'} an FP8 scale, but its"
                " quantization_config.modules_to_not_convert"
                f" {'lists' if quantized else 'omits'} it"
            )
