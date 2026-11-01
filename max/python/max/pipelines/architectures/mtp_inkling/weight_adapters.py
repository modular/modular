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
"""Weight adapter for the MTP depths an Inkling checkpoint ships."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from max.graph.weights import WeightData, Weights
from max.pipelines.lib.config import PipelineConfig
from transformers import AutoConfig

from ..inkling.model_config import parse_inkling_mtp_config
from ..inkling.weight_adapters import _TEXT, _convert, _rename

logger = logging.getLogger("max.pipelines")

_MTP = "model.mtp."


def _resolved_mtp_depths(
    huggingface_config: AutoConfig | None,
    pipeline_config: PipelineConfig | None,
) -> int | None:
    """Depths the draft will build, or ``None`` when that can't be resolved."""
    if huggingface_config is None or pipeline_config is None:
        return None
    mtp = parse_inkling_mtp_config(huggingface_config)
    spec = pipeline_config.speculative
    if mtp is None or spec is None or not spec.num_speculative_tokens:
        # A width of 0 is a config error; don't turn it into a weight-name
        # failure here.
        return None
    return mtp.num_depths_for(spec)


def _beyond_depth_cap(name: str, n_depths: int | None) -> bool:
    """Whether ``name`` is an MTP weight for a depth the draft won't build."""
    if n_depths is None:
        return False
    parts = name[len(_MTP) :].split(".", 2)
    if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
        # Not a per-depth weight (``chain_norm``), so never capped.
        return False
    return int(parts[1]) >= n_depths


def convert_inkling_mtp_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: AutoConfig | None,
    pipeline_config: PipelineConfig | None = None,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Keeps an Inkling checkpoint's MTP depths, under the draft's names.

    The depths embed and project through the target's table and head, so
    only their own weights load here. Weights for depths past the requested
    speculative width are dropped without being read.
    """
    del unused_kwargs
    n_depths = _resolved_mtp_depths(huggingface_config, pipeline_config)
    converted: dict[str, WeightData] = {}
    skipped = 0
    for name, value in state_dict.items():
        if not name.startswith(_MTP) or name.endswith(".original_shape"):
            continue
        if _beyond_depth_cap(name, n_depths):
            skipped += 1
            continue
        converted |= _convert_mtp(name, value.data())
    if skipped:
        logger.info(
            f"Inkling MTP: skipped {skipped} checkpoint weights of depths past"
            f" the {n_depths} the draft builds"
        )
    return converted


def _convert_mtp(name: str, data: WeightData) -> dict[str, WeightData]:
    rest = name[len(_MTP) :]
    if rest.startswith("chain_norm."):
        return {rest: _rename(data, rest)}
    parts = rest.split(".", 2)
    if len(parts) != 3 or parts[0] != "layers":
        raise ValueError(f"unrecognized Inkling MTP weight {name!r}")
    _, depth_idx, suffix = parts
    prefix = f"layers.{depth_idx}."
    if suffix.startswith("transformer_block."):
        block = suffix[len("transformer_block.") :]
        fake = f"{_TEXT}layers.{depth_idx}.{block}"
        out = {}
        for key, weight in _convert(fake, data).items():
            inner = key[len(f"layers.{depth_idx}.") :]
            dest = f"{prefix}decoder_layer.{inner}"
            out[dest] = _rename(weight, dest)
        return out
    dest = f"{prefix}{suffix}"
    return {dest: _rename(data, dest)}
