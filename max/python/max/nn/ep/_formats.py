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
"""The dispatch formats of the exchange, and the results each one returns."""

from __future__ import annotations

from dataclasses import dataclass

from max.nn._graph_ops import ep as kernels
from max.nn.ep_config import EPConfig
from max.nn.quant_config import QuantFormat


@dataclass(frozen=True)
class DispatchLayout:
    """The results of one dispatch format's kernel, by ``Grouped`` field."""

    fields: tuple[str, ...]
    """The ``Grouped`` field of each result before the route, in order."""


_UNSCALED = DispatchLayout(("tokens", "expert_start", "expert_ids"))
_SCALED = DispatchLayout(("tokens", "scales", "expert_start", "expert_ids"))
_PADDED = DispatchLayout(
    ("tokens", "scales", "expert_start", "scales_offset", "expert_ids")
)
_PADDED_WITH_ROW_SCALES = DispatchLayout(
    (
        "tokens",
        "scales",
        "row_scales",
        "expert_start",
        "scales_offset",
        "expert_ids",
    )
)

# Keyed by the dispatch quantization format, whether the NVIDIA block-scaled
# layout serves it, and whether NVFP4 scales each token on its own. The
# block-scaled layout pads each expert's scales to whole tiles, which
# `scales_offset` locates. Every FP8 format but MXFP8 takes the same float8
# kernel, and MXFP6 takes the MX token format kernel of MXFP4. A format
# missing here is not supported.
_LAYOUTS: dict[tuple[QuantFormat | None, bool, bool], DispatchLayout] = {
    (None, False, False): _UNSCALED,
    (QuantFormat.BLOCKSCALED_FP8, False, False): _SCALED,
    (QuantFormat.COMPRESSED_TENSORS_FP8, False, False): _SCALED,
    (QuantFormat.FBGEMM_FP8, False, False): _SCALED,
    (QuantFormat.MXFP4, False, False): _SCALED,
    (QuantFormat.MXFP6, False, False): _SCALED,
    (QuantFormat.MXFP8, False, False): _SCALED,
    (QuantFormat.NVFP4, True, False): _PADDED,
    (QuantFormat.NVFP4, True, True): _PADDED_WITH_ROW_SCALES,
    (QuantFormat.MXFP8, True, False): _PADDED,
}


def dispatch_format(config: EPConfig) -> QuantFormat | None:
    """Returns the format of ``config``'s dispatch, or ``None`` for BF16.

    Args:
        config: The expert-parallel configuration.
    """
    return getattr(config.dispatch_quant_config, "format", None)


def _key(config: EPConfig) -> tuple[QuantFormat | None, bool, bool]:
    return (
        dispatch_format(config),
        kernels.uses_block_scaled_nv_ep_layout(config),
        config.nvfp4_dyn_global_scales,
    )


def is_supported(config: EPConfig) -> bool:
    """Returns whether the exchange can dispatch in ``config``'s format."""
    return _key(config) in _LAYOUTS


def dispatch_layout(config: EPConfig) -> DispatchLayout:
    """Returns the results of ``config``'s dispatch format.

    Args:
        config: The expert-parallel configuration, of a supported format.

    Returns:
        The layout of the dispatch kernel's results.
    """
    return _LAYOUTS[_key(config)]
