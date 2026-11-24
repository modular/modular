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
"""The F1 fixture's config, derived from the full V4.1 config.

F1 keeps 12 real layers of the checkpoint, renumbered 0..11, with 16 experts
per layer and row-cut Engram tables. This stands in for the config the F1
builder emits until that builder lands, and keeps the full config's
vision_config: the fixture checkpoint is cut from the multimodal full model,
so its router carries the vision-only `gate.bias_vl` bias the tree declares.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

# Full-model layer behind each F1 layer: 0 and 1 are sliding-window only, 2, 3,
# 4, 14 and 19 run ratio 2, and 20, 21, 24, 25 and 36 run ratio 1.
F1_LAYERS = (0, 1, 2, 3, 4, 14, 19, 20, 21, 24, 25, 36)
F1_EXPERTS = 16
# Engram tables cut to engram_vocab_size=20000 (a prime layout per table).
F1_ENGRAM_ROWS = (482648, 488250)


def _renumber(ids: Sequence[int]) -> list[int]:
    return [F1_LAYERS.index(layer) for layer in ids if layer in F1_LAYERS]


def f1_config(full: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the F1 HF config for the full-model HF config ``full``."""
    config = copy.deepcopy(dict(full))
    text = config["text_config"]
    text["num_hidden_layers"] = len(F1_LAYERS)
    text["compress_ratios"] = [text["compress_ratios"][l] for l in F1_LAYERS]
    for key in (
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "engram_layer_ids",
    ):
        text[key] = _renumber(text[key])
    text["candidate_source_layer_id"] = F1_LAYERS.index(
        text["candidate_source_layer_id"]
    )
    text["n_routed_experts"] = F1_EXPERTS
    text["engram_num_embeddings"] = list(F1_ENGRAM_ROWS)
    return config
