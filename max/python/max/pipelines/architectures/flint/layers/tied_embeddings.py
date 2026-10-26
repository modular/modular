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

"""Decides whether the lm head shares the token embedding weights.

Requires both the config tie and an absent ``lm_head`` tensor, so a missing
head fails the strict load instead of serving embedding-similarity logits.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def resolve_tie_word_embeddings(
    hf_cfg: Any, state_dict_keys: Iterable[str]
) -> bool:
    config_ties = bool(getattr(hf_cfg, "tie_word_embeddings", False))
    has_head = any(k.endswith("lm_head.weight") for k in state_dict_keys)
    return config_ties and not has_head
