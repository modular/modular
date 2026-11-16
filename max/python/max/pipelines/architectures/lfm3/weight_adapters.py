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
"""Declarative loading for LFM3's checkpoint-native parameter names."""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import WeightRule
from transformers import PretrainedConfig


def lfm3_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Keeps the text checkpoint's names and tensor storage unchanged."""
    del huggingface_config
    return ()
