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
"""Model inputs for the MPNet ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class MPNetInputs(ModelInputs):
    """Input tensors for the MPNet model."""

    next_tokens_batch: Buffer
    attention_mask: Buffer
