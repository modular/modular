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
"""Weight loading for the Kimi K3 DSpark draft.

The draft module tree mirrors its checkpoint's names, so tensors load as
stored. The only tensor nothing reads is the confidence head's: it gates
adaptive proposal lengths, which MAX's block driver does not consume, so a
rule ignores it rather than loading a module nothing calls.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.lib import Ignore, WeightRule

__all__ = ["k3_dspark_weight_rules"]


def k3_dspark_weight_rules(huggingface_config: object) -> Sequence[WeightRule]:
    """Returns the rules that load the draft checkpoint as stored."""
    del huggingface_config
    return (
        Ignore(
            r"^confidence_head\.",
            reason="the confidence head, which the block driver doesn't run",
        ),
    )
