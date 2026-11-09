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
"""Constructs HYV4's registered MTP layer."""

from max.dtype import DType
from max.pipelines.lib.registration_draft import DraftRegistration
from max.tensor import default_dtype

from ..hy_v4.model_config import HYV4Config
from ..hy_v4.mtp import HYV4MultiTokenPredictor
from .model_config import configure_mtp_draft


class HYV4MTPDraft(HYV4MultiTokenPredictor):
    """Preserves the MTP layer's BF16 defaults and tensor-parallel sharding."""

    def __init__(self, config: HYV4Config) -> None:
        with default_dtype(DType.bfloat16):
            super().__init__(config)
            if len(config.devices) > 1:
                self.shard()


mtp_draft_registration = DraftRegistration[HYV4MTPDraft, HYV4Config](
    method="mtp", configure=configure_mtp_draft
)
