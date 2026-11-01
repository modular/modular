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
"""Constructs Inkling's registered MTP depths."""

from max.pipelines.lib.registration_draft import DraftRegistration

from ..inkling.mtp import InklingMultiTokenPredictor
from .model_config import MTPInklingDraftArchConfig, configure_mtp_draft


class InklingMTPDraft(InklingMultiTokenPredictor):
    """Builds the checkpoint's selected depths from their finalized config."""

    def __init__(self, config: MTPInklingDraftArchConfig) -> None:
        super().__init__(config, config.num_draft_depths, config.kv_params)


mtp_draft_registration = DraftRegistration[
    InklingMTPDraft, MTPInklingDraftArchConfig
](method="mtp", configure=configure_mtp_draft)
