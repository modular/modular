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
"""Input batching for MPNet embedding models."""

from __future__ import annotations

from max.pipelines.lib.registration_encoder import PaddedEncoderProcessor


class MPNetBatchProcessor(PaddedEncoderProcessor):
    """Fixed-shape padded batching for MPNet encoder embedding models."""

    def _pad_token_id(self) -> int:
        # MPNet defaults pad_token_id to 1 when absent from the HF config.
        return self.runtime.pad_token_id or 1
