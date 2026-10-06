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
"""Input batching for the Muse Glimmer ModuleV3 pipeline."""

from __future__ import annotations

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.batch_processor import (
    ModuleV3SingleReplicaBatchProcessor,
)

from .inputs import MuseGlimmerInputs


class MuseGlimmerBatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextAndVisionContext, MuseGlimmerInputs]
):
    """Ragged batching for Muse Glimmer (single GPU, no signals).

    Only tokens and offsets are built here; the pipeline's
    ``VisionEncoderCache`` fills the base vision fields.
    """

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> MuseGlimmerInputs:
        return MuseGlimmerInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )
