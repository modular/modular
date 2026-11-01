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
"""Text-only batching for Muse Spark."""

from __future__ import annotations

from collections.abc import Sequence

from max.driver import Buffer
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.context import TextContext
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
)


class MuseSparkBatchProcessor(DecoderBatchProcessor):
    """Rejects media until Spark's vision computation is implemented."""

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Batches text and preserves the existing rejection of image inputs."""
        if any(
            getattr(context, "images", ())
            for batch in replica_batches
            for context in batch
        ):
            raise ValueError("Muse Spark does not support image inputs yet.")
        return super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
