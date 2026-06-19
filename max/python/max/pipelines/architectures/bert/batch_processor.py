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
"""Input batching for BERT embedding models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelOutputs
from max.pipelines.modeling.dataprocessing import collate_batch

if TYPE_CHECKING:
    from .model import BertInputs


class BertBatchProcessor(BatchProcessor[TextContext, "BertInputs"]):
    """Fixed-shape padded batching for encoder-only BERT models."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        del kv_params
        device_ref = device_refs[0]
        return [
            TensorType(
                DType.int64,
                shape=["batch_size", "seq_len"],
                device=device_ref,
            ),
            TensorType(
                DType.float32,
                shape=["batch_size", "seq_len"],
                device=device_ref,
            ),
        ]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> BertInputs:
        from .model import BertInputs

        del kv_cache_inputs, return_n_logits
        if len(replica_batches) > 1:
            raise ValueError("BertBatchProcessor does not support DP>1")

        context_batch = replica_batches[0]
        device0 = self.runtime.devices[0]
        tokens = [ctx.tokens.active for ctx in context_batch]
        pad_value = self.runtime.pad_token_id
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)
        return BertInputs(
            next_tokens_batch=Buffer.from_numpy(next_tokens_batch).to(device0),
            attention_mask=Buffer.from_numpy(attention_mask).to(device0),
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])
