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
"""d1: an LFM3-VL or LFM3-Omni backbone read at its answer slots.

A d1 prompt packs a state and a set of questions, each question in its own
segment. Full-attention layers read the state and each question both ways,
and a question never reads another; sliding and convolution layers stay
causal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from max import functional as F
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import ModelInputs, ModuleV3Outputs
from max.tensor import Tensor

from ..lfm3_audio.model_config import Lfm3AudioEncoderConfig
from ..lfm3_omni.model import Lfm3ListenerModel
from ..lfm3_vl.lfm3_vl import Lfm3VlLanguage
from ..lfm3_vl.model import Lfm3VlInputs
from .model_config import D1Config


@tree.dataclass(kw_only=True)
class D1Outputs(ModuleV3Outputs):
    """The decoder outputs, with the answer slots' reads."""

    slot_logits: Tensor
    """The logits at each label slot, shape ``[label_slots, V]``."""

    pointer_logits: Tensor
    """Each pointer read's scores over the batch's tokens."""


class D1Language(Lfm3VlLanguage):
    """The LFM3 decoder under segment attention.

    Returns the logits at the label slots and each pointer read's scores over
    the prompt.
    """

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *inputs: Tensor,
    ) -> D1Outputs:
        """Runs the decoder over the media, segment and slot inputs.

        Takes the media inputs, each token's segment and the slot rows ahead of
        the cache inputs.
        """
        del return_n_logits
        media, indices, segment_ids, slot_rows, pointer_rows, *kv_inputs = (
            inputs
        )
        embeds = self.embed(tokens, media, indices)
        kv, conv_pool, conv_rows = self.unpack_kv(
            tuple(kv_inputs), embeds.mapping
        )
        h = self.model.embedding_norm(
            self.model(
                embeds, kv, conv_pool, conv_rows, input_row_offsets, segment_ids
            )
        )
        head = self.model.embed_tokens.weight.T

        def logits(rows: Tensor) -> Tensor:
            return F.cast(F.gather(h, rows, axis=0) @ head, DType.float32)

        states = F.cast(h, DType.float32)
        return D1Outputs(
            next_token_logits=logits(input_row_offsets[1:] - 1),
            slot_logits=logits(slot_rows),
            pointer_logits=F.gather(states, pointer_rows, axis=0)
            @ states.T
            / math.sqrt(int(h.shape[-1])),
        )


class D1Model(Lfm3ListenerModel):
    """d1 on either backbone.

    The engine that scores answer slots feeds each token's segment and the
    slot rows; a request without answer slots reads its whole prompt both
    ways.
    """

    model_config_cls: ClassVar[type[Any]] = D1Config
    language_cls: ClassVar[type[Lfm3VlLanguage]] = D1Language
    _config: D1Config

    @property
    def audio_encoder(self) -> Lfm3AudioEncoderConfig | None:
        """The speech encoder's config, or ``None`` if the model can't hear."""
        return self._config.encoder

    def _language_input_types(
        self, model_config: Any
    ) -> list[TensorType | BufferType]:
        types = super()._language_input_types(model_config)
        device = DeviceRef.from_device(self.devices[0])
        return [
            *types[:5],
            TensorType(DType.int32, ["total_tokens"], device=device),
            TensorType(DType.int64, ["label_slots"], device=device),
            TensorType(DType.int64, ["pointer_reads"], device=device),
            *types[5:],
        ]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Adds the segment and answer slot inputs.

        This pipeline scores no answer slots, so each prompt is one segment
        read both ways.
        """
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        assert isinstance(inputs, Lfm3VlInputs)
        n = sum(
            ctx.tokens.active_length
            for batch in replica_batches
            for ctx in batch
        )
        device = self.devices[0]
        inputs.extras = tuple(
            Buffer.from_numpy(values).to(device)
            for values in (
                np.full(n, -1, np.int32),
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
            )
        )
        return inputs
