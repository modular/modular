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
"""Inkling with MTP: the chained draft depths on the shared driver."""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..inkling.inkling import Inkling
from ..inkling.model_config import InklingConfig
from .inkling_mtp import InklingMultiTokenPredictor
from .spec_adapters import InklingMTPProposer, InklingTarget

__all__ = ["UnifiedMTPInkling"]


class UnifiedMTPInkling(SequentialDriver[list[TensorValue]]):
    """Inkling target plus chained MTP depths, driven by the shared phases.

    The graph declares more than the canonical spec-decode signature: a
    position ramp, Inkling's single-tensor vision inputs, and the cache slot
    indices and conv-state pools its Mamba-style layers thread through. Those
    ride the trailing group and reach the adapters as batch extras, so the
    phases here are the ordinary seven.
    """

    target: Inkling
    draft: InklingMultiTokenPredictor

    def __init__(
        self,
        config: InklingConfig,
        draft: InklingMultiTokenPredictor,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        target_config = config
        kv = config.kv_params
        if "target" in kv.children:
            target_kv = kv.children["target"]
            assert isinstance(target_kv, MultiKVCacheParams)
            target_config = replace(config, kv_params=target_kv)
        target = Inkling(
            target_config,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        )
        target_adapter = InklingTarget(target, target_config)
        super().__init__(
            target_adapter,
            InklingMTPProposer(draft, target_adapter),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                # Tensor-parallel only -- no replica split, no host mirrors --
                # but the embedding and LM head are collective, so signal
                # buffers are still needed.
                distributed=False,
                include_signal_buffers=len(config.devices) > 1,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
            # The head ships the depths it was trained with, which the config's
            # speculative width can exceed.
            num_draft_steps=draft.n_depths,
        )
        self.config = config

    @override
    @property
    def has_trailing_inputs(self) -> bool:
        return True

    @override
    def input_types(
        self, kv_params: KVCacheParamInterface | None = None
    ) -> tuple[TensorType | BufferType, ...]:
        """The canonical signature, then Inkling's own inputs."""
        devices = self.config.devices
        device = devices[0]
        return (
            *super().input_types(kv_params),
            TensorType(DType.uint32, shape=["total_seq_len"], device=device),
            TensorType(
                self.config.dtype,
                shape=[
                    "total_image_tokens",
                    self.config.text_config.hidden_size,
                ],
                device=device,
            ),
            TensorType(
                DType.int32, shape=["total_image_tokens"], device=device
            ),
            *(
                TensorType(DType.uint32, shape=["batch_size"], device=dev)
                for dev in devices
            ),
            *(
                TensorType(DType.bool, shape=["batch_size"], device=dev)
                for dev in devices
            ),
            *self.target.conv_layout.buffer_types(devices),
            *self.draft.conv_layout.buffer_types(devices),
        )
