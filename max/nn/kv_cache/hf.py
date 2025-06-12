# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from typing import Any, Optional, Union

import numpy as np
import torch
from torch import device as torch_device
from transformers import PretrainedConfig, StaticCache


class ContinuousHFStaticCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_seq_len: int,
        device: torch_device,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[
            dict[int, Union[str, torch_device, int]]
        ] = None,
    ) -> None:
        super().__init__(
            config, max_batch_size, max_seq_len, device, dtype, layer_device_map
        )
        self.max_batch_size = max_batch_size
        self.device = device
        self._init_slots()

    def _init_slots(self) -> None:
        self.available_slots: set[int] = set(range(self.max_batch_size))
        self.active_slots: list[int] = []
        self.attention_patterns: dict[int, torch.Tensor] = {}
        self.tokens: dict[int, np.ndarray] = {}
        self.cache_position: torch.Tensor = torch.arange(
            0, len(self.active_slots), device=self.device
        )

    def external_claim(self, seq_ids: list[int]) -> None:
        if not self.available_slots:
            raise RuntimeError("No slots are available for claiming.")

        unavailable_slots = set(seq_ids) - self.available_slots
        if unavailable_slots:
            raise ValueError(
                f"The following seq_ids are already claimed: {unavailable_slots}"
            )

        for seq_id in seq_ids:
            self.available_slots.remove(seq_id)
            self.attention_patterns[seq_id] = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self.tokens[seq_id] = np.array([])

    def set_active_slots(self, seq_ids: list[int]) -> None:
        self.active_slots = seq_ids

    def set_cache_position(self, cache_position: torch.Tensor):
        self.cache_position = cache_position

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = self.cache_position

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # Prepare slot indices tensor once
        slot_indices = torch.tensor(self.active_slots, device=self.device)

        # Update each sequence individually to handle different shapes correctly
        for batch_idx, slot_idx in enumerate(self.active_slots):
            k_out[slot_idx].index_copy_(
                1, cache_position, key_states[batch_idx]
            )
            v_out[slot_idx].index_copy_(
                1, cache_position, value_states[batch_idx]
            )

        # Return reordered key value cache for the layer
        return (
            k_out.index_select(0, slot_indices),
            v_out.index_select(0, slot_indices),
        )

    def update_attention_pattern(
        self, seq_id: int, attention_mask: torch.Tensor
    ) -> None:
        self.attention_patterns[seq_id] = attention_mask.to(
            device=self.device, dtype=torch.long
        )

    def get_attention_mask(self, seq_ids: list[int]) -> torch.Tensor:
        max_len = max(
            self.attention_patterns[seq_id].size(0) for seq_id in seq_ids
        )
        attention_mask = torch.zeros(
            (len(seq_ids), max_len), dtype=torch.long, device=self.device
        )

        for i, seq_id in enumerate(seq_ids):
            pattern = self.attention_patterns[seq_id]
            attention_mask[i, : pattern.size(0)] = pattern

        return attention_mask

    def release(self, seq_id: int) -> None:
        if seq_id in self.available_slots:
            raise KeyError(
                f"The seq_id {seq_id} is not currently claimed and cannot be released."
            )

        # Zero out cache tensors for the sequence
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx][seq_id].zero_()
            self.value_cache[layer_idx][seq_id].zero_()

        self.available_slots.add(seq_id)
        del self.attention_patterns[seq_id]
        del self.tokens[seq_id]

    def reset(self) -> None:
        super().reset()
        self._init_slots()
