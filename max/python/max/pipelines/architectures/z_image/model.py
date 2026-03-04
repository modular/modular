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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import Device
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .model_config import ZImageConfig
from .weight_adapters import convert_z_image_transformer_state_dict
from .z_image import ZImageTransformer2DModel


class ZImageTransformerModel(ComponentModel):
    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        super().__init__(
            config,
            encoding,
            devices,
            weights,
        )
        self.config = ZImageConfig.generate(config, encoding, devices)
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        target_dtype = self.config.dtype
        state_dict = {}
        for key, value in self.weights.items():
            weight = value.data()
            if weight.dtype != target_dtype:
                if weight.dtype.is_float() and target_dtype.is_float():
                    weight = weight.astype(target_dtype)
            state_dict[key] = weight
        state_dict = convert_z_image_transformer_state_dict(state_dict)

        with F.lazy():
            transformer = ZImageTransformer2DModel(self.config)
            transformer.to(self.devices[0])

        self.model = transformer.compile(
            *transformer.input_types(),
            weights=state_dict,
        )
        return self.model

    @staticmethod
    def _default_ids(
        seq_len: int,
        axes: int,
        start_index: int,
        device: Device,
    ) -> Tensor:
        ids = np.zeros((seq_len, axes), dtype=np.int64)
        ids[:, 0] = np.arange(start_index, start_index + seq_len, dtype=np.int64)
        return Tensor.from_dlpack(ids).to(device)

    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor | None = None,
        txt_ids: Tensor | None = None,
        controlnet_block_samples: Tensor | None = None,
        siglip_feats: Tensor | None = None,
        image_noise_mask: Tensor | None = None,
    ) -> Any:
        if controlnet_block_samples is not None:
            raise NotImplementedError(
                "controlnet_block_samples is not supported in z-image phase 1"
            )
        if siglip_feats is not None or image_noise_mask is not None:
            raise NotImplementedError(
                "Omni(siglip/image_noise_mask) is not supported in z-image phase 1"
            )

        axes = len(self.config.axes_dims)
        txt_len = int(encoder_hidden_states.shape[1])
        img_len = int(hidden_states.shape[1])

        if txt_ids is None:
            txt_ids = self._default_ids(txt_len, axes, 1, hidden_states.device)
        if img_ids is None:
            img_ids = self._default_ids(
                img_len,
                axes,
                txt_len + 1,
                hidden_states.device,
            )

        return self.model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
