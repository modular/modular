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

from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.interfaces.component_model import ComponentModel
from PIL import Image

from .clip_encoder import CLIPVisionModel
from .model_config import ClipVisionConfig
from .weight_adapters import convert_safetensor_state_dict


class ClipModel(ComponentModel):
    CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = ClipVisionConfig.initialize_from_config(
            config,
            encoding,
            devices,
        )
        self.session = session or InferenceSession(devices=devices)
        self.load_model()

    def load_model(self) -> Model:
        assert self.weights is not None, "Weights already freed"
        state_dict = convert_safetensor_state_dict(self.weights)
        device = self.devices[0]
        device_ref = DeviceRef.from_device(device)

        pixel_values_type = TensorType(
            self.config.dtype,
            shape=[
                "batch_size",
                self.config.num_channels,
                self.config.image_size,
                self.config.image_size,
            ],
            device=device,
        )

        with Graph("clip_vision", input_types=[pixel_values_type]) as graph:
            vision_model = CLIPVisionModel(
                self.config,
                dtype=self.config.dtype,
                device=device_ref,
            )
            vision_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=True,
            )
            (pixel_values,) = graph.inputs
            graph.output(vision_model(pixel_values.tensor))

        self.model = self.session.load(
            graph,
            weights_registry=vision_model.state_dict(),
        )
        self.weights = None  # type: ignore[assignment]
        return self.model

    def preprocess_image(self, image: Any) -> np.ndarray:
        """Center-crop and normalize an input image for CLIP vision."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("RGB")
        size = self.config.image_size

        width, height = image.size
        if height < width:
            new_height = size
            new_width = int(width * size / height + 0.5)
        else:
            new_width = size
            new_height = int(height * size / width + 0.5)
        image = image.resize((new_width, new_height), Image.BICUBIC)

        width, height = image.size
        left = (width - size) // 2
        top = (height - size) // 2
        image = image.crop((left, top, left + size, top + size))

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(self.CLIP_IMAGE_MEAN, dtype=np.float32)
        std = np.asarray(self.CLIP_IMAGE_STD, dtype=np.float32)
        pixels = (pixels - mean) / std
        return pixels.transpose(2, 0, 1)[np.newaxis]

    def encode(self, image: Any) -> Buffer:
        """Encode a single image into CLIP vision features."""
        pixels = self.preprocess_image(image)
        if self.config.dtype == DType.bfloat16:
            u16 = float32_to_bfloat16_as_uint16(np.ascontiguousarray(pixels))
            pixel_buf = (
                Buffer.from_numpy(u16)
                .to(self.devices[0])
                .view(dtype=DType.bfloat16, shape=pixels.shape)
            )
        else:
            pixel_buf = Buffer.from_numpy(np.ascontiguousarray(pixels)).to(
                self.devices[0]
            )

        result = self.model.execute(pixel_buf)[0]
        return result
