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
"""Gemma4 decoder and optional image encoder role implementations."""

from __future__ import annotations

from collections import OrderedDict

from max import tree
from max._artifact.vision import VisionMetadata
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.pipelines.architectures.gemma4.vision_model.vision_spec import (
    gemma4_checkpoint_vision_metadata,
    gemma4_vision_preprocess_spec,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.graph_specs import GraphSpec
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh

from .gemma4 import Gemma4
from .model_config import Gemma4Config
from .vision_model.vision_model import Gemma4VisionModel


class Gemma4Decoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the existing language tower with the shared semantic input tree."""

    graph_name = "Gemma4"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        assert config.runtime is not None
        dtype = (
            config.unquantized_dtype
            if config.text_config.quant_config is not None
            else config.dtype
        )
        with default_dtype(dtype):
            self.model = Gemma4(config, config.kv_params, config.runtime.mesh)
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: Gemma4Config, mesh: DeviceMesh
    ) -> VisionInputs[TensorType]:
        """Declares device-zero image inputs even for text-only checkpoints."""
        device = DeviceRef.from_device(mesh.devices[0])
        return VisionInputs(
            embeddings=TensorType(
                config.unquantized_dtype,
                ["num_image_tokens", config.text_config.hidden_size],
                device=device,
            ),
            scatter_indices=TensorType(
                DType.int32, ["total_image_tokens"], device=device
            ),
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs text and image rows, retaining the existing KV layout."""
        assert inputs.vision is not None
        output = self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        # The outputs are replicated, and a serving engine reads them from one
        # device.
        output = tuple(t.local_shards[0] for t in output)
        return DecoderOutputs(
            next_token_logits=output[0],
            logits=output[1] if len(output) == 3 else None,
            logit_offsets=output[2] if len(output) == 3 else None,
        )


class Gemma4Vision(Gemma4VisionModel):
    """Constructs the optional image tower in its unquantized dtype."""

    graph_name = "Gemma4VisionModel"

    @classmethod
    def is_enabled(cls, config: Gemma4Config) -> bool:
        """Skips the encoder for the text-only Unified checkpoint line."""
        return config.vision_config is not None

    def __init__(self, config: Gemma4Config) -> None:
        with default_dtype(config.unquantized_dtype):
            super().__init__(config)
        self.config = config
        self.embedding_dtype = config.unquantized_dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> OrderedDict[str, object]:
        """Groups the encoder's inputs as a serving engine packs them.

        The groups and their fields are the graph-API Gemma 4's, on the one
        device the encoder runs on.
        """
        config = self.config
        assert config.vision_config is not None
        device = config.devices[0]
        patch_dim = 3 * config.vision_config.patch_size**2
        return OrderedDict(
            patches=VisionRows(
                patches_flat=TensorType(
                    config.unquantized_dtype,
                    ["total_patches", patch_dim],
                    device=device,
                ),
                pixel_position_ids=TensorType(
                    DType.int32, ["total_patches", 2], device=device
                ),
            ),
            segments=VisionSegments(
                cu_seqlens=TensorType(
                    DType.uint32, ["num_images_plus_1"], device=device
                )
            ),
            pooling=VisionRowIndex(
                pool_gather_index=TensorType(
                    DType.int32,
                    ["num_pooled_tokens", "max_pool_patches"],
                    device=device,
                )
            ),
            segment_bound=VisionHostMax(
                max_seq_len=TensorType(DType.uint32, [], device=DeviceRef.CPU())
            ),
        )

    @classmethod
    def vision_metadata(
        cls, pipeline_config: PipelineConfig, config: Gemma4Config
    ) -> VisionMetadata | None:
        """Returns the media layout the checkpoint is served with."""
        del config
        return gemma4_checkpoint_vision_metadata(pipeline_config)

    @classmethod
    def vision_preprocess_spec(
        cls, pipeline_config: PipelineConfig, config: Gemma4Config
    ) -> GraphSpec:
        """Returns the graph-API Gemma 4's image preprocessing graph."""
        return gemma4_vision_preprocess_spec(
            pipeline_config, config.unquantized_dtype
        )
