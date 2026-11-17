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
"""Registered Muse Glimmer text and optional vision computations."""

from __future__ import annotations

from collections import OrderedDict

from max import tree
from max._artifact.vision import (
    ModalityLayout,
    MuseSmartResize,
    PromptText,
    Span,
    VisionMetadata,
)
from max.nn import Module
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.graph_specs import GraphSpec
from max.pipelines.lib.module_registration import decoder_vision_input_types
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh, TensorLayout
from transformers import AutoTokenizer

from .image_processor import MuseGlimmerImageProcessor
from .model_config import MuseGlimmerConfig
from .muse_glimmer import MuseGlimmer
from .preprocess_graph import MuseGlimmerImagePreprocess
from .tokenizer import load_processor_config
from .vision import MuseGlimmerVisionModel, vision_input_tree


class MuseGlimmerDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the existing hybrid-KV text tower with semantic role inputs."""

    graph_name = "MuseGlimmer"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        assert config.mesh is not None
        self.model = MuseGlimmer(config, config.mesh)
        self.model.name_transparent = True

    @classmethod
    def vision_input_types(
        cls, config: MuseGlimmerConfig, mesh: DeviceMesh
    ) -> VisionInputs[TensorLayout]:
        """Preserves empty image fields when a checkpoint has no vision tower."""
        return decoder_vision_input_types(
            mesh, config.dtype, config.text_config.hidden_size
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs the text computation with the registered KV and image trees."""
        assert inputs.vision is not None
        outputs = self.model(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
            *tree.leaves(inputs.kv),
        )
        return DecoderOutputs(
            next_token_logits=outputs[0],
            logits=outputs[1] if len(outputs) == 3 else None,
            logit_offsets=outputs[2] if len(outputs) == 3 else None,
        )


class MuseGlimmerVision(MuseGlimmerVisionModel):
    """Constructs the image tower only for checkpoints that declare one."""

    graph_name = "MuseGlimmerVisionModel"

    @classmethod
    def is_enabled(cls, config: MuseGlimmerConfig) -> bool:
        """Returns whether this checkpoint includes a vision tower."""
        return config.vision_config is not None

    def __init__(self, config: MuseGlimmerConfig) -> None:
        assert config.vision_config is not None
        with default_dtype(config.dtype):
            super().__init__(
                config.vision_config,
                config.text_config.hidden_size,
                config.text_config.rms_norm_eps,
            )
        self.config = config
        self.embedding_dtype = config.dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> OrderedDict[str, object]:
        """Groups the tower's inputs as a serving engine packs them."""
        assert self.config.vision_config is not None
        return vision_input_tree(
            self.config.vision_config, self.config.devices[0], self.config.dtype
        )

    @classmethod
    def vision_metadata(
        cls, pipeline_config: PipelineConfig, config: MuseGlimmerConfig
    ) -> VisionMetadata:
        """Returns the image layout the tokenizer sizes and splices images by."""
        assert config.vision_config is not None
        model = pipeline_config.model
        tokenizer = AutoTokenizer.from_pretrained(
            model.model_path,
            revision=model.huggingface_model_revision,
            trust_remote_code=model.trust_remote_code,
        )
        begin, end = tokenizer.convert_tokens_to_ids(
            ["<|image_start|>", "<|image_end|>"]
        )
        huggingface_config = model.huggingface_config
        assert huggingface_config is not None
        return muse_glimmer_vision_metadata(
            _image_processor(pipeline_config),
            huggingface_config.image_token_id,
            begin,
            end,
        )

    @classmethod
    def vision_preprocess_spec(
        cls, pipeline_config: PipelineConfig, config: MuseGlimmerConfig
    ) -> GraphSpec:
        """Returns the graph that preprocesses one image for the tower."""
        assert config.vision_config is not None
        return GraphSpec(
            MuseGlimmerImagePreprocess(
                _image_processor(pipeline_config),
                config.vision_config,
                config.dtype,
            ),
            "muse_glimmer_vision_preprocess",
            image_preprocess_input_types(),
            weights={},
        )


def muse_glimmer_vision_metadata(
    processor: MuseGlimmerImageProcessor,
    image_token_id: int,
    image_start_id: int,
    image_end_id: int,
) -> VisionMetadata:
    """Returns the image layout a Muse Glimmer checkpoint is served with.

    Each image's run is ``<|image_start|>``, a ``<|patch|>`` per merged
    patch, and ``<|image_end|>``, as :class:`MuseGlimmerProcessor` expands
    the chat template's placeholder. The model serves no video.

    Args:
        processor: The processor the tokenizer sizes images with.
        image_token_id: The id of ``<|patch|>``.
        image_start_id: The id of ``<|image_start|>``.
        image_end_id: The id of ``<|image_end|>``.
    """
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=processor.patch_size,
            merge_size=processor.merge_size,
            strategy=MuseSmartResize(processor.max_image_tokens),
            span=Span(image_token_id, image_start_id, image_end_id),
        ),
        video=None,
        max_image_bytes=0,
        max_video_bytes=0,
        prompt=PromptText(),
    )


def _image_processor(
    pipeline_config: PipelineConfig,
) -> MuseGlimmerImageProcessor:
    """Returns the processor the checkpoint's tokenizer sizes images with."""
    model = pipeline_config.model
    config = load_processor_config(
        model.model_path, model.huggingface_model_revision
    )
    return MuseGlimmerImageProcessor(config["image_processor"])
