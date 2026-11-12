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
"""DiffusionGemma's registered roles.

``DiffusionGemmaEncoder`` is the ``DECODER``, the causal encoder that commits
tokens into the KV cache, and ``DiffusionGemmaDenoiser`` is the
``BLOCK_DIFFUSION`` denoise step. ``DiffusionGemmaVision`` is built and
weighted but not run, since serving is text-only.
"""

from __future__ import annotations

from max import tree
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.lib.role_contracts import (
    BlockDiffusionInputs,
    BlockDiffusionOutputs,
    DecoderInputs,
    DecoderOutputs,
    VisionInputs,
)
from max.tensor import Tensor, default_dtype, defaults
from max.tensor.sharding import DeviceMesh

from .model_config import DiffusionGemmaForBlockDiffusionConfig
from .text_model import (
    DiffusionGemmaDecoderTextModel,
    DiffusionGemmaEncoderTextModel,
    unflatten_kv,
)
from .vision_model import Gemma4VisionModel


def _module_dtype(config: DiffusionGemmaForBlockDiffusionConfig) -> DType:
    """The dtype the text towers' unquantized leaves build in.

    Under NVFP4 the checkpoint's model dtype is packed uint8, which only the
    quantized expert projections consume; every other leaf builds in the
    checkpoint's unquantized dtype, matching the graph arch's
    ``unquantized_dtype`` handling.
    """
    if config.text_config.quant_config is not None:
        return config.unquantized_dtype
    return config.dtype


class DiffusionGemmaEncoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """Runs the causal encoder tower with the shared semantic input tree."""

    graph_name = "DiffusionGemmaEncoder"
    consumes_vision = True
    input_layout = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "vision",
        "kv",
    )

    def __init__(self, config: DiffusionGemmaForBlockDiffusionConfig) -> None:
        super().__init__()
        _, mesh = defaults()
        with default_dtype(_module_dtype(config)):
            self.model = DiffusionGemmaEncoderTextModel(config, mesh)
        self.model.name_transparent = True
        self.kv_params = config.kv_params
        self.mesh = mesh

    @classmethod
    def vision_input_types(
        cls,
        config: DiffusionGemmaForBlockDiffusionConfig,
        mesh: DeviceMesh,
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
        sliding_kv, global_kv = unflatten_kv(
            self.kv_params, tuple(tree.leaves(inputs.kv)), self.mesh
        )
        output = self.model(
            inputs.tokens,
            sliding_kv,
            global_kv,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            inputs.vision.embeddings,
            inputs.vision.scatter_indices,
        )
        return DecoderOutputs(
            next_token_logits=output[0],
            logits=output[1] if len(output) == 3 else None,
            logit_offsets=output[2] if len(output) == 3 else None,
        )


class DiffusionGemmaDenoiser(
    Module[[BlockDiffusionInputs[Tensor]], BlockDiffusionOutputs]
):
    """One block-diffusion denoise step over a canvas.

    The registered ``BLOCK_DIFFUSION`` role. The generated pipeline model
    drives it through ``execute_denoise_step`` with the canvas buffers the
    block-diffusion pipeline prepares.
    """

    graph_name = "DiffusionGemmaDenoiser"

    def __init__(self, config: DiffusionGemmaForBlockDiffusionConfig) -> None:
        super().__init__()
        _, mesh = defaults()
        with default_dtype(_module_dtype(config)):
            self.model = DiffusionGemmaDecoderTextModel(config, mesh)
        self.model.name_transparent = True
        self.kv_params = config.kv_params
        self.mesh = mesh
        self.vocab_size = config.text_config.vocab_size

    def forward(
        self, inputs: BlockDiffusionInputs[Tensor]
    ) -> BlockDiffusionOutputs:
        """Runs one denoise step over the canvas tokens."""
        sliding_kv, global_kv = unflatten_kv(
            self.kv_params, tuple(tree.leaves(inputs.kv)), self.mesh
        )
        return self.model(
            inputs.tokens,
            sliding_kv,
            global_kv,
            inputs.input_row_offsets,
            inputs.sc_logits,
            inputs.sc_enabled,
            inputs.temperature,
        )


class DiffusionGemmaVision(Gemma4VisionModel):
    """Constructs the vision tower in its unquantized dtype."""

    graph_name = "DiffusionGemmaVision"

    @classmethod
    def is_enabled(cls, config: DiffusionGemmaForBlockDiffusionConfig) -> bool:
        """Every DiffusionGemma checkpoint carries the full vision tower."""
        return config.vision_config is not None

    def __init__(self, config: DiffusionGemmaForBlockDiffusionConfig) -> None:
        with default_dtype(config.unquantized_dtype):
            super().__init__(config)
        self.config = config
        self.embedding_dtype = config.unquantized_dtype
        self.embedding_dim = config.text_config.hidden_size

    def input_types(self) -> tuple[TensorType, ...]:
        """Returns the tower's patch, position and pooling inputs."""
        config = self.config
        assert config.vision_config is not None
        device = config.devices[0]
        patch_dim = 3 * config.vision_config.patch_size**2
        return (
            TensorType(
                config.unquantized_dtype,
                ["total_patches", patch_dim],
                device=device,
            ),
            TensorType(DType.int32, ["total_patches", 2], device=device),
            TensorType(DType.uint32, ["num_images_plus_1"], device=device),
            TensorType(
                DType.int32,
                ["num_pooled_tokens", "max_pool_patches"],
                device=device,
            ),
            TensorType(DType.uint32, [], device=DeviceRef.CPU()),
        )


__all__ = [
    "DiffusionGemmaDenoiser",
    "DiffusionGemmaEncoder",
    "DiffusionGemmaVision",
]
