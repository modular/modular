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
"""Text-only Qwen3.5 checkpoints (``Qwen3_5ForCausalLM``).

Such a checkpoint has no vision tower, so its graph takes none of the
vision-merge inputs and the batch must not pass any, even though the pipeline
fills them in for every vision-capable model.
"""

from __future__ import annotations

from max.driver import CPU, Buffer
from max.dtype import DType
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.architectures.qwen3_5.model import Qwen3_5Inputs
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.modeling.types import InputModality


def test_text_only_arch_resolves_by_name() -> None:
    arch = PIPELINE_REGISTRY.retrieve_architecture("Qwen3_5ForCausalLM")
    assert arch is not None
    assert arch.input_modalities == {InputModality.TEXT}
    assert arch.tokenizer is Qwen3_5Tokenizer
    # It reuses the multimodal arch's model and config, only the name and
    # modalities differ.
    full = PIPELINE_REGISTRY.retrieve_architecture(
        "Qwen3_5ForConditionalGeneration"
    )
    assert full is not None
    assert arch.pipeline_model is full.pipeline_model
    assert arch.config is full.config


def _inputs(vision_inputs_in_graph: bool) -> Qwen3_5Inputs:
    cpu = CPU()

    def buffer(size: int) -> Buffer:
        return Buffer.zeros([size], DType.int32, device=cpu)

    inputs = Qwen3_5Inputs(
        tokens=buffer(1),
        input_row_offsets=buffer(2),
        signal_buffers=[],
        return_n_logits=buffer(1),
        vision_inputs_in_graph=vision_inputs_in_graph,
    )
    # What the pipeline's vision seam sets on every prepared batch.
    inputs.vision_embeddings = [buffer(0)]
    inputs.vision_scatter_indices = [buffer(0)]
    return inputs


def test_vision_inputs_are_passed_only_when_the_graph_declares_them() -> None:
    with_vision = _inputs(vision_inputs_in_graph=True)
    without_vision = _inputs(vision_inputs_in_graph=False)
    assert len(with_vision.buffers) == len(without_vision.buffers) + 2
    assert all(
        buffer not in without_vision.buffers
        for buffer in without_vision.vision_embeddings
        + without_vision.vision_scatter_indices
    )
