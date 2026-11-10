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
"""Supported architecture entry for the Kimi-K3 DSpark draft."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from .config import k3_dspark_draft_width
from .model_config import K3DSparkDraftArchConfig

__all__ = ["k3_dspark_draft_arch"]

# The draft checkpoint declares architectures ["K3DSparkModel"]. It is only
# served as a --draft-model beside a Kimi K3 (ModuleV3) target, so its
# tokenizer and context type are never used.
k3_dspark_draft_arch = SupportedArchitecture(
    name="K3DSparkModel",
    example_repo_ids=["Inferact/Kimi-K3-DSpark"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=None,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    task=PipelineTask.TEXT_GENERATION,
    config=K3DSparkDraftArchConfig,
    checkpoint_draft_width=k3_dspark_draft_width,
    # The spec graph's MLA prefill plan takes its extent from the host
    # pair the spec adapter threads as a trailing input (dspark/
    # mla_extent.py), so no captured step syncs the host; the fused
    # pipeline replays bit-exact against eager execution at the same
    # aligned characteristics, per step and end to end at TP1 and TP2
    # (spec/test_spec_capture.py).
    supports_device_graph_capture=True,
)
