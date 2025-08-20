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
"""Shared utilities for transformer components."""

from max.graph import ops
from max.pipelines.lib import PipelineConfig

ACTIVATIONS = {
    "gelu": ops.gelu,
    "relu": ops.relu,
    "silu": ops.silu,
    "sigmoid": ops.sigmoid,
    "tanh": ops.tanh,
}


def _quantization_encoding(pipeline_config: PipelineConfig):
    """Helper to get quantization encoding from pipeline config."""
    if supported_encoding := pipeline_config.model_config.quantization_encoding:
        return supported_encoding.quantization_encoding
    return None
