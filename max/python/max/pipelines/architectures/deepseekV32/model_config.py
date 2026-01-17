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
"""Config for DeepseekV3.2 models."""

from __future__ import annotations

from max.pipelines.architectures.deepseekV3.model_config import DeepseekV3Config


class DeepseekV32Config(DeepseekV3Config):
    """Configuration for DeepseekV3.2 models."""

    # Added parameters for the Indexer used in DeepSeek Sparse Attention.
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
