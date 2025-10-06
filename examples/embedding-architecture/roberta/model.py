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

"""RoBERTa model implementation that extends BERT."""

import logging
import os

# Import the base BERT model - RoBERTa inherits from BERT
import sys

from max.driver import Device
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig, SupportedEncoding
from transformers import AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from bert.model import BertPipelineModel

from .weight_adapters import RobertaWeightsAdapter

logger = logging.getLogger(__name__)


class RobertaPipelineModel(BertPipelineModel):
    """RoBERTa model that inherits from BERT with RoBERTa-specific adaptations."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        # Auto-select RoBERTa adapter if not provided
        if adapter is None:
            adapter = RobertaWeightsAdapter()
            logger.info("Using RobertaWeightsAdapter")

        # Call parent BERT model initialization
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

    # RoBERTa uses the same graph structure as BERT
    # The main differences are:
    # 1. Weight naming (handled by adapter)
    # 2. Tokenizer (has EOS token)
    # 3. No token type embeddings in some variants
    # All these are handled by the parent BERT implementation
