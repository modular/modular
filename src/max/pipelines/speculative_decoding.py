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
# mypy: disable-error-code="import-not-found"
"""Speculative Decoding Text Generation Pipeline"""

from typing import Any, TypeVar

from max.driver import load_devices, scan_available_devices
from max.engine import InferenceSession
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn import ReturnLogits

from .core import InputContext, TextGenerationResponse, TokenGenerator
from .hf_utils import download_weight_files
from .pipeline import PipelineModel

T = TypeVar("T", bound=InputContext)


class SpeculativeDecodingTextGenerationPipeline(TokenGenerator[T]):
    """Generalized token generator pipeline with speculative decoding."""

    def __init__(
        self,
        pipeline_config: Any,  # PipelineConfig
        pipeline_model: type[PipelineModel],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
    ) -> None:
        self.pipeline_config = pipeline_config

        # Load target model
        target_devices = load_devices(
            self.pipeline_config.model_config.device_specs
        )
        target_session = InferenceSession(devices=target_devices)
        target_config = self.pipeline_config.model_config.huggingface_config

        target_hf_repo = (
            self.pipeline_config.model_config.huggingface_weights_repo
        )

        weight_paths = download_weight_files(
            huggingface_model_id=target_hf_repo.repo_id,
            filenames=[
                str(x) for x in self.pipeline_config.model_config.weight_path
            ],
            revision=self.pipeline_config.model_config.huggingface_weight_revision,
            max_workers=8,
        )
        target_weights = load_weights(weight_paths)
        _target_weights_format = weights_format(weight_paths)

        if not self.pipeline_config.model_config.quantization_encoding:
            raise ValueError(
                f"quantization_encoding must be provided, {self.pipeline_config.model_config.quantization_encoding}"
            )

        self._target_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=target_session,
            huggingface_config=target_config,
            encoding=self.pipeline_config.model_config.quantization_encoding,
            devices=target_devices,
            kv_cache_config=self.pipeline_config.model_config.kv_cache_config,
            weights=target_weights,
            adapter=weight_adapters.get(_target_weights_format, None),
            return_logits=ReturnLogits.VARIABLE,
        )

        # Load draft model
        # For now, we are assuming we are placing the draft model will sit
        draft_devices = load_devices(scan_available_devices()[:1])
        draft_session = InferenceSession(devices=draft_devices)

        draft_config = (
            self.pipeline_config.draft_model_config.huggingface_config
        )

        # Retrieve Encoding, and Files for Draft Model
        if self.pipeline_config.draft_model_config is None:
            raise ValueError(
                "draft_model must be provided for speculative decoding"
            )

        draft_hf_repo = (
            self.pipeline_config.draft_model_config.huggingface_weights_repo
        )
        encodings = draft_hf_repo.supported_encodings
        if not encodings:
            raise ValueError(
                "could not identify supported encodings for draft model."
            )

        if len(encodings) > 1:
            raise ValueError(
                "repos that only support one encoding, currently supported for draft model."
            )

        # Get weight files
        weight_files = draft_hf_repo.files_for_encoding(
            encoding=encodings[0],
        )

        if not weight_files:
            raise ValueError("could not identify weight_files for draft model.")

        _draft_weights_format = list(weight_files.keys())[0]
        _draft_weight_paths = download_weight_files(
            huggingface_model_id=self.pipeline_config.draft_model_config.model_path,
            filenames=[str(x) for x in weight_files[_draft_weights_format]],
            revision=None,
            max_workers=8,
        )
        draft_weights = load_weights(_draft_weight_paths)

        self._draft_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=draft_session,
            huggingface_config=draft_config,
            encoding=encodings[0],
            devices=draft_devices,
            kv_cache_config=self.pipeline_config.draft_model_config.kv_cache_config,
            weights=draft_weights,
            adapter=weight_adapters.get(_draft_weights_format, None),
            return_logits=ReturnLogits.LAST_TOKEN,
        )

        # Check that the max length for both models are the same
        draft_seq_len = self._draft_model.calculate_max_seq_len(
            self.pipeline_config, draft_config
        )
        target_seq_len = self._target_model.calculate_max_seq_len(
            self.pipeline_config, target_config
        )
        if draft_seq_len != target_seq_len:
            msg = f"draft maximum sequence length ({draft_seq_len}) must match target maximum sequence length."
            raise ValueError(msg)

    def next_token(
        self, batch: dict[str, T], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, execute both the draft model for num_steps and the target model for num_steps + 1 tokens, accepting final tokens via rejection sampling, returning the variable list of token integers."""
        raise NotImplementedError(
            "next_token not yet implemented for SpeculativeDecodingTextGenerationPipeline"
        )

    def release(self, context: T) -> None:
        """Releases resources associated with this context.

        Args:
            context (TokenGeneratorContext): Finished context.

        """
        raise NotImplementedError(
            "release not yet implemented for SpeculativeDecodingTextGenerationPipeline"
        )
