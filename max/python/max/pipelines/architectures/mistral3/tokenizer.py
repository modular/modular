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

"""Mistral-specific tokenizer implementation."""

from __future__ import annotations

import json
import logging

import huggingface_hub
from max.pipelines.lib import TextTokenizer, try_to_load_from_cache
from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")


class Mistral3Tokenizer(TextTokenizer):
    """Mistral-specific tokenizer that corrects the chat template.

    This class only overrides __init__ to correct the chat template, while inheriting
    all other methods from TextTokenizer.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        pipeline_config: PipelineConfig | None = None,
        **unused_kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            revision=revision,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )

        self._load_and_set_chat_template(
            revision=revision, pipeline_config=pipeline_config
        )

    def _load_and_set_chat_template(
        self,
        revision: str | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        """Load chat template from chat_template.json file and set it on the tokenizer."""

        if pipeline_config and hasattr(pipeline_config, "model_config"):
            revision = (
                revision
                or pipeline_config.model_config.huggingface_model_revision
            )
        revision = revision or "main"

        # Try to load from cache first
        template_file_path = try_to_load_from_cache(
            repo_id=self.model_path,
            filename="chat_template.json",
            revision=revision,
        )

        # If not in cache, try to download
        if not template_file_path:
            logger.info(
                "chat_template.json not in cache, attempting to download..."
            )
            try:
                template_file_path = huggingface_hub.hf_hub_download(
                    repo_id=self.model_path,
                    filename="chat_template.json",
                    revision=revision,
                )
                logger.info("Successfully downloaded chat_template.json")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download 'chat_template.json' from model repo '{self.model_path}' "
                    f"at revision '{revision}': {e}"
                ) from e

        # Load and set the chat template
        try:
            with open(template_file_path) as f:
                template_data = json.load(f)
                chat_template = template_data.get("chat_template")

            if not chat_template:
                raise KeyError(
                    f"No 'chat_template' key found in {template_file_path} for model {self.model_path}"
                )

            self.delegate.chat_template = chat_template
            logger.info(
                f"Loaded custom chat template from {template_file_path}"
            )

        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to load chat template from {template_file_path}: {e}"
            ) from e
