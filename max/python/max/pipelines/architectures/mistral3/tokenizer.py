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
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        root_model_path: str | None = None,
        **unused_kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            pipeline_config=pipeline_config,
            revision=revision,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )
        
        # Store root_model_path for chat_template loading
        self._root_model_path = root_model_path

        self._load_and_set_chat_template(
            revision=revision, pipeline_config=pipeline_config
        )

    def _load_and_set_chat_template(
        self,
        revision: str | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        """Load chat template from chat_template.json file and set it on the tokenizer."""

        if revision is None:
            # Prefer revision from pipeline config when not explicitly provided.
            model_cfg = getattr(pipeline_config, "model", None)
            candidate = getattr(model_cfg, "huggingface_model_revision", None)
            revision = (
                candidate if isinstance(candidate, str) and candidate else None
            )
        revision = revision or "main"

        # Check if model_path is a local file path or HuggingFace repo ID
        import os
        from pathlib import Path
        
        template_file_path = None
        is_jinja_file = False
        
        # Try both chat_template.json and chat_template.jinja
        template_files = [
            ("chat_template.json", False),
            ("chat_template.jinja", True),
        ]
        
        # First, try to load from local model_path (text_encoder subdirectory)
        is_local_path = os.path.exists(self.model_path) or Path(self.model_path).exists()
        if is_local_path:
            # For local paths, check files directly
            for filename, is_jinja in template_files:
                local_file = Path(self.model_path) / filename
                if local_file.exists():
                    template_file_path = str(local_file)
                    is_jinja_file = is_jinja
                    logger.info(f"Found {filename} at local path: {template_file_path}")
                    break
        
        # If not found locally and root_model_path is available, try from root model
        if not template_file_path and self._root_model_path:
            # For HuggingFace repos, try cache first then download
            # Try both root directory and text_encoder subdirectory
            search_paths = [
                "",  # Root directory
                "text_encoder/",  # text_encoder subdirectory
            ]
            for search_path in search_paths:
                for filename, is_jinja in template_files:
                    full_filename = search_path + filename if search_path else filename
                    try:
                        cached_path = try_to_load_from_cache(
                            repo_id=self._root_model_path,
                            filename=full_filename,
                            revision=revision,
                        )
                        # try_to_load_from_cache returns object() singleton if file not found
                        if cached_path and isinstance(cached_path, (str, os.PathLike)):
                            template_file_path = cached_path
                            is_jinja_file = is_jinja
                            break
                    except (ValueError, Exception):
                        pass
                if template_file_path:
                    break
            
            # If not in cache, try to download
            if not template_file_path:
                for search_path in search_paths:
                    for filename, is_jinja in template_files:
                        full_filename = search_path + filename if search_path else filename
                        try:
                            template_file_path = huggingface_hub.hf_hub_download(
                                repo_id=self._root_model_path,
                                filename=full_filename,
                                revision=revision,
                            )
                            is_jinja_file = is_jinja
                            logger.info(f"Successfully downloaded {full_filename}")
                            break
                        except Exception:
                            continue
                    if template_file_path:
                        break
        
        # If still not found, try model_path as HuggingFace repo ID
        if not template_file_path and not is_local_path:
            for filename, is_jinja in template_files:
                try:
                    cached_path = try_to_load_from_cache(
                        repo_id=self.model_path,
                        filename=filename,
                        revision=revision,
                    )
                    # try_to_load_from_cache returns object() singleton if file not found
                    if cached_path and isinstance(cached_path, (str, os.PathLike)):
                        template_file_path = cached_path
                        is_jinja_file = is_jinja
                        break
                except (ValueError, Exception):
                    pass
            
            if not template_file_path:
                for filename, is_jinja in template_files:
                    try:
                        template_file_path = huggingface_hub.hf_hub_download(
                            repo_id=self.model_path,
                            filename=filename,
                            revision=revision,
                        )
                        is_jinja_file = is_jinja
                        logger.info(f"Successfully downloaded {filename}")
                        break
                    except Exception as e:
                        if filename == template_files[-1][0]:  # Last file to try
                            raise RuntimeError(
                                f"Failed to download 'chat_template.json' or 'chat_template.jinja' "
                                f"from model repo '{self.model_path}' at revision '{revision}': {e}"
                            ) from e
                        continue

        # If no template file found, use tokenizer's default if available
        if not template_file_path:
            if hasattr(self.delegate, "chat_template") and self.delegate.chat_template:
                logger.info(
                    f"No chat template file found, using tokenizer's default for {self.model_path}"
                )
                return
            else:
                raise RuntimeError(
                    f"Failed to find 'chat_template.json' in model path '{self.model_path}' "
                    f"or root model path '{self._root_model_path}'"
                )

        # Load and set the chat template
        try:
            with open(template_file_path, encoding="utf-8") as f:
                if is_jinja_file:
                    chat_template = f.read().strip()
                else:
                    content = f.read().strip()
                    
                    # Check if file is empty
                    if not content:
                        logger.warning(
                            f"chat_template.json is empty at {template_file_path}, "
                            "using tokenizer's default if available"
                        )
                        if hasattr(self.delegate, "chat_template") and self.delegate.chat_template:
                            return
                        else:
                            raise ValueError(
                                f"chat_template.json is empty and tokenizer has no default "
                                f"template at {template_file_path}"
                            )
                    
                    template_data = json.loads(content)
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
