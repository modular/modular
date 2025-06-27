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

"""Generalized Token Generation Pipeline"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import torch
from max.driver import Tensor
from max.nn.kv_cache import ContinuousHFStaticCache
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
)

if TYPE_CHECKING:
    from .config import PipelineConfig
from max.pipelines.core import (
    EmbeddingsGenerator,
    EmbeddingsResponse,
    TextContext,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
)

logger = logging.getLogger("max.pipelines")

DEFAULT_MAX_SEQ_LEN = 512


class HFTextGenerationPipeline(TokenGenerator[TextContext]):
    """HuggingFace text token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        torch_device_type: str,
    ) -> None:
        self._pipeline_config = pipeline_config
        self._torch_device = torch.device(torch_device_type)
        self._huggingface_config = AutoConfig.from_pretrained(
            pipeline_config.model_config.model_path,
            trust_remote_code=pipeline_config.model_config.trust_remote_code,
            revision=pipeline_config.model_config.huggingface_model_revision,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            pipeline_config.model_config.model_path,
            trust_remote_code=pipeline_config.model_config.trust_remote_code,
        ).to(self._torch_device)
        self._dtype = self._model.dtype

        if self._model.config.model_type == "gemma2":
            raise RuntimeError(
                "Gemma2 architecture is currently not supported."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            pipeline_config.model_config.model_path
        )

        eos_token_id = self._tokenizer.eos_token_id

        # Expand eos tokens if more are provided in pipeline_config
        if "eos_token_id" in self._huggingface_config:
            eos_tokens = self._huggingface_config.eos_token_id
            if isinstance(eos_tokens, int):
                if eos_tokens != eos_token_id:
                    msg = f"eos_token_id provided in huggingface config ({eos_tokens}), does not match provided eos_token_id ({eos_token_id}), using provided eos_token_id"
                    logger.warning(msg)

                eos_token_id = set([eos_tokens])
            elif isinstance(eos_tokens, list):
                if eos_token_id in eos_tokens:
                    eos_token_id = set(eos_tokens)
                else:
                    eos_token_id = set([eos_token_id])
            else:
                msg = f"eos_token_id in huggingface_config, is neither int or list: {eos_tokens}"
                logger.warning(msg)
                self._eos_token_id = set([eos_token_id])
        else:
            eos_token_id = set([eos_token_id])

        self._eos_token_id = eos_token_id
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        max_batch_size = pipeline_config.max_batch_size
        assert max_batch_size is not None, (
            "max_batch_size must be set before constructing ContinuousHFStaticCache"
        )
        self._cache = ContinuousHFStaticCache(
            config=self._model.config,
            max_batch_size=max_batch_size,
            max_seq_len=DEFAULT_MAX_SEQ_LEN
            if pipeline_config.max_length is None
            else pipeline_config.max_length,
            dtype=self._dtype,
            device=self._torch_device,
        )

    def next_token(
        self, batch: dict[str, TextContext], num_steps: int
    ) -> dict[str, TextGenerationResponse]:
        """Provided a batch, process batch inputs, execute the graph for num_steps in a multi-step scenario,
        then decode the tokens holistically and return the list of decoded tokens.
        """

        context_batch = list(batch.values())
        new_sequences = False

        # Claim cache slots for new sequences
        for ctx in context_batch:
            seq_id = ctx.cache_seq_id
            if seq_id in self._cache.available_slots:
                self._cache.external_claim([seq_id])
                self._cache.tokens[seq_id] = ctx.next_tokens
                new_sequences = True

        cache_seq_ids = [ctx.cache_seq_id for ctx in context_batch]
        self._cache.set_active_slots(cache_seq_ids)

        # Initial inputs preparation
        input_ids, attention_mask, cache_position = (
            self._prepare_initial_token_inputs(
                cache_seq_ids, context_batch, new_sequences
            )
        )

        # Pre-allocate tensor for generated tokens
        generated_tokens = torch.zeros(
            (len(context_batch), num_steps),
            dtype=torch.long,
            device=self._torch_device,
        )

        # Generate tokens
        with torch.no_grad():
            for step in range(num_steps):
                self._cache.set_cache_position(cache_position)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=self._cache,
                    use_cache=True,
                )

                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                generated_tokens[:, step] = next_token

                # if last step, no need to prepare next batch input
                if step == num_steps - 1:
                    break

                # Next inputs preparation
                input_ids, attention_mask, cache_position = (
                    self._prepare_next_token_inputs(
                        cache_seq_ids, next_tokens=next_token.unsqueeze(-1)
                    )
                )

        # Move generated tokens to CPU for processing
        generated_tokens = generated_tokens.cpu()

        # Prepare the response, pruning away completed requests as we go.
        res: dict[str, TextGenerationResponse] = {}
        for batch_idx, (request_id, context) in enumerate(batch.items()):
            status = TextGenerationStatus.ACTIVE
            res[request_id] = TextGenerationResponse([], status)
            for step in range(num_steps):
                next_token_id = generated_tokens[batch_idx, step].item()

                # Update context
                context.update(next_token_id)
                next_token = np.array([next_token_id])
                self._cache.tokens[context.cache_seq_id] = np.append(
                    self._cache.tokens[context.cache_seq_id], next_token
                )

                max_length = (
                    DEFAULT_MAX_SEQ_LEN
                    if context.max_length is None
                    else context.max_length
                )

                if next_token_id in self._eos_token_id:
                    status = TextGenerationStatus.END_OF_SEQUENCE
                    res[request_id].update_status(status)
                elif context.current_length > max_length:
                    status = TextGenerationStatus.MAXIMUM_LENGTH
                    res[request_id].update_status(status)
                elif context.current_length == max_length:
                    res[request_id].append_token(TextResponse(next_token))
                    status = TextGenerationStatus.MAXIMUM_LENGTH
                    res[request_id].update_status(status)
                else:
                    res[request_id].append_token(TextResponse(next_token))

                if status.is_done:
                    break

        return res

    def _prepare_initial_token_inputs(
        self,
        cache_seq_ids: list[int],
        context_batch: list[TextContext],
        new_sequences: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if new_sequences:
            # Process all tokens for new sequences
            tokens = [
                torch.tensor(self._cache.tokens[ctx.cache_seq_id])
                for ctx in context_batch
            ]
            padded = self._tokenizer.pad(
                BatchEncoding({"input_ids": tokens}),
                padding=True,
                return_tensors="pt",
            )

            input_ids = padded["input_ids"].to(self._torch_device)
            attention_mask = padded["attention_mask"].to(self._torch_device)

            # Initialize attention patterns
            for seq_id, mask in zip(cache_seq_ids, attention_mask):
                self._cache.update_attention_pattern(seq_id, mask)
        else:
            # Process only next tokens for the iterative steps
            tokens = [torch.tensor(ctx.next_tokens) for ctx in context_batch]
            padded = self._tokenizer.pad(
                BatchEncoding({"input_ids": tokens}),
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(self._torch_device)

            # Extend attention patterns by 1
            ones = torch.ones(1, dtype=torch.long, device=self._torch_device)
            for seq_id in cache_seq_ids:
                pattern = self._cache.attention_patterns[seq_id]
                self._cache.update_attention_pattern(
                    seq_id, torch.cat([pattern, ones])
                )

        # Get complete attention mask for all sequences
        attention_mask = self._cache.get_attention_mask(cache_seq_ids)

        # Calculate cache position
        seq_length = input_ids.size(1)
        max_length = attention_mask.size(1)
        cache_position = torch.arange(
            max_length - seq_length, max_length, device=self._torch_device
        )

        return input_ids, attention_mask, cache_position

    def _prepare_next_token_inputs(
        self,
        cache_seq_ids: list[int],
        next_tokens: Optional[Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process only next tokens for the iterative steps
        encoded_input = BatchEncoding({"input_ids": next_tokens})
        padded = self._tokenizer.pad(
            encoded_input, padding=True, return_tensors="pt"
        )
        input_ids = padded["input_ids"].to(self._torch_device)

        # Extend attention patterns by 1
        ones = torch.ones(1, dtype=torch.long, device=self._torch_device)
        for seq_id in cache_seq_ids:
            pattern = self._cache.attention_patterns[seq_id]
            self._cache.update_attention_pattern(
                seq_id, torch.cat([pattern, ones])
            )

        attention_mask = self._cache.get_attention_mask(cache_seq_ids)

        # Calculate cache position
        seq_length = input_ids.size(1)
        max_length = attention_mask.size(1)
        cache_position = torch.arange(
            max_length - seq_length, max_length, device=self._torch_device
        )

        return input_ids, attention_mask, cache_position

    def release(self, context: TextContext) -> None:
        if context.cache_seq_id not in self._cache.available_slots:
            self._cache.release(context.cache_seq_id)


class HFEmbeddingsPipeline(EmbeddingsGenerator[TextContext]):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        torch_device_type: str,
    ) -> None:
        self._pipeline_config = pipeline_config
        self._torch_device = torch.device(torch_device_type)
        self._model = AutoModel.from_pretrained(
            pipeline_config.model_config.model_path,
            trust_remote_code=pipeline_config.model_config.trust_remote_code,
        ).to(self._torch_device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            pipeline_config.model_config.model_path
        )

    def prepare_initial_token_inputs(
        self,
        context_batch: list[TextContext],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get tokens and seq_ids.
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens for the batch.
        padded = self._tokenizer.pad(
            BatchEncoding({"input_ids": tokens}),
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=2,
        )
        input_ids = cast(torch.Tensor, padded["input_ids"]).to(
            self._torch_device
        )
        attention_mask = cast(torch.Tensor, padded["attention_mask"]).to(
            self._torch_device
        )
        return input_ids, attention_mask

    def encode(
        self, batch: dict[str, TextContext]
    ) -> dict[str, EmbeddingsResponse]:
        """Encodes a batch of text inputs."""

        # Flatten our batch for consistent indexing.
        context_batch = list(batch.values())
        input_ids, attention_mask = self.prepare_initial_token_inputs(
            context_batch
        )

        outputs = self._model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Pool the embeddings of together, and copy to cpu.
        batch_embeddings = (
            _mean_pooling(outputs, attention_mask).cpu().detach().numpy()
        )

        # Prepare the response.
        res: dict[str, EmbeddingsResponse] = {}
        for batch_index, request_id in enumerate(batch.keys()):
            request_embeddings = batch_embeddings[batch_index]
            res[request_id] = EmbeddingsResponse(request_embeddings)
        return res


# Taken from the sentence piece transformer:
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
