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

"""Generates text from text in-process with a model and its tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import overload

from .auto_model import AutoModel, AutoTokenizer
from .generate import generate

__all__ = ["AutoPipeline"]


class AutoPipeline:
    """Generates text from text with an :class:`~max.pipelines.AutoModel`.

    It encodes each prompt with an :class:`~max.pipelines.AutoTokenizer`,
    generates, and decodes. Each output starts with its prompt exactly as
    passed. The prompt goes to the model as is, without a chat template.

    .. Skipped: the example downloads a model.
    .. skip: next

    .. code-block:: python

        pipe = AutoPipeline("HuggingFaceTB/SmolLM2-135M", max_batch_size=2)
        texts = pipe(
            ["The capital of France is", "2 + 2 ="], max_new_tokens=16
        )
        print(texts)

    Args:
        model_path: A Hugging Face repo ID or a local model directory.
        max_batch_size: The most prompts per data-parallel replica, as for
            :class:`~max.pipelines.AutoModel`.
        devices: The devices to load on, as for
            :class:`~max.pipelines.AutoModel`.
        device_graph_capture: Whether decode steps replay captured graphs, as
            for :class:`~max.pipelines.AutoModel`.
        **config_overrides: Flat :class:`~max.pipelines.lib.PipelineArgs`
            field names. The tokenizer loads with the revision and
            ``trust_remote_code`` of the resolved config.

    Raises:
        TypeError: As described for :class:`~max.pipelines.AutoModel`.
        ValueError: As described for :class:`~max.pipelines.AutoModel`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_batch_size: int = 1,
        devices: str | None = None,
        device_graph_capture: bool = False,
        **config_overrides: object,
    ) -> None:
        model = AutoModel(
            model_path,
            max_batch_size=max_batch_size,
            devices=devices,
            device_graph_capture=device_graph_capture,
            **config_overrides,
        )
        try:
            # Serve builds its tokenizer from these resolved fields.
            model_config = model.pipeline_config.model
            self._tokenizer = AutoTokenizer(
                model_config.model_path,
                revision=model_config.huggingface_model_revision,
                trust_remote_code=model_config.trust_remote_code,
            )
        except BaseException:
            model.close()
            raise
        self._model = model

    @property
    def model(self) -> AutoModel:
        """The model that generates the tokens."""
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """The tokenizer that encodes prompts and decodes outputs."""
        return self._tokenizer

    @overload
    def __call__(
        self,
        text: str,
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> str: ...

    @overload
    def __call__(
        self,
        text: list[str],
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> list[str]: ...

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> str | list[str]:
        """Returns each prompt followed by the text the model generates.

        ``text`` is one prompt or a list of prompts, which run as one batch.
        The sampling arguments are those of
        :meth:`~max.pipelines.AutoModel.generate`.

        Returns:
            A string for a string prompt, or a list with one string per
            prompt.

        Raises:
            TypeError: If ``text`` is not a string or a list of strings.
            ValueError: If ``text`` is an empty list, or as described for
                :meth:`~max.pipelines.AutoModel.generate`.
            RuntimeError: If the model has been closed, or as described for
                :meth:`~max.pipelines.AutoModel.generate`.
        """
        if isinstance(text, str):
            prompts = [text]
        elif isinstance(text, Sequence) and all(
            isinstance(prompt, str) for prompt in text
        ):
            prompts = list(text)
        else:
            raise TypeError("text must be a string or a list of strings.")
        if not prompts:
            raise ValueError("text must hold at least one prompt.")
        prompt_ids: list[list[int]] = self._tokenizer(prompts)["input_ids"]
        sequences = generate(
            self._model.pipeline,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            ignore_eos=ignore_eos,
            stop_token_ids=stop_token_ids,
            seed=seed,
            caller="AutoPipeline",
        ).sequences
        texts = [
            self._with_prompt(prompt, ids, sequence)
            for prompt, ids, sequence in zip(
                prompts, prompt_ids, sequences, strict=True
            )
        ]
        return texts[0] if isinstance(text, str) else texts

    def _with_prompt(
        self, prompt: str, prompt_ids: Sequence[int], sequence: Sequence[int]
    ) -> str:
        # As Hugging Face's text generation pipeline does with
        # return_full_text, this cuts the decoded prompt off the decoded
        # sequence, so the caller's prompt comes back unchanged and the
        # spacing where the new text starts stays right.
        full = self._tokenizer.decode(sequence, skip_special_tokens=True)
        decoded_prompt = self._tokenizer.decode(
            prompt_ids, skip_special_tokens=True
        )
        return prompt + full[len(decoded_prompt) :]

    def close(self) -> None:
        """Closes the model, as :meth:`~max.pipelines.AutoModel.close` does."""
        self._model.close()

    def __enter__(self) -> AutoPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
