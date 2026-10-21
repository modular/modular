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

"""Generates text from text or conversations in-process with a model."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import overload

from .auto_model import AutoModel, AutoTokenizer
from .auto_processor import _model_inputs
from .generate import generate

__all__ = ["AutoPipeline"]

_TEXT_TYPE_ERROR = "text must be a string, a conversation, or a list of either."


class AutoPipeline:
    """Generates text with an :class:`~max.pipelines.AutoModel`.

    A string prompt goes to the model as is, without a chat template, and its
    output starts with the prompt exactly as passed. A conversation, in the
    message format :class:`~max.pipelines.AutoProcessor` takes, goes through
    the chat template with its images, and its output is only the new text.

    .. Skipped: the example downloads a model.
    .. skip: next

    .. code-block:: python

        pipe = AutoPipeline("HuggingFaceTB/SmolLM2-135M", max_batch_size=2)
        texts = pipe(
            ["The capital of France is", "2 + 2 ="], max_new_tokens=16
        )
        print(texts)

    A conversation can hold images.

    .. skip: next

    .. code-block:: python

        from PIL import Image

        pipe = AutoPipeline("google/gemma-3-4b-it")
        text = pipe(
            [{"role": "user", "content": [
                {"type": "image", "image": Image.open("cat.png")},
                {"type": "text", "text": "Describe this image."},
            ]}],
            max_new_tokens=32,
        )
        print(text)

    Args:
        model_path: A Hugging Face repo ID or a local model directory.
        max_batch_size: The most prompts per data-parallel replica, as for
            :class:`~max.pipelines.AutoModel`.
        devices: The devices to load on, as for
            :class:`~max.pipelines.AutoModel`.
        device_graph_capture: Whether decode steps replay captured graphs, as
            for :class:`~max.pipelines.AutoModel`.
        prefer_module_v3: Whether to load the architecture's ModuleV3
            version, as for :class:`~max.pipelines.AutoModel`.
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
        prefer_module_v3: bool = False,
        **config_overrides: object,
    ) -> None:
        model = AutoModel(
            model_path,
            max_batch_size=max_batch_size,
            devices=devices,
            device_graph_capture=device_graph_capture,
            prefer_module_v3=prefer_module_v3,
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
        text: str | Sequence[Mapping[str, object]],
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
        text: list[str]
        | tuple[str, ...]
        | Sequence[Sequence[Mapping[str, object]]],
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
        text: str
        | Sequence[str]
        | Sequence[Mapping[str, object]]
        | Sequence[Sequence[Mapping[str, object]]],
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
        """Returns the text the model generates for each prompt or conversation.

        ``text`` is a prompt, a conversation, or a list of either, which run
        as one batch. A prompt's output starts with the prompt. A
        conversation's output is only the new text. The sampling arguments
        are those of :meth:`~max.pipelines.AutoModel.generate`.

        Returns:
            A string for one prompt or conversation, or a list with one
            string per item.

        Raises:
            TypeError: If ``text`` is not a prompt, a conversation, or a list
                of either, or as described for
                :class:`~max.pipelines.AutoProcessor`.
            ValueError: If ``text`` is empty, or as described for
                :class:`~max.pipelines.AutoProcessor` and
                :meth:`~max.pipelines.AutoModel.generate`.
            RuntimeError: If the model has been closed, or as described for
                :meth:`~max.pipelines.AutoModel.generate`.
            OSError: If an image path cannot be read.
        """
        if isinstance(text, Mapping):
            raise TypeError(_TEXT_TYPE_ERROR)
        if isinstance(text, (str, bytes)) or not isinstance(text, Sequence):
            items: list[object] = [text]
        else:
            items = list(text)
        if not items:
            raise ValueError("text must hold at least one prompt.")
        single = isinstance(text, str) or isinstance(items[0], Mapping)
        prompts = [item for item in items if isinstance(item, str)]
        all_prompts = len(prompts) == len(items)
        if not all_prompts and not (
            single
            or all(
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                for item in items
            )
        ):
            raise TypeError(_TEXT_TYPE_ERROR)
        generate_rows = functools.partial(
            generate,
            self._model.pipeline,
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
        )
        if all_prompts:
            prompt_ids: list[list[int]] = self._tokenizer(prompts)["input_ids"]
            sequences = generate_rows(prompt_ids).sequences
            texts = [
                self._with_prompt(prompt, ids, sequence)
                for prompt, ids, sequence in zip(
                    prompts, prompt_ids, sequences, strict=True
                )
            ]
        else:
            result = generate_rows(**_model_inputs(text))
            texts = self._tokenizer.batch_decode(
                [
                    sequence[length:]
                    for sequence, length in zip(
                        result.sequences, result.prompt_lengths, strict=True
                    )
                ],
                skip_special_tokens=True,
            )
        return texts[0] if single else texts

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
