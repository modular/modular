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

from __future__ import annotations

import json
import os
import shutil
import tempfile

# Standard library
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 3rd-party
import hf_repo_lock
import huggingface_hub
import torch
import transformers
from idefics3 import torch_utils as idefics3_torch_utils
from internvl import torch_utils as internvl_torch_utils
from max import driver, pipelines
from max.interfaces import PipelineTask, PipelineTokenizer
from max.pipelines import TextGenerationPipelineInterface
from max.pipelines.architectures.internvl.tokenizer import InternVLProcessor
from peft.peft_model import PeftModel

# Tests
from qwen2_5vl import generate_utils as qwen2_5vl_utils
from qwen3vl import generate_utils as qwen3vl_utils
from test_common import test_data, torch_utils
from test_common.torch_utils import MockTextGenerationRequest


# This is required since the presence of peft changes
# code-path that our `transformer` pipelines take.
# More specifically, when `weights_path` are present in an oracle,
# the `UNUSED` value for the model path is used to try and query
# for LoRA specific config. There isn't a good way to disable this path
# since it is always taken when `peft` is available in the env.
@contextmanager
def disable_peft() -> Generator[None, None, None]:
    original_peft_available = transformers.utils.import_utils._peft_available

    transformers.utils.import_utils._peft_available = False
    try:
        yield
    finally:
        transformers.utils.import_utils._peft_available = (
            original_peft_available
        )


ENCODING_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "gptq": torch.float16,
    "q4_k": torch.float32,
    "q4_0": torch.float32,
    "q6_k": torch.float32,
}


@dataclass
class MaxPipelineAndTokenizer:
    """An instantiated MAX pipeline and pieces necessary to run it."""

    pipeline: (
        TextGenerationPipelineInterface[Any] | pipelines.EmbeddingsPipeline
    )
    tokenizer: PipelineTokenizer[Any, Any, Any]


@dataclass
class TorchModelAndDataProcessor:
    """An instantiated Torch model and pieces necessary to run it."""

    model: transformers.PreTrainedModel
    data_processor: (
        transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | transformers.MllamaProcessor
        | transformers.PixtralProcessor
        | InternVLProcessor
    )


@dataclass
class VLLMPipeline:
    """Configuration to run a vLLM pipeline.

    We do not instantiate the LLM engine here to avoid CUDA context initialization
    in the main process.
    """

    model_path: str
    trust_remote_code: bool = False
    encoding: str | None = None
    tensor_parallel_size: int = 1


class PipelineOracle(ABC):
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    task: PipelineTask = PipelineTask.TEXT_GENERATION
    default_batch_size: int | list[int] | None = None

    @property
    @abstractmethod
    def device_encoding_map(self) -> dict[str, list[str]] | None:
        """A dict where the key are the supported device types, and the
        values are lists of supported encodings.

        Example:
            {
                "cpu": ["float32"],
                "gpu": ["bfloat16"]
            }
        """
        raise NotImplementedError

    @abstractmethod
    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        """Instantiate a MAX pipeline for the given encoding/device."""
        raise NotImplementedError

    @abstractmethod
    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device | str
    ) -> TorchModelAndDataProcessor:
        """Instantiate a Torch pipeline for the given encoding/device."""
        raise NotImplementedError

    def create_vllm_pipeline(
        self, *, encoding: str | None, device_specs: list[driver.DeviceSpec]
    ) -> VLLMPipeline:
        """Instantiate a vLLM pipeline config."""
        path = getattr(self, "model_path", None)
        # We shouldn't hit this; we only have it because using the string
        # `model_path` is standard practice rather than enforced behavior.
        if not path:
            raise ValueError(
                f"Cannot find `model_path` for {self.__class__.__name__}"
            )
        config = getattr(self, "config_params", {})
        # Use tensor parallelism across all GPU devices
        gpu_count = sum(1 for d in device_specs if d.device_type == "gpu")
        return VLLMPipeline(
            model_path=path,
            trust_remote_code=config.get("trust_remote_code", False)
            or getattr(self, "trust_remote_code", False),
            encoding=encoding,
            tensor_parallel_size=max(1, gpu_count),
        )

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for the model.

        By default, creates text-only requests from test data. Multimodal pipelines
        should override this to include images.
        """
        return test_data.DEFAULT_TEXT_ONLY

    @property
    def use_cache(self) -> bool:
        """Whether to use the KV cache, for HF transformers models only."""
        return True

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using the standard torch_utils implementation.

        Can be overridden by subclasses that need custom preprocessing logic.
        """
        return torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class InternVLPipelineOracle(PipelineOracle):
    """Pipeline oracle for InternVL3 architectures."""

    model_path: str
    """ID of the Hugging Face repository."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for InternVL."""
        return (
            test_data.DEFAULT_TEXT_ONLY + test_data.INTERNVL_INSTRUCT_REQUESTS
        )

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)

        # InternVL uses dynamic image sizing, so use a reasonable default
        max_length = 8192

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy="paged",
            model_path=self.model_path,
            huggingface_model_revision=revision,
            max_length=max_length,
            max_num_steps=1,
            trust_remote_code=True,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.8,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, revision=revision, trust_remote_code=True
        )
        processor = InternVLProcessor(tokenizer, config)
        model = transformers.AutoModel.from_pretrained(
            self.model_path,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using InternVL-specific preprocessing logic."""
        return internvl_torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            generate_logprobs=generate_logprobs,
            # Omit `use_cache` since the InternVL code hardcodes it.
        )

    def create_vllm_pipeline(
        self, *, encoding: str | None, device_specs: list[driver.DeviceSpec]
    ) -> VLLMPipeline:
        gpu_count = sum(1 for d in device_specs if d.device_type == "gpu")
        return VLLMPipeline(
            model_path=self.model_path,
            trust_remote_code=True,
            encoding=encoding,
            tensor_parallel_size=max(1, gpu_count),
        )


class Idefics3PipelineOracle(PipelineOracle):
    """Pipeline oracle for Idefics3 architectures."""

    model_path: str
    """ID of the Hugging Face repository."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Idefics3."""

        return (
            test_data.DEFAULT_TEXT_ONLY + test_data.IDEFICS3_INSTRUCT_REQUESTS
        )

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)

        max_length = 8192

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy="paged",
            model_path=self.model_path,
            huggingface_model_revision=revision,
            huggingface_weight_revision=revision,
            max_length=max_length,
            max_num_steps=1,
            trust_remote_code=True,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.8,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, revision=revision, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, revision=revision
        )
        # Use AutoModelForVision2Seq instead of AutoModel for Idefics3
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using Idefics3-specific preprocessing logic."""

        return idefics3_torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class Qwen2_5VLPipelineOracle(PipelineOracle):
    """Pipeline oracle for Qwen2.5VL architectures."""

    model_path: str
    """ID of the Hugging Face repository."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Qwen2.5VL."""
        # Torch model tries to return EOT for the default long text prompt,
        # so add another bullet point to get it to generate more tokens.
        long_prompt = test_data.LONG_TEXT_PROMPT + "\n    * "
        text_only_prompts = [long_prompt] + list(test_data.SHORT_TEXT_PROMPTS)
        text_only_requests = [
            MockTextGenerationRequest.text_only(prompt)
            for prompt in text_only_prompts
        ]
        return qwen2_5vl_utils.INSTRUCT_REQUESTS + text_only_requests

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        max_length = 8192

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy="paged",
            model_path=self.model_path,
            huggingface_model_revision=revision,
            max_length=max_length,
            max_num_steps=1,
            trust_remote_code=True,
            # Chunked prefill is not supported for image prompts.
            # (technically, this script doesn't go through the scheduler so
            # it's not a problem, but it's a good idea to disable it anyway.)
            enable_chunked_prefill=False,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.6,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, revision=revision, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, revision=revision
        )
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            revision=revision,
            config=config,
            device_map=device,
            # Qwen2.5VL 32B uses float32 for the vision model, and bfloat16 for the language model
            # So, we don't set the encoding dtype for the torch model
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using Qwen2.5VL-specific preprocessing logic."""

        return qwen2_5vl_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class Qwen3VLPipelineOracle(PipelineOracle):
    """Pipeline oracle for Qwen3VL architectures."""

    model_path: str
    """ID of the Hugging Face repository."""

    def __init__(
        self,
        model_path: str,
        device_encoding_map: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self._device_encoding_map = device_encoding_map or {"gpu": ["bfloat16"]}

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return self._device_encoding_map

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Qwen3VL."""
        # Torch model tries to return EOT for the default long text prompt,
        # so add another bullet point to get it to generate more tokens.
        long_prompt = test_data.LONG_TEXT_PROMPT + "\n    * "
        text_only_prompts = [long_prompt] + list(test_data.SHORT_TEXT_PROMPTS)
        text_only_requests = [
            MockTextGenerationRequest.text_only(prompt)
            for prompt in text_only_prompts
        ]
        return qwen3vl_utils.INSTRUCT_REQUESTS + text_only_requests

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        max_length = 8192

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy="paged",
            model_path=self.model_path,
            huggingface_model_revision=revision,
            max_length=max_length,
            max_num_steps=1,
            trust_remote_code=True,
            # Chunked prefill is not supported for image prompts.
            # (technically, this script doesn't go through the scheduler so
            # it's not a problem, but it's a good idea to disable it anyway.)
            enable_chunked_prefill=False,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.4,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, revision=revision, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, revision=revision, trust_remote_code=True
        )
        # For FP8 models, use bfloat16 as compute dtype since the FP8 weights
        # are pre-quantized and have their own scale tensors.
        if encoding == "float8_e4m3fn":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = (
                ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None
            )
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
        num_steps: int,
        inputs: list[Any],
        generate_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run text generation using Qwen3VL-specific preprocessing logic."""

        return qwen3vl_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=inputs,
            num_steps=num_steps,
            print_outputs=True,
            use_cache=self.use_cache,
            generate_logprobs=generate_logprobs,
        )


class PixtralPipelineOracle(PipelineOracle):
    def __init__(self) -> None:
        super().__init__()
        self.model_path = "mistral-community/pixtral-12b"

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for Pixtral model."""
        return test_data.PIXTRAL_REQUESTS

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        # TODO (AIPIPE-234): Implement MAX pipeline generation for Pixtral.
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            model_path=self.model_path,
            huggingface_model_revision=revision,
            max_length=8192,
            max_num_steps=1,
        )
        hf_repo_lock.apply_to_config(config)
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)

        assert isinstance(pipeline, pipelines.TextGenerationPipelineInterface)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_path, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            self.model_path, revision=revision
        )
        model = transformers.LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class GenericOracle(PipelineOracle):
    def __init__(
        self,
        *,
        model_path: str,
        device_encoding_map: dict[str, list[str]] | None = None,
        weight_path_map: dict[str, str] | None = None,
        config_params: dict[str, Any] = {},  # noqa: B006
        prompts: list[str] | None = None,
        use_cache: bool = True,
        auto_model_cls: Any = transformers.AutoModelForCausalLM,
        auto_processor_cls: Any = transformers.AutoTokenizer,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
        batch_size: int | list[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self._device_encoding_map = device_encoding_map
        self._weight_path_map = weight_path_map
        self.config_params = config_params
        self._prompts = prompts
        self.auto_model_cls = auto_model_cls
        self.auto_processor_cls = auto_processor_cls
        self.task = task
        self._use_cache = use_cache
        self.default_batch_size = batch_size

    @property
    def device_encoding_map(self) -> dict[str, list[str]] | None:
        return self._device_encoding_map

    def weight_path(self, encoding: str) -> str | None:
        if self._weight_path_map and encoding in self._weight_path_map:
            return self._weight_path_map[encoding]
        return None

    def _parse_weight_path(
        self, weight_path: str
    ) -> tuple[str, str, str | None]:
        """Parse weight path into (repo_id, filename, revision)."""
        path_pieces = weight_path.split("/")
        weight_repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
        weight_filename = "/".join(path_pieces[2:])
        weight_revision = hf_repo_lock.revision_for_hf_repo(weight_repo_id)
        return weight_repo_id, weight_filename, weight_revision

    def create_max_pipeline(
        self,
        *,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        model_revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        weight_path = self.weight_path(encoding) if encoding else None

        # Determine weight revision: use weight repo's revision if different
        weight_revision = model_revision
        if weight_path:
            weight_repo_id, _, weight_revision = self._parse_weight_path(
                weight_path
            )
            if weight_repo_id == self.model_path:
                weight_revision = model_revision

        config = pipelines.PipelineConfig(
            device_specs=device_specs if device_specs else None,
            quantization_encoding=pipelines.SupportedEncoding[encoding]
            if encoding
            else None,
            model_path=self.model_path,
            huggingface_model_revision=model_revision,
            huggingface_weight_revision=weight_revision,
            weight_path=[] if weight_path is None else [weight_path],
            max_num_steps=1,
            **self.config_params,
        )
        hf_repo_lock.apply_to_config(config)
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            config, task=self.task
        )
        assert isinstance(
            pipeline,
            pipelines.TextGenerationPipelineInterface
            | pipelines.EmbeddingsPipeline,
        )
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        trust_remote_code = self.config_params.get("trust_remote_code", False)
        processor = self.auto_processor_cls.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        weight_path = self.weight_path(encoding) if encoding else None
        if weight_path:
            config_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=self.model_path,
                    filename="config.json",
                    revision=hf_repo_lock.revision_for_hf_repo(self.model_path),
                )
            )
            weight_repo_id, weight_filename, weight_revision = (
                self._parse_weight_path(weight_path)
            )
            downloaded_weight_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=weight_repo_id,
                    filename=weight_filename,
                    revision=weight_revision,
                )
            )
            config = transformers.AutoConfig.from_pretrained(config_path)

            with disable_peft():
                model = self.auto_model_cls.from_pretrained(
                    "UNUSED",
                    config=config,
                    gguf_file=str(downloaded_weight_path),
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding]
                    if encoding
                    else None,
                )
        else:
            model = self.auto_model_cls.from_pretrained(
                self.model_path,
                revision=hf_repo_lock.revision_for_hf_repo(self.model_path),
                device_map=device,
                trust_remote_code=trust_remote_code,
                torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding]
                if encoding
                else None,
            )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        return (
            [
                MockTextGenerationRequest.text_only(prompt=prompt)
                for prompt in self._prompts
            ]
            if self._prompts
            else test_data.DEFAULT_TEXT_ONLY
        )

    @property
    def use_cache(self) -> bool:
        return self._use_cache


class LoRAOracle(PipelineOracle):
    """Oracle for models with LoRA adapters."""

    def __init__(
        self,
        *,
        model_path: str,
        lora_repo_id: str,
        device_encoding_map: dict[str, list[str]],
        config_params: dict[str, Any] = {},  # noqa: B006
        prompts: list[str] | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize LoRA oracle.

        Args:
            model_path: Path to the base model
            target_modules: Target modules for LoRA (qkvo, gate, down, up, etc.)
            lora_adapter_path: Path to the LoRA adapter (if None, creates test adapter)
            device_encoding_map: Device to encoding mapping
            config_params: Additional config parameters
            prompts: Custom prompts to use
            use_cache: Whether to use cache
            lora_rank: LoRA rank parameter
        """
        self.model_path = model_path
        self.lora_repo_id = lora_repo_id
        self._device_encoding_map = device_encoding_map
        self.config_params = config_params
        self._prompts = prompts
        self._use_cache = use_cache
        self.lora_rank = -1
        self._adapter_path: str | None = None

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return self._device_encoding_map

    def _get_shared_adapter(self) -> str:
        if self._adapter_path is None:
            revision = hf_repo_lock.revision_for_hf_repo(self.lora_repo_id)
            original_adapter_path = huggingface_hub.snapshot_download(
                repo_id=self.lora_repo_id,
                revision=revision,
            )

            # Copy the adapter to /tmp/ to avoid modifying the original
            tmp_dir = tempfile.mkdtemp(prefix="lora_adapter_", dir="/tmp")
            self._adapter_path = os.path.join(
                tmp_dir, os.path.basename(original_adapter_path)
            )
            shutil.copytree(original_adapter_path, self._adapter_path)

            # Fix adapter config for compatibility with older PEFT versions
            config_path = Path(self._adapter_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                self.lora_rank = config["r"]
                unsupported_fields = [
                    "corda_config",
                    "eva_config",
                    "exclude_modules",
                    "lora_bias",
                    "qalora_group_size",
                    "target_parameters",
                    "trainable_token_indices",
                    "use_qalora",
                ]
                for field in unsupported_fields:
                    if field in config:
                        del config[field]

                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

        assert self._adapter_path is not None
        return self._adapter_path

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        """Create MAX pipeline with LoRA adapter."""

        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        lora_path = self._get_shared_adapter()

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            model_path=self.model_path,
            huggingface_model_revision=revision,
            max_num_steps=1,
            enable_lora=True,
            lora_paths=[lora_path],
            max_num_loras=1,
            max_lora_rank=self.lora_rank,
            cache_strategy="paged",
            enable_prefix_caching=False,  # LoRA requires prefix caching disabled
            trust_remote_code=True,
            **self.config_params,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)

        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        assert pipeline._pipeline_model._lora_manager is not None
        pipeline._pipeline_model._lora_manager.activate_adapter(lora_path)
        return MaxPipelineAndTokenizer(pipeline, tokenizer)

    def create_torch_pipeline(
        self, *, encoding: str | None, device: torch.device
    ) -> TorchModelAndDataProcessor:
        """Create PyTorch pipeline with LoRA adapter using PEFT."""

        # Load base model
        revision = hf_repo_lock.revision_for_hf_repo(self.model_path)
        lora_path = self._get_shared_adapter()

        processor = transformers.AutoTokenizer.from_pretrained(
            self.model_path, revision=revision, trust_remote_code=True
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=revision,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding] if encoding else None,
        )

        model = PeftModel.from_pretrained(model, lora_path, "lora")
        model.set_adapter("lora")

        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        prompts = self._prompts if self._prompts else test_data.DEFAULT_PROMPTS
        return [
            MockTextGenerationRequest(
                prompt=prompt,
                images=[],
                messages=[],
                is_multimodal=False,
                model_name=self._get_shared_adapter(),
            )
            for prompt in prompts
        ]

    @property
    def use_cache(self) -> bool:
        return self._use_cache


PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    "allenai/OLMo-1B-hf": GenericOracle(
        model_path="allenai/OLMo-1B-hf",
        config_params={"max_length": 1024},
        device_encoding_map={"cpu": ["float32"], "gpu": ["float32"]},
    ),
    "microsoft/Phi-3.5-mini-instruct": GenericOracle(
        model_path="microsoft/Phi-3.5-mini-instruct",
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "microsoft/phi-4": GenericOracle(
        model_path="microsoft/phi-4",
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": GenericOracle(
        model_path="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        config_params={
            "max_length": 1024,
            "max_batch_size": 128,  # TODO(E2EOPT-48): Remove batch size override.
            "trust_remote_code": True,
        },
        device_encoding_map={"cpu": ["float32"], "gpu": ["float32"]},
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": GenericOracle(
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        weight_path_map={
            "q4_k": "bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "float32": "bartowski/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-fp32.gguf",
        },
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float32", "bfloat16"],
            "cpu": ["float32", "q4_k"],
        },
    ),
    "meta-llama/Llama-3.1-8B-Instruct": GenericOracle(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        weight_path_map={
            "q4_k": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "float32": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-f32.gguf",
        },
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float32", "bfloat16"],
            "cpu": ["float32", "q4_k"],
        },
    ),
    "meta-llama/Llama-3.1-8B-Instruct-data-parallel": GenericOracle(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        config_params={
            "max_length": 512,
            "data_parallel_degree": 2,
            "enable_prefix_caching": False,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
        # prompts=test_data.DEFAULT_PROMPTS[1:2] + test_data.DEFAULT_PROMPTS[1:3],
        # Run the 4 text prompts with batch sizes 1, 2, 1.
        batch_size=[1, 2, 1],
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-float8-static": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "nvidia/Llama-3.1-8B-Instruct-NVFP4": GenericOracle(
        model_path="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": GenericOracle(
        model_path="nvidia/Llama-3.1-405B-Instruct-NVFP4",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "meta-llama/Llama-3.2-1B": GenericOracle(
        model_path="meta-llama/Llama-3.2-1B",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "meta-llama/Llama-3.3-70B-Instruct": GenericOracle(
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "allenai/OLMo-2-0425-1B": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "allenai/OLMo-2-0425-1B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-0425-1B-RLVR1": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B-RLVR1",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-1124-7B": GenericOracle(
        model_path="allenai/OLMo-2-1124-7B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "allenai/OLMo-2-1124-7B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-1124-7B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-1124-13B": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "allenai/OLMo-2-1124-13B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-1124-13B-Instruct-RLVR1": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct-RLVR1",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-1124-13B-Instruct-RLVR2": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct-RLVR2",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/OLMo-2-0325-32B-Instruct": GenericOracle(
        model_path="allenai/OLMo-2-0325-32B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "tngtech/OLMo-2-Instruct-Math-32B": GenericOracle(
        model_path="tngtech/OLMo-2-Instruct-Math-32B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "allenai/Olmo-3-7B-Instruct": GenericOracle(
        model_path="allenai/Olmo-3-7B-Instruct",
        config_params={
            "max_length": 32768,
            "use_legacy_module": False,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "mistralai/Mistral-Nemo-Instruct-2407": GenericOracle(
        model_path="mistralai/Mistral-Nemo-Instruct-2407",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": GenericOracle(
        model_path="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
        auto_model_cls=transformers.AutoModelForImageTextToText,
    ),
    "OpenGVLab/InternVL3-1B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-1B-Instruct"
    ),
    "OpenGVLab/InternVL3-8B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-8B-Instruct"
    ),
    "OpenGVLab/InternVL3-14B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-14B-Instruct"
    ),
    "OpenGVLab/InternVL3-38B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-38B-Instruct"
    ),
    "OpenGVLab/InternVL3-78B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-78B-Instruct"
    ),
    "OpenGVLab/InternVL3_5-8B-Instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3_5-8B-Instruct"
    ),
    "HuggingFaceM4/Idefics3-8B-Llama3": Idefics3PipelineOracle(
        "HuggingFaceM4/Idefics3-8B-Llama3"
    ),
    "mistral-community/pixtral-12b": PixtralPipelineOracle(),
    "Qwen/Qwen2.5-7B-Instruct": GenericOracle(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "unsloth/gpt-oss-20b-BF16": GenericOracle(
        model_path="unsloth/gpt-oss-20b-BF16",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2_5VLPipelineOracle(
        "Qwen/Qwen2.5-VL-3B-Instruct"
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5VLPipelineOracle(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    ),
    "Qwen/Qwen2.5-VL-32B-Instruct": Qwen2_5VLPipelineOracle(
        "Qwen/Qwen2.5-VL-32B-Instruct"
    ),
    # Qwen2.VL-FP8
    "allenai/olmOCR-2-7B-1025-FP8": Qwen2_5VLPipelineOracle(
        "allenai/olmOCR-2-7B-1025-FP8"
    ),
    "Qwen/Qwen3-VL-30B-A3B-Instruct": Qwen3VLPipelineOracle(
        "Qwen/Qwen3-VL-30B-A3B-Instruct"
    ),
    "Qwen/Qwen3-VL-4B-Instruct": Qwen3VLPipelineOracle(
        "Qwen/Qwen3-VL-4B-Instruct"
    ),
    "Qwen/Qwen3-VL-4B-Instruct-FP8": Qwen3VLPipelineOracle(
        "Qwen/Qwen3-VL-4B-Instruct-FP8",
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "Qwen/Qwen3-8B": GenericOracle(
        model_path="Qwen/Qwen3-8B",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-32B": GenericOracle(
        model_path="Qwen/Qwen3-32B",
        config_params={"max_length": 512, "max_batch_size": 1},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-30B-A3B": GenericOracle(
        model_path="Qwen/Qwen3-30B-A3B",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": GenericOracle(
        model_path="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "HuggingFaceTB/SmolLM2-135M": GenericOracle(
        model_path="HuggingFaceTB/SmolLM2-135M",
        config_params={
            "max_length": 512,
            "cache_strategy": "paged",
        },
        prompts=[p[:502] for p in test_data.DEFAULT_PROMPTS],
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "HuggingFaceTB/SmolLM2-360M-Instruct": LoRAOracle(
        model_path="HuggingFaceTB/SmolLM2-360M-Instruct",
        lora_repo_id="fausap/peft-smollm2-lora-gtx1660",
        config_params={
            "max_length": 2048,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic-BF16-LoRA": LoRAOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        lora_repo_id="FinGPT/fingpt-mt_llama3-8b_lora",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "sentence-transformers/all-mpnet-base-v2": GenericOracle(
        model_path="sentence-transformers/all-mpnet-base-v2",
        # Maximum length accepted by MPNet tokenizer is 512.
        config_params={"max_length": 512, "pool_embeddings": False},
        prompts=[p[:502] for p in test_data.DEFAULT_PROMPTS],
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32"],
        },
    ),
    "Qwen/Qwen3-Embedding-0.6B": GenericOracle(
        model_path="Qwen/Qwen3-Embedding-0.6B",
        config_params={"max_length": 8192, "pool_embeddings": True},
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "Qwen/Qwen3-Embedding-4B": GenericOracle(
        model_path="Qwen/Qwen3-Embedding-4B",
        config_params={"max_length": 8192, "pool_embeddings": True},
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "Qwen/Qwen3-Embedding-8B": GenericOracle(
        model_path="Qwen/Qwen3-Embedding-8B",
        config_params={"max_length": 8192, "pool_embeddings": True},
        auto_model_cls=transformers.AutoModel,
        task=PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    # GPTQ llama with perm_idx
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": GenericOracle(
        model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        auto_model_cls=transformers.AutoModelForCausalLM,
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16", "gptq"],
        },
    ),
    # GPTQ llama without perm_idx
    "kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit": GenericOracle(
        model_path="kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit",
        auto_model_cls=transformers.AutoModelForCausalLM,
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16", "gptq"],
        },
    ),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": GenericOracle(
        model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # TODO(bduke): test chunked attention with >8192 context length cases.
        config_params={"max_length": 8192},
        device_encoding_map={"gpu": ["bfloat16"]},
        # TODO(bduke): remove this once upstream [issue](https://github.com/huggingface/transformers/issues/37380) is fixed.
        use_cache=False,
    ),
    "google/gemma-3-1b-it": GenericOracle(
        model_path="google/gemma-3-1b-it",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "google/gemma-3-27b-it": GenericOracle(
        model_path="google/gemma-3-27b-it",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "google/gemma-3-12b-it": GenericOracle(
        model_path="google/gemma-3-12b-it",
        config_params={"max_length": 8192},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": GenericOracle(
        model_path="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": GenericOracle(
        model_path="deepseek-ai/DeepSeek-V2-Lite-Chat",
        config_params={"max_length": 516, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
        prompts=[prompt[:1500] for prompt in test_data.DEFAULT_PROMPTS],
        # upstream modeling_deepsek.py uses a deprecated transformers function
        use_cache=False,
    ),
    "kathywu95/deepseek-v3-small-random": GenericOracle(
        model_path="kathywu95/deepseek-v3-small-random",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
        },
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "kathywu95/deepseek-v3-small-random-fp8": GenericOracle(
        model_path="kathywu95/deepseek-v3-small-random-fp8",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
            "use_subgraphs": False,
        },
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "deepseek-ai/DeepSeek-R1": GenericOracle(
        model_path="deepseek-ai/DeepSeek-R1",
        config_params={
            "max_length": 516,
            "trust_remote_code": False,
            "max_batch_input_tokens": 512,
            "ep_size": 8,
            "data_parallel_degree": 8,
        },
        device_encoding_map={"gpu": ["float8_e4m3fn"]},
    ),
    "nvidia/DeepSeek-R1-0528-NVFP4-v2": GenericOracle(
        model_path="nvidia/DeepSeek-R1-0528-NVFP4-v2",
        config_params={
            "max_length": 1028,
            "trust_remote_code": False,
            "max_batch_input_tokens": 1024,
            "ep_size": 8,
            "data_parallel_degree": 8,
        },
        device_encoding_map={"gpu": ["float4_e2m1fnx2"]},
    ),
    "HKUSTAudio/Llasa-8B": GenericOracle(
        model_path="HKUSTAudio/Llasa-8B",
        config_params={
            "max_length": 2048,
            "trust_remote_code": False,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
        # TTS-specific prompts formatted according to the HF model card.
        prompts=[
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>Hello, this is a test of the Llasa text-to-speech system.<|TEXT_UNDERSTANDING_END|>",
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>The quick brown fox jumps over the lazy dog.<|TEXT_UNDERSTANDING_END|>",
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>Good morning! How are you today?<|TEXT_UNDERSTANDING_END|>",
            "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>In a hole in the ground there lived a hobbit.<|TEXT_UNDERSTANDING_END|>",
        ],
        use_cache=True,
    ),
}
