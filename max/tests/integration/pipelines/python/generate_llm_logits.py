# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import functools
import os
import sys
import traceback

# Standard library
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

# 3rd-party
import click
import hf_repo_lock
import huggingface_hub

# Tests
import requests
import torch
import transformers
from idefics3 import torch_utils as idefics3_torch_utils
from internvl import torch_utils as internvl_torch_utils
from max import driver, pipelines
from max.entrypoints.cli import DevicesOptionType
from max.interfaces import PipelineTask, PipelineTokenizer
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.architectures.internvl.tokenizer import InternVLProcessor
from test_common import (
    evaluate,
    evaluate_embeddings,
    numpy_encoder,
    test_data,
    torch_utils,
)
from test_common.evaluate import ModelOutput
from test_common.github_utils import github_log_group
from test_common.torch_utils import MockTextGenerationRequest
from typing_extensions import ParamSpec

# This is far from a universal standard, but this is the closest to a standard
# that I could find: BSD-derived programs sometimes use exit codes from
# "sysexits.h", which defines this exit code as "temp failure; user is invited
# to retry".  generate_llm_logits will emit this if it detects a failure is
# likely caused by a network flake and could be resolved by a retry.
EX_TEMPFAIL = 75

ENCODING_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "gptq": torch.float16,
    "q4_k": torch.float32,
    "q4_0": torch.float32,
    "q6_k": torch.float32,
}


@contextmanager
def maybe_log_hf_downloads(enable_logging: bool):
    """Context manager that conditionally logs HuggingFace file downloads."""
    if not enable_logging:
        yield
        return

    original_hf_hub_download = huggingface_hub.hf_hub_download

    def logged_hf_hub_download(*args, **kwargs):
        repo_id = kwargs.get("repo_id") or (
            args[0] if len(args) > 0 else "unknown"
        )
        filename = kwargs.get("filename") or (
            args[1] if len(args) > 1 else "unknown"
        )
        print(f"Accessing {filename} from {repo_id}")
        result = original_hf_hub_download(*args, **kwargs)
        print(f"-> Located at: {result}\n")
        return result

    huggingface_hub.hf_hub_download = logged_hf_hub_download
    try:
        yield
    finally:
        huggingface_hub.hf_hub_download = original_hf_hub_download


@dataclass
class MaxPipelineAndTokenizer:
    """An instantiated MAX pipeline and pieces necessary to run it."""

    model: pipelines.PipelineModel
    generator: Union[
        pipelines.TextGenerationPipeline,
        pipelines.EmbeddingsPipeline,
    ]  # TODO(kcaverly): Move to only TextGenerationPipeline
    tokenizer: PipelineTokenizer


@dataclass
class TorchModelAndDataProcessor:
    """An instantiated Torch model and pieces necessary to run it."""

    model: transformers.PreTrainedModel
    data_processor: Union[
        transformers.PreTrainedTokenizer,
        transformers.PreTrainedTokenizerFast,
        transformers.MllamaProcessor,
        transformers.PixtralProcessor,
        InternVLProcessor,
    ]


class PipelineOracle(ABC):
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    task: PipelineTask = PipelineTask.TEXT_GENERATION

    @property
    @abstractmethod
    def device_encoding_map(self) -> dict[str, list[str]]:
        """A dict where the key are the supported device types, and the
        values are lists of supported encodings.

        Example:
            {
                "cpu": ["float32"],
                "gpu": ["bfloat16"]
            }
        """
        raise NotImplementedError

    def is_supported(
        self, *, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        """Check that a particular encoding/device tuple is supported.

        Returns True if supported, False if not.
        """
        device_type = device_spec.device_type
        if device_type not in self.device_encoding_map:
            return False
        return encoding in self.device_encoding_map[device_type]

    @abstractmethod
    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        """Instantiate a MAX pipeline for the given encoding/device."""
        raise NotImplementedError

    @abstractmethod
    def create_torch_pipeline(
        self, *, encoding: str, device: torch.device | str
    ) -> TorchModelAndDataProcessor:
        """Instantiate a Torch pipeline for the given encoding/device."""
        raise NotImplementedError

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
    ) -> list[dict]:
        """Run text generation using the standard torch_utils implementation.

        Can be overridden by subclasses that need custom preprocessing logic.
        """
        return torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=self.inputs,
            print_outputs=True,
            use_cache=self.use_cache,
        )


class MultiModalPipelineOracle(PipelineOracle):
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    @property
    def inputs(self) -> list[MockTextGenerationRequest]:
        """Input requests for multimodal model."""
        return test_data.DEFAULT_MULTIMODAL


class InternVLPipelineOracle(MultiModalPipelineOracle):
    """Pipeline oracle for InternVL3 architectures."""

    hf_repo_id: str
    """ID of the Hugging Face repository."""

    def __init__(self, hf_repo_id: str) -> None:
        super().__init__()
        self.hf_repo_id = hf_repo_id

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
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)

        revision = hf_repo_lock.revision_for_hf_repo(self.hf_repo_id)

        # Compute the max sequence length for InternVL
        hf_config = transformers.AutoConfig.from_pretrained(
            self.hf_repo_id, revision=revision, trust_remote_code=True
        )
        # InternVL uses dynamic image sizing, so use a reasonable default
        max_length = 8192

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy=KVCacheStrategy.PAGED,
            model_path=self.hf_repo_id,
            huggingface_model_revision=revision,
            max_length=max_length,
            trust_remote_code=True,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.8,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hf_repo_id,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        config = transformers.AutoConfig.from_pretrained(
            self.hf_repo_id, revision=revision, trust_remote_code=True
        )
        processor = InternVLProcessor(tokenizer, config)
        model = transformers.AutoModel.from_pretrained(
            self.hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
    ) -> list[dict]:
        """Run text generation using InternVL-specific preprocessing logic."""
        return internvl_torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=self.inputs,
            print_outputs=True,
            # Omit `use_cache` since the InternVL code hardcodes it.
        )


class Idefics3PipelineOracle(MultiModalPipelineOracle):
    """Pipeline oracle for Idefics3 architectures."""

    hf_repo_id: str
    """ID of the Hugging Face repository."""

    def __init__(self, hf_repo_id: str) -> None:
        super().__init__()
        self.hf_repo_id = hf_repo_id

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
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)

        revision = hf_repo_lock.revision_for_hf_repo(self.hf_repo_id)

        # Compute the max sequence length for Idefics3
        hf_config = transformers.AutoConfig.from_pretrained(
            self.hf_repo_id, revision=revision, trust_remote_code=True
        )

        max_length = 8192

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy=KVCacheStrategy.PAGED,
            model_path=self.hf_repo_id,
            huggingface_model_revision=revision,
            max_length=max_length,
            trust_remote_code=True,
            # TODO(GEX-2365): Handle this in model memory estimation.
            device_memory_utilization=0.8,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        revision = hf_repo_lock.revision_for_hf_repo(self.hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hf_repo_id,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )
        config = transformers.AutoConfig.from_pretrained(
            self.hf_repo_id, revision=revision, trust_remote_code=True
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self.hf_repo_id, revision=revision
        )
        # Use AutoModelForVision2Seq instead of AutoModel for Idefics3
        model = transformers.AutoModelForVision2Seq.from_pretrained(
            self.hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    def run_torch_text_generation(
        self,
        *,
        torch_pipeline_and_tokenizer: TorchModelAndDataProcessor,
        device: torch.device,
    ) -> list[dict]:
        """Run text generation using Idefics3-specific preprocessing logic."""

        return idefics3_torch_utils.run_text_generation(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=device,
            textgen_requests=self.inputs,
            print_outputs=True,
            use_cache=self.use_cache,
        )


class LlamaVisionPipelineOracle(MultiModalPipelineOracle):
    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "gpu": ["bfloat16"],
        }

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)

        hf_repo_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)

        # Compute the max sequence length, which determines up-front memory
        # allocated for the KV cache.
        hf_config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision, trust_remote_code=True
        )
        vision_cfg = hf_config.vision_config
        img_size = vision_cfg.image_size
        patch_size = vision_cfg.patch_size
        max_num_tiles = vision_cfg.max_num_tiles
        num_vision_embeddings = (
            (img_size // patch_size) ** 2 + 1
        ) * max_num_tiles

        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            model_path=hf_repo_id,
            huggingface_model_revision=revision,
            max_length=num_vision_embeddings,
            trust_remote_code=True,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        hf_repo_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        processor = transformers.AutoProcessor.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision
        )
        model = transformers.MllamaForConditionalGeneration.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class PixtralPipelineOracle(MultiModalPipelineOracle):
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
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        hf_repo_id = "mistral-community/pixtral-12b"
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            model_path=hf_repo_id,
            max_length=8192,
        )
        hf_repo_lock.apply_to_config(config)
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)

        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        hf_repo_id = "mistral-community/pixtral-12b"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        processor = transformers.AutoProcessor.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision
        )
        model = transformers.LlavaForConditionalGeneration.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class GenericOracle(PipelineOracle):
    def __init__(
        self,
        *,
        model_path: str,
        device_encoding_map: dict[str, list[str]],
        weight_path_map: dict[str, str] | None = None,
        config_params: dict[str, Any] = {},  # noqa: B006
        prompts: list[str] | None = None,
        use_cache: bool = True,
        auto_model_cls: Any = transformers.AutoModelForCausalLM,
        auto_processor_cls: Any = transformers.AutoTokenizer,
        task: PipelineTask = PipelineTask.TEXT_GENERATION,
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

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return self._device_encoding_map

    def weight_path(self, encoding: str) -> str | None:
        if self._weight_path_map and encoding in self._weight_path_map:
            return self._weight_path_map[encoding]
        return None

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        weight_path = self.weight_path(encoding)
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            model_path=self.model_path,
            weight_path=[] if weight_path is None else [weight_path],
            **self.config_params,
        )
        hf_repo_lock.apply_to_config(config)
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            config, task=self.task
        )
        assert isinstance(
            pipeline,
            (pipelines.TextGenerationPipeline, pipelines.EmbeddingsPipeline),
        )
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        trust_remote_code = self.config_params.get("trust_remote_code", False)
        processor = self.auto_processor_cls.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        weight_path = self.weight_path(encoding)
        if weight_path:
            config_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=self.model_path,
                    filename="config.json",
                    revision=hf_repo_lock.revision_for_hf_repo(self.model_path),
                )
            )
            path_pieces = weight_path.split("/")
            weight_repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
            weight_filename = "/".join(path_pieces[2:])
            downloaded_weight_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=weight_repo_id,
                    filename=weight_filename,
                    revision=hf_repo_lock.revision_for_hf_repo(weight_repo_id),
                )
            )
            config = transformers.AutoConfig.from_pretrained(config_path)
            model = self.auto_model_cls.from_pretrained(
                "UNUSED",
                config=config,
                gguf_file=str(downloaded_weight_path),
                device_map=device,
                trust_remote_code=trust_remote_code,
                torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
            )
        else:
            model = self.auto_model_cls.from_pretrained(
                self.model_path,
                revision=hf_repo_lock.revision_for_hf_repo(self.model_path),
                device_map=device,
                trust_remote_code=trust_remote_code,
                torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
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


PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    "olmo": GenericOracle(
        model_path="allenai/OLMo-1B-hf",
        config_params={"max_length": 1024},
        device_encoding_map={"cpu": ["float32"], "gpu": ["float32"]},
    ),
    "phi-3.5-mini": GenericOracle(
        model_path="microsoft/Phi-3.5-mini-instruct",
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "phi-4": GenericOracle(
        model_path="microsoft/phi-4",
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "exaone": GenericOracle(
        model_path="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        config_params={
            "max_length": 1024,
            "max_batch_size": 128,  # TODO(E2EOPT-48): Remove batch size override.
            "trust_remote_code": True,
        },
        device_encoding_map={"cpu": ["float32"], "gpu": ["float32"]},
    ),
    "llama3-8b": GenericOracle(
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
    "llama3.1-8b": GenericOracle(
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
    "llama3.1-8b-float8-static": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "llama3.1-8b-float8-dynamic": GenericOracle(
        model_path="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["float8_e4m3fn"],
        },
    ),
    "llama3.2-1b": GenericOracle(
        model_path="meta-llama/Llama-3.2-1B",
        config_params={"max_length": 512},
        device_encoding_map={
            "gpu": ["bfloat16"],
        },
    ),
    "llama3.3-70b": GenericOracle(
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "olmo2-1b": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "olmo2-1b-instruct": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-1b-rvlr1": GenericOracle(
        model_path="allenai/OLMo-2-0425-1B-RLVR1",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-7b": GenericOracle(
        model_path="allenai/OLMo-2-1124-7B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "olmo2-7b-instruct": GenericOracle(
        model_path="allenai/OLMo-2-1124-7B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-13b": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["float32"],
            "cpu": ["float32"],
        },
    ),
    "olmo2-13b-instruct": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-13b-instruct-rvlr1": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct-RLVR1",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-13b-instruct-rvlr2": GenericOracle(
        model_path="allenai/OLMo-2-1124-13B-Instruct-RLVR2",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-32b-instruct": GenericOracle(
        model_path="allenai/OLMo-2-0325-32B-Instruct",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "olmo2-32b-math": GenericOracle(
        model_path="tngtech/OLMo-2-Instruct-Math-32B",
        config_params={
            "max_length": 4096,
        },
        device_encoding_map={
            "gpu": ["bfloat16"],
            "cpu": ["bfloat16"],
        },
    ),
    "mistral": GenericOracle(
        model_path="mistralai/Mistral-Nemo-Instruct-2407",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "mistral3": GenericOracle(
        model_path="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
        auto_model_cls=transformers.AutoModelForImageTextToText,
    ),
    "internvl3-1b-instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-1B-Instruct"
    ),
    "internvl3-8b-instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-8B-Instruct"
    ),
    "internvl3-14b-instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-14B-Instruct"
    ),
    "internvl3-38b-instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-38B-Instruct"
    ),
    "internvl3-78b-instruct": InternVLPipelineOracle(
        "OpenGVLab/InternVL3-78B-Instruct"
    ),
    "idefics3-8b-llama3": Idefics3PipelineOracle(
        "HuggingFaceM4/Idefics3-8B-Llama3"
    ),
    "llama3-vision": LlamaVisionPipelineOracle(),
    "pixtral": PixtralPipelineOracle(),
    "qwen": GenericOracle(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "qwen3": GenericOracle(
        model_path="Qwen/Qwen3-8B",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "qwen3-32b": GenericOracle(
        model_path="Qwen/Qwen3-32B",
        config_params={"max_length": 512, "max_batch_size": 1},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "qwen3-30b-a3b": GenericOracle(
        model_path="Qwen/Qwen3-30B-A3B",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "smollm": GenericOracle(
        model_path="HuggingFaceTB/SmolLM2-135M",
        config_params={
            "max_length": 512,
            "cache_strategy": KVCacheStrategy.CONTINUOUS,
        },
        prompts=[p[:502] for p in test_data.DEFAULT_PROMPTS],
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "mpnet": GenericOracle(
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
    # GPTQ llama with perm_idx
    "llama-gptq": GenericOracle(
        model_path="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        auto_model_cls=transformers.AutoModelForCausalLM,
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16", "gptq"],
        },
    ),
    # GPTQ llama without perm_idx
    "llama-gptq-no-perm-idx": GenericOracle(
        model_path="kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit",
        auto_model_cls=transformers.AutoModelForCausalLM,
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16", "gptq"],
        },
    ),
    "llama4-scout": GenericOracle(
        model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        # TODO(bduke): test chunked attention with >8192 context length cases.
        config_params={"max_length": 8192},
        device_encoding_map={"gpu": ["bfloat16"]},
        # TODO(bduke): remove this once upstream [issue](https://github.com/huggingface/transformers/issues/37380) is fixed.
        use_cache=False,
    ),
    "gemma3-1b": GenericOracle(
        model_path="google/gemma-3-1b-it",
        config_params={"max_length": 8192, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "gemma3-multimodal": GenericOracle(
        model_path="google/gemma-3-12b-it",
        config_params={"max_length": 8192},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "deepseek-v2-lite": GenericOracle(
        model_path="deepseek-ai/DeepSeek-V2-Lite-Chat",
        config_params={"max_length": 516, "trust_remote_code": True},
        device_encoding_map={"gpu": ["bfloat16"]},
        prompts=[prompt[:1500] for prompt in test_data.DEFAULT_PROMPTS],
        # upstream modeling_deepsek.py uses a deprecated transformers function
        use_cache=False,
    ),
}


class Flake(Exception):
    """A failure has occurred that appears to be of a temporary nature.

    It is likely that retrying the operation would succeed.
    """


_ParamsT = ParamSpec("_ParamsT")
_ReturnT = TypeVar("_ReturnT")


def _detect_hf_flakes(
    inner: Callable[_ParamsT, _ReturnT],
) -> Callable[_ParamsT, _ReturnT]:
    """Decorator to exit with a distinct status on Hugging Face flake."""

    def is_client_error(exc: requests.RequestException) -> bool:
        if not isinstance(exc, requests.HTTPError):
            return False
        if exc.response is None:
            return False
        # 4xx status codes indicate client error.
        return 400 <= exc.response.status_code < 500

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        except requests.RequestException as exc:
            if (
                exc.request is not None
                and exc.request.url is not None
                and "huggingface.co" in exc.request.url
                and not is_client_error(exc)
            ):
                # This is probably a Hugging Face flake.
                print(
                    "Seems like a Hugging Face flake has occurred:",
                    file=sys.stderr,
                )
                traceback.print_exc()
                print(
                    "-- End of Hugging Face flake traceback --", file=sys.stderr
                )
                raise Flake("Hugging Face API flake detected") from exc
            else:
                raise

    return wrapper


@click.command()
@click.option(
    "--device",
    "device_type",
    type=DevicesOptionType(),
    required=True,
    help="Type of device to run pipeline with",
)
@click.option(
    "--framework",
    "framework_name",
    type=click.Choice(["max", "torch"]),
    required=True,
    help="Framework to run pipeline with",
)
@click.option(
    "--pipeline",
    "pipeline_name",
    type=click.Choice(sorted(list(PIPELINE_ORACLES.keys()))),
    required=True,
    help="Pipeline to run",
)
@click.option(
    "--encoding",
    "encoding_name",
    required=True,
    help="Quantization encoding to run pipeline with",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output resulting goldens JSON to",
)
@click.option(
    "--print/--no-print",
    "print_output",
    type=bool,
    default=False,
    help="Dump goldens in non-JSON format to stdout",
)
@click.option(
    "--max-batch-size",
    "max_batch_size",
    type=int,
    default=1,
    help="The maximum batch size to use when evaluating the model.",
)
@click.option(
    "--log-hf-downloads",
    "log_hf_downloads",
    is_flag=True,
    default=False,
    help="Log HuggingFace file downloads for MAX and Torch models.",
)
def main(
    device_type: str | list[int],
    framework_name: str,
    pipeline_name: str,
    encoding_name: str,
    output_path: Path,
    print_output: bool,
    max_batch_size: int,
    log_hf_downloads: bool,
) -> None:
    """Click command entry point that delegates to the implementation function.

    This wrapper exists because Click command functions aren't easily picklable,
    which causes issues when called from multiprocessing.
    """

    try:
        generate_llm_logits(
            device_specs=DevicesOptionType.device_specs(device_type),
            framework_name=framework_name,
            pipeline_name=pipeline_name,
            encoding_name=encoding_name,
            output_path=output_path,
            print_output=print_output,
            max_batch_size=max_batch_size,
            log_hf_downloads=log_hf_downloads,
        )
    except Flake:
        sys.exit(EX_TEMPFAIL)


@_detect_hf_flakes
def generate_llm_logits(
    device_specs: list[driver.DeviceSpec],
    framework_name: str,
    pipeline_name: str,
    encoding_name: str,
    output_path: Path,
    print_output: bool,
    max_batch_size: int = 1,
    reference: list[ModelOutput] | None = None,
    log_hf_downloads: bool = False,
) -> None:
    """Output logits to a file for a model based on a fixed set of prompts.

    The resulting logit golden files for two different frameworks can be used
    with //SDK/integration-test/pipelines/python/llama3/verify to check their
    similarity.

    """

    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)

    pipeline_oracle = PIPELINE_ORACLES[pipeline_name]

    for device_spec in device_specs:
        if not pipeline_oracle.is_supported(
            encoding=encoding_name,
            device_spec=device_spec,
        ):
            msg = (
                f"Unsupported combination of encoding '{encoding_name}' and "
                f"device '{device_spec.device_type}'. For pipeline "
                f"'{pipeline_name}', supported combinations are: "
                f"{pipeline_oracle.device_encoding_map}"
            )
            raise ValueError(msg)

    title = f"{pipeline_name} - {framework_name.upper()} - {encoding_name}"
    with github_log_group(title):
        if framework_name == "max":
            with maybe_log_hf_downloads(log_hf_downloads):
                max_pipeline_and_tokenizer = (
                    pipeline_oracle.create_max_pipeline(
                        encoding=encoding_name,
                        device_specs=device_specs,
                    )
                )
            print(f"Running {pipeline_name} model on MAX")
            if pipeline_oracle.task == PipelineTask.TEXT_GENERATION:
                results = evaluate.run_model(
                    max_pipeline_and_tokenizer.model,
                    max_pipeline_and_tokenizer.tokenizer,
                    requests=pipeline_oracle.inputs,
                    print_outputs=True,
                    batch_size=max_batch_size,
                    reference=reference,
                )
            elif pipeline_oracle.task == PipelineTask.EMBEDDINGS_GENERATION:
                assert isinstance(
                    max_pipeline_and_tokenizer.generator,
                    pipelines.EmbeddingsPipeline,
                )
                results = evaluate_embeddings.encode(
                    max_pipeline_and_tokenizer.generator,
                    max_pipeline_and_tokenizer.tokenizer,
                    prompts=(inp.prompt for inp in pipeline_oracle.inputs),
                    batch_size=max_batch_size,
                )
            else:
                raise ValueError(
                    f"Evaluating task {pipeline_oracle.task} is not supported."
                )
        elif framework_name == "torch":
            print(f"Running {pipeline_name} model on Torch")
            torch_device: torch.device
            if device_specs[0].device_type == "cpu":
                torch_device = torch.device("cpu")
            elif device_specs[0].device_type == "gpu":
                torch_device = torch.device("cuda:0")

            # For multi-gpu, use auto to handle mapping automatically.
            if len(device_specs) > 1:
                device = "auto"
            else:
                device = torch_device

            with maybe_log_hf_downloads(log_hf_downloads):
                torch_pipeline_and_tokenizer = (
                    pipeline_oracle.create_torch_pipeline(
                        encoding=encoding_name,
                        device=device,
                    )
                )
            if pipeline_oracle.task == PipelineTask.TEXT_GENERATION:
                results = pipeline_oracle.run_torch_text_generation(
                    torch_pipeline_and_tokenizer=torch_pipeline_and_tokenizer,
                    device=torch_device,
                )
            elif pipeline_oracle.task == PipelineTask.EMBEDDINGS_GENERATION:
                results = torch_utils.run_embeddings_generation(
                    model=torch_pipeline_and_tokenizer.model,
                    data_processor=torch_pipeline_and_tokenizer.data_processor,
                    device=torch_device,
                    prompts=(inp.prompt for inp in pipeline_oracle.inputs),
                )
            else:
                raise ValueError(
                    f"Evaluating task {pipeline_oracle.task} is not supported."
                )
        else:
            raise NotImplementedError(
                f"Framework {framework_name!r} not implemented"
            )

    if print_output:
        print(f"Framework:    {framework_name}")
        print(f"Pipeline:     {pipeline_name}")
        print(f"Encoding:     {encoding_name}")
        print(f"Device specs: {device_specs}")
        print("Results:")
        print(results)
    with open(output_path, "w") as f:
        f.write(numpy_encoder.NumpyEncoder().encode(results))


if __name__ == "__main__":
    main()
