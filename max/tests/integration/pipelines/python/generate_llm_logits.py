# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os

# Standard library
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

# 3rd-party
import click
import hf_repo_lock
import huggingface_hub

# Tests
import replit_compat
import torch
import transformers

# MAX
from max import driver, pipelines
from max.entrypoints.cli import DevicesOptionType
from max.pipelines import interfaces
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.llama3.config import get_llama_huggingface_file
from max.pipelines.kv_cache import KVCacheStrategy
from test_common import (
    evaluate,
    evaluate_embeddings,
    numpy_encoder,
    torch_utils,
)


@dataclass
class MaxPipelineAndTokenizer:
    """An instantiated MAX pipeline and pieces necessary to run it."""

    model: pipelines.PipelineModel
    generator: Union[
        pipelines.TextGenerationPipeline,
        pipelines.EmbeddingsPipeline,
    ]  # TODO(kcaverly): Move to only TextGenerationPipeline
    tokenizer: interfaces.PipelineTokenizer


@dataclass
class TorchModelAndDataProcessor:
    """An instantiated Torch model and pieces necessary to run it."""

    model: transformers.PreTrainedModel
    data_processor: Union[
        transformers.PreTrainedTokenizer,
        transformers.PreTrainedTokenizerFast,
        transformers.MllamaProcessor,
        transformers.PixtralProcessor,
    ]


class PipelineOracle(ABC):
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    task: interfaces.PipelineTask = interfaces.PipelineTask.TEXT_GENERATION

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
        self, *, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        """Instantiate a Torch pipeline for the given encoding/device."""
        raise NotImplementedError

    @property
    def prompts(self) -> Sequence[str]:
        """Prompts to run the model on.

        Should only be overridden if a pipeline has a particular reason the
        defaults are inappropriate.
        """
        return evaluate.PROMPTS


class MultiModalPipelineOracle(PipelineOracle):
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    @property
    def prompts(self) -> Sequence[str]:
        """Prompts to run a multi-modal model on."""
        return [evaluate.PROMPTS_MULTI_MODAL]

    @property
    def images(self) -> Optional[Sequence[str]]:
        """Images to run a multi-modal model on."""
        return [evaluate.IMAGES_MULTI_MODAL]


class Llama3_1PipelineOracle(PipelineOracle):
    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        supported_encodings = [e.name for e in pipelines.SupportedEncoding]
        return {
            "cpu": [e for e in supported_encodings if e != "bfloat16"],
            "gpu": ["float32", "bfloat16"],
        }

    def _weight_path_for(self, encoding: str) -> Path:
        return Path(
            get_llama_huggingface_file(
                "3.1",
                pipelines.SupportedEncoding[encoding],
                hf_repo_lock.revision_for_hf_repo("modularai/llama-3.1"),
            ).download()
        )

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        config = pipelines.PipelineConfig(
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            # Normally we do not override batch size, preferring to test the
            # defaults, but the default is known to be broken on A10.  Product
            # has requested that we work around rather than fix for now.  Do
            # NOT override, and instead leave as default, for CPU.
            max_batch_size=(
                32
                if any(
                    device_spec.device_type == "gpu"
                    for device_spec in device_specs
                )
                else None
            ),
            max_new_tokens=10,
            model_path="modularai/llama-3.1",
            weight_path=[self._weight_path_for(encoding=encoding)],
            device_specs=device_specs,
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
        # Tokenizer from testdata is used even for non-tiny Llama.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            Path(os.environ["PIPELINES_TESTDATA"])
        )
        config_path = Path(
            huggingface_hub.hf_hub_download(
                repo_id="modularai/llama-3.1",
                filename="config.json",
                revision=hf_repo_lock.revision_for_hf_repo(
                    "modularai/llama-3.1"
                ),
            )
        )
        weight_path = self._weight_path_for(encoding=encoding)
        config = transformers.AutoConfig.from_pretrained(config_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "UNUSED", config=config, gguf_file=weight_path, device_map=device
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)


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
            huggingface_revision=revision,
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
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class ExaonePipelineOracle(PipelineOracle):
    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "cpu": ["float32"],
            "gpu": ["float32"],
        }

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            model_path="LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_length=1024,
            # TODO(E2EOPT-48): Remove batch size override.
            max_batch_size=128,
            trust_remote_code=True,
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
        hf_repo_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision, trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)


class ReplitPipelineOracle(PipelineOracle):
    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "cpu": ["float32"],
            "gpu": ["bfloat16", "float32"],
        }

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            model_path="modularai/replit-code-1.5",
            trust_remote_code=True,
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
        # Need to use upstream instead of modularai/replit-code-1.5, because
        # the modularai version does not have the custom Python code needed
        # (also why trust_remote_code is needed).  Without this, we get:
        #     ValueError: `attn_type` has to be either `multihead_attention` or
        #     `multiquery_attention`. Received: grouped_query_attention
        hf_repo_id = "replit/replit-code-v1_5-3b"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, revision=revision, trust_remote_code=True
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision, trust_remote_code=True
        )
        replit_compat.monkeypatch_transformers()
        # Ideally we would still use our GGUF weights:
        # weight_path = replit.config.get_replit_huggingface_file(
        #     pipelines.SupportedEncoding[encoding]
        # ).download()
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     "UNUSED",
        #     config=config,
        #     gguf_file=weight_path,
        #     device_map=device,
        #     trust_remote_code=True,
        # )
        # However we receive this error if we do:
        #     ValueError: Architecture mpt not supported
        # So we cannot use GGUF here.
        torch_dtype: torch.dtype
        if encoding == "float32":
            torch_dtype = torch.float32
        elif encoding == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Could not convert encoding {encoding} to a torch dtype."
            )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)

    @property
    def prompts(self) -> Sequence[str]:
        # Default prompts are too long for MAX Replit.
        # Truncate the prompts so it fits.
        prompt_length_limit = 2000
        return [prompt[:prompt_length_limit] for prompt in super().prompts]


class MistralPipelineOracle(PipelineOracle):
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
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            model_path="mistralai/Mistral-Nemo-Instruct-2407",
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_length=512,
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
        hf_repo_id = "mistralai/Mistral-Nemo-Instruct-2407"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)


class PixtralPipelineOracle(MultiModalPipelineOracle):
    @property
    def prompts(self) -> Sequence[str]:
        """Prompts to run a multi-modal model on."""
        return [evaluate.PIXTRAL_PROMPT]

    @property
    def images(self) -> Optional[Sequence[str]]:
        """Images to run a multi-modal model on."""
        return [evaluate.PIXTRAL_IMG_URL]

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
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class QwenPipelineOracle(PipelineOracle):
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
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            model_path="Qwen/Qwen2.5-7B-Instruct",
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_length=512,
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
        hf_repo_id = "Qwen/Qwen2.5-7B-Instruct"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)


class Llama3_3_70BInstructPipelineOracle(PipelineOracle):
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
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            model_path="meta-llama/Llama-3.3-70B-Instruct",
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_length=512,
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
        hf_repo_id = "meta-llama/Llama-3.3-70B-Instruct"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            # Set device map to auto and just hope for enough GPU VRAM.
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)

    @property
    def prompts(self) -> Sequence[str]:
        return evaluate.PROMPTS


class OlmoPipelineOracle(PipelineOracle):
    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return {
            "cpu": ["float32"],
            "gpu": ["float32"],
        }

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            model_path="allenai/OLMo-1B-hf",
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_length=1024,
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
        hf_repo_id = "allenai/OLMo-1B-hf"
        revision = hf_repo_lock.revision_for_hf_repo(hf_repo_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, revision=revision
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, revision=revision
        )
        assert encoding == "float32"
        torch_dtype = torch.float32
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            torch_dtype=torch_dtype,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)


class GenericOracle(PipelineOracle):
    def __init__(
        self,
        *,
        model_path: str,
        device_encoding_map: dict[str, list[str]],
        config_params: dict[str, Any] = {},
        prompts: list[str] | None = None,
        auto_model_cls: Any = transformers.AutoModelForCausalLM,
        auto_processor_cls: Any = transformers.AutoTokenizer,
        task: interfaces.PipelineTask = interfaces.PipelineTask.TEXT_GENERATION,
    ) -> None:
        self.model_path = model_path
        self.config_params = config_params
        self._prompts = prompts
        self.auto_model_cls = auto_model_cls
        self.auto_processor_cls = auto_processor_cls
        self.task = task
        self._device_encoding_map = device_encoding_map

    @property
    def device_encoding_map(self) -> dict[str, list[str]]:
        return self._device_encoding_map

    def create_max_pipeline(
        self, *, encoding: str, device_specs: list[driver.DeviceSpec]
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(encoding=encoding, device_spec=device_spec)
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            model_path=self.model_path,
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
        processor = self.auto_processor_cls.from_pretrained(self.model_path)
        torch_dtype: torch.dtype
        if encoding == "float32":
            torch_dtype = torch.float32
        elif encoding == "bfloat16":
            torch_dtype = torch.bfloat16
        elif encoding == "gptq":
            torch_dtype = torch.float16
        else:
            raise ValueError(
                f"Could not convert encoding {encoding} to a torch dtype."
            )
        model = self.auto_model_cls.from_pretrained(
            self.model_path,
            revision=hf_repo_lock.revision_for_hf_repo(self.model_path),
            device_map=device,
            torch_dtype=torch_dtype,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)

    @property
    def prompts(self) -> Sequence[str]:
        return self._prompts or evaluate.PROMPTS


PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    "olmo": OlmoPipelineOracle(),
    "exaone": ExaonePipelineOracle(),
    "llama3_1": Llama3_1PipelineOracle(),
    "llama3.3-70b": Llama3_3_70BInstructPipelineOracle(),
    "replit": ReplitPipelineOracle(),
    "mistral": MistralPipelineOracle(),
    "llama3-vision": LlamaVisionPipelineOracle(),
    "pixtral": PixtralPipelineOracle(),
    "qwen": QwenPipelineOracle(),
    "smollm": GenericOracle(
        model_path="HuggingFaceTB/SmolLM2-135M",
        config_params={
            "max_length": 512,
            "cache_strategy": KVCacheStrategy.CONTINUOUS,
        },
        prompts=[p[:502] for p in evaluate.PROMPTS],
        device_encoding_map={
            "cpu": ["float32", "q4_k", "q4_0", "q6_k", "gptq"],
            "gpu": ["float32", "bfloat16"],
        },
    ),
    "mpnet": GenericOracle(
        model_path="sentence-transformers/all-mpnet-base-v2",
        # Maximum length accepted by MPNet tokenizer is 512.
        config_params={"max_length": 512, "pool_embeddings": False},
        prompts=[p[:502] for p in evaluate.PROMPTS],
        auto_model_cls=transformers.AutoModel,
        task=interfaces.PipelineTask.EMBEDDINGS_GENERATION,
        device_encoding_map={
            "cpu": ["float32"],
            "gpu": ["float32", "bfloat16"],
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
}


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
def main(
    device_type: str,
    framework_name: str,
    pipeline_name: str,
    encoding_name: str,
    output_path: Path,
    print_output: bool,
) -> None:
    """Output logits to a file for a model based on a fixed set of prompts.

    The resulting logit golden files for two different frameworks can be used
    with //SDK/integration-test/pipelines/python/llama3/verify to check their
    similarity.
    """

    # Register all models.
    register_all_models()

    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)

    pipeline_oracle = PIPELINE_ORACLES[pipeline_name]

    device_specs = DevicesOptionType.device_specs(device_type)
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

    if framework_name == "max":
        max_pipeline_and_tokenizer = pipeline_oracle.create_max_pipeline(
            encoding=encoding_name,
            device_specs=device_specs,
        )
        if pipeline_oracle.task == interfaces.PipelineTask.TEXT_GENERATION:
            results = evaluate.run_model(
                max_pipeline_and_tokenizer.model,
                max_pipeline_and_tokenizer.tokenizer,
                prompts=pipeline_oracle.prompts,
                images=pipeline_oracle.images
                if isinstance(pipeline_oracle, MultiModalPipelineOracle)
                else None,
                print_outputs=True,
            )
        elif (
            pipeline_oracle.task
            == interfaces.PipelineTask.EMBEDDINGS_GENERATION
        ):
            assert isinstance(
                max_pipeline_and_tokenizer.generator,
                pipelines.EmbeddingsPipeline,
            )
            results = evaluate_embeddings.encode(
                max_pipeline_and_tokenizer.generator,
                max_pipeline_and_tokenizer.tokenizer,
                pipeline_oracle.prompts,
            )
        else:
            raise ValueError(
                f"Evaluating task {pipeline_oracle.task} is not supported."
            )
    elif framework_name == "torch":
        torch_device: torch.device
        if device_type == "cpu":
            torch_device = torch.device("cpu")
        elif device_type == "gpu" or isinstance(device_type, list):
            # Set device to device 0 even for multi-GPU: model parallelism is
            # handled at model construction time in `create_torch_pipeline`.
            torch_device = torch.device("cuda:0")

        torch_pipeline_and_tokenizer = pipeline_oracle.create_torch_pipeline(
            encoding=encoding_name, device=torch_device
        )
        if pipeline_oracle.task == interfaces.PipelineTask.TEXT_GENERATION:
            results = torch_utils.run_text_generation(
                model=torch_pipeline_and_tokenizer.model,
                data_processor=torch_pipeline_and_tokenizer.data_processor,
                device=torch_device,
                prompts=pipeline_oracle.prompts,
                images=pipeline_oracle.images
                if isinstance(pipeline_oracle, MultiModalPipelineOracle)
                else None,
                print_outputs=True,
            )
        elif (
            pipeline_oracle.task
            == interfaces.PipelineTask.EMBEDDINGS_GENERATION
        ):
            results = torch_utils.run_embeddings_generation(
                model=torch_pipeline_and_tokenizer.model,
                data_processor=torch_pipeline_and_tokenizer.data_processor,
                device=torch_device,
                prompts=pipeline_oracle.prompts,
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
        print(f"Framework: {framework_name}")
        print(f"Pipeline:  {pipeline_name}")
        print(f"Encoding:  {encoding_name}")
        print(f"Device:    {device_type}")
        print("Results:")
        print(results)
    with open(output_path, "w") as f:
        f.write(numpy_encoder.NumpyEncoder().encode(results))


if __name__ == "__main__":
    main()
