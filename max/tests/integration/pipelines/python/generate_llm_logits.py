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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

# 3rd-party
import click
import hf_repo_lock
import huggingface_hub

# Tests
import replit_compat
import requests
import torch
import transformers

# MAX
from max import driver, pipelines
from max.entrypoints.cli import DevicesOptionType
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import interfaces
from max.pipelines.lib import PipelineEngine
from test_common import (
    evaluate,
    evaluate_embeddings,
    numpy_encoder,
    torch_utils,
)
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
    "gptq": torch.float16,
    "q4_k": torch.float32,
    "q4_0": torch.float32,
    "q6_k": torch.float32,
}


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
        self, *, encoding: str, device: torch.device | str
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

    @property
    def use_cache(self) -> bool:
        """Whether to use the KV cache, for HF transformers models only."""
        return True


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
            engine=PipelineEngine.MAX,
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
            engine=PipelineEngine.MAX,
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
        #     torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
        # )
        # However we receive this error if we do:
        #     ValueError: Architecture mpt not supported
        # So we cannot use GGUF here.
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            revision=revision,
            config=config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=ENCODING_TO_TORCH_DTYPE[encoding],
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)

    @property
    def prompts(self) -> Sequence[str]:
        # Default prompts are too long for MAX Replit.
        # Truncate the prompts so it fits.
        prompt_length_limit = 2000
        return [prompt[:prompt_length_limit] for prompt in super().prompts]


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
            engine=PipelineEngine.MAX,
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
        config_params: dict[str, Any] = {},
        prompts: list[str] | None = None,
        use_cache: bool = True,
        auto_model_cls: Any = transformers.AutoModelForCausalLM,
        auto_processor_cls: Any = transformers.AutoTokenizer,
        task: interfaces.PipelineTask = interfaces.PipelineTask.TEXT_GENERATION,
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
            engine=PipelineEngine.MAX,
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
    def prompts(self) -> Sequence[str]:
        return self._prompts or evaluate.PROMPTS

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
    "llama3.3-70b": GenericOracle(
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "replit": ReplitPipelineOracle(),
    "mistral": GenericOracle(
        model_path="mistralai/Mistral-Nemo-Instruct-2407",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "mistral3": GenericOracle(
        model_path="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
    "llama3-vision": LlamaVisionPipelineOracle(),
    "pixtral": PixtralPipelineOracle(),
    "qwen": GenericOracle(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        config_params={"max_length": 512},
        device_encoding_map={"gpu": ["bfloat16"]},
    ),
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
        config_params={"max_length": 8192},
        device_encoding_map={"gpu": ["bfloat16"]},
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
def main(
    device_type: str | list[int],
    framework_name: str,
    pipeline_name: str,
    encoding_name: str,
    output_path: Path,
    print_output: bool,
    max_batch_size: int,
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
                batch_size=max_batch_size,
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
                batch_size=max_batch_size,
            )
        else:
            raise ValueError(
                f"Evaluating task {pipeline_oracle.task} is not supported."
            )
    elif framework_name == "torch":
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

        torch_pipeline_and_tokenizer = pipeline_oracle.create_torch_pipeline(
            encoding=encoding_name, device=device
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
                use_cache=pipeline_oracle.use_cache,
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
