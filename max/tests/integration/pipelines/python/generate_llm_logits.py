# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os

# Standard library
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

# 3rd-party
import click
import huggingface_hub
import llama3

# Tests
import replit_compat
import run_torch_llama
import torch
import transformers

# Pipelines
from architectures import register_all_models

# MAX
from max import driver, pipelines
from max.pipelines import interfaces
from max.pipelines.kv_cache import KVCacheStrategy
from test_common import evaluate, numpy_encoder


@dataclass
class MaxPipelineAndTokenizer:
    """An instantiated MAX pipeline and pieces necessary to run it."""

    model: pipelines.PipelineModel
    generator: Union[
        interfaces.TokenGenerator, pipelines.TextGenerationPipeline
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


class PipelineOracle:
    """Knows about a kind of pipeline.

    Can provide information about that pipeline, and create other objects
    necessary to run the model.
    """

    @property
    def supported_versions(self) -> Sequence[str]:
        """Versions of a model that are (ever) supported.

        Not all combinations of version, encoding, and device are necessarily
        supported.  To check support for a particular combination, use
        is_supported.

        Should be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def supported_encodings(self) -> Sequence[str]:
        """Encodings of a model that are (ever) supported.

        Not all combinations of version, encoding, and device are necessarily
        supported.  To check support for a particular combination, use
        is_supported.

        Should be overridden by subclasses.
        """
        raise NotImplementedError

    def is_supported(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        """Check that a particular version/encoding/device tuple is supported.

        Returns True if supported, False if not.
        """
        raise NotImplementedError

    def create_max_pipeline(
        self,
        *,
        version: str,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        """Instantiate a MAX pipeline for the given version/encoding/device."""
        raise NotImplementedError

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        """Instantiate a Torch pipeline for the given version/encoding/device."""
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
        return [evaluate.IMAGES_MULTI_MODEL]


class LlamaPipelineOracle(PipelineOracle):
    @property
    def supported_versions(self) -> Sequence[str]:
        return ["tinyllama", "llama3", "llama3_1"]

    @property
    def supported_encodings(self) -> Sequence[str]:
        return list(pipelines.SupportedEncoding)

    def is_supported(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        assert version in self.supported_versions
        assert encoding in self.supported_encodings
        if device_spec.device_type == "cpu":
            if encoding == "bfloat16":
                return False
        elif device_spec.device_type == "gpu":
            if encoding != "bfloat16":
                return False
        else:
            return False
        if version == "tinyllama":
            return encoding in ["float32", "bfloat16"]
        return True

    def _map_to_internal_version(self, version: str) -> str:
        assert version in self.supported_versions
        if version == "tinyllama" or "3_1" in version:
            return "3.1"
        else:
            return "3"

    def _weight_path_for(self, version: str, encoding: str) -> Path:
        if version == "tinyllama":
            testdata_directory = Path(os.environ["PIPELINES_TESTDATA"])
            if encoding == "float32":
                return testdata_directory / "tiny_llama.gguf"
            elif encoding == "bfloat16":
                return testdata_directory / "tiny_llama_bf16.gguf"
            else:
                raise ValueError(
                    f"Could not find tiny llama checkpoint for {encoding!r}"
                )
        return Path(
            llama3.config.get_llama_huggingface_file(
                self._map_to_internal_version(version),
                pipelines.SupportedEncoding[encoding],
            ).download()
        )

    def create_max_pipeline(
        self,
        *,
        version: str,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(
                version=version, encoding=encoding, device_spec=device_spec
            )
        internal_version = self._map_to_internal_version(version)
        config = pipelines.PipelineConfig(
            architecture="LlamaForCausalLM",
            version=self._map_to_internal_version(version),
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_new_tokens=10,
            huggingface_repo_id=f"modularai/llama-{internal_version}",
            weight_path=[
                self._weight_path_for(version=version, encoding=encoding)
            ],
            device_specs=device_specs,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def _config_path_for(self, version: str, encoding: str) -> Path:
        if version == "tinyllama":
            testdata_directory = Path(os.environ["PIPELINES_TESTDATA"])
            if encoding == "float32":
                return testdata_directory / "tiny_llama_config.json"
            elif encoding == "bfloat16":
                return testdata_directory / "tiny_llama_bf16_config.json"
            else:
                raise ValueError(
                    f"Could not find config path for tinyllama {encoding!r}"
                )
        hf_repo_id = f"modularai/llama-{self._map_to_internal_version(version)}"
        return Path(
            huggingface_hub.hf_hub_download(
                repo_id=hf_repo_id, filename="config.json"
            )
        )

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        # Tokenizer from testdata is used even for non-tiny Llama.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            Path(os.environ["PIPELINES_TESTDATA"])
        )
        config_path = self._config_path_for(version=version, encoding=encoding)
        weight_path = self._weight_path_for(version=version, encoding=encoding)
        config = transformers.AutoConfig.from_pretrained(config_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "UNUSED", config=config, gguf_file=weight_path, device_map=device
        )
        return TorchModelAndDataProcessor(model=model, data_processor=tokenizer)


class LlamaVisionPipelineOracle(MultiModalPipelineOracle):
    @property
    def supported_versions(self) -> Sequence[str]:
        return ["llama3_2"]

    @property
    def supported_encodings(self) -> Sequence[str]:
        return [pipelines.SupportedEncoding.bfloat16]

    def is_supported(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        assert version in self.supported_versions
        assert encoding in self.supported_encodings
        return device_spec.device_type in {"cpu", "gpu"}

    def create_max_pipeline(
        self,
        *,
        version: str,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(
                version=version, encoding=encoding, device_spec=device_spec
            )
        config = pipelines.PipelineConfig(
            architecture="MllamaForConditionalGeneration",
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy=KVCacheStrategy.CONTINUOUS,
            huggingface_repo_id="meta-llama/Llama-3.2-11B-Vision",
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
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        hf_repo_id = "meta-llama/Llama-3.2-11B-Vision"
        processor = transformers.AutoProcessor.from_pretrained(hf_repo_id)
        config = transformers.AutoConfig.from_pretrained(hf_repo_id)
        model = transformers.MllamaForConditionalGeneration.from_pretrained(
            hf_repo_id,
            config=config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


class ReplitPipelineOracle(PipelineOracle):
    @property
    def supported_versions(self) -> Sequence[str]:
        return ["replit-code-v1_5-3b"]

    @property
    def supported_encodings(self) -> Sequence[str]:
        return ["bfloat16", "float32"]

    def is_supported(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        assert version in self.supported_versions
        assert encoding in self.supported_encodings
        if device_spec.device_type == "cpu":
            if encoding == "bfloat16":
                return False
        elif device_spec.device_type == "gpu":
            if encoding != "bfloat16":
                return False
        else:
            return False
        return True

    def create_max_pipeline(
        self,
        *,
        version: str,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(
                version=version, encoding=encoding, device_spec=device_spec
            )
        config = pipelines.PipelineConfig(
            architecture="MPTForCausalLM",
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            huggingface_repo_id="modularai/replit-code-1.5",
            trust_remote_code=True,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)
        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            # Unlike the other pipelines, replit.Replit is both a model and a
            # generator at the same time.
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        # Need to use upstream instead of modularai/replit-code-1.5, because
        # the modularai version does not have the custom Python code needed
        # (also why trust_remote_code is needed).  Without this, we get:
        #     ValueError: `attn_type` has to be either `multihead_attention` or
        #     `multiquery_attention`. Received: grouped_query_attention
        hf_repo_id = "replit/replit-code-v1_5-3b"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_repo_id, trust_remote_code=True
        )
        config = transformers.AutoConfig.from_pretrained(
            hf_repo_id, trust_remote_code=True
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
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id, config=config, device_map=device, trust_remote_code=True
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
    def supported_versions(self) -> Sequence[str]:
        return ["nemo-instruct-2407"]

    @property
    def supported_encodings(self) -> Sequence[str]:
        return ["bfloat16"]

    def is_supported(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        assert version in self.supported_versions
        assert encoding in self.supported_encodings
        return device_spec.device_type == "gpu"

    def create_max_pipeline(
        self,
        *,
        version: str,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        for device_spec in device_specs:
            assert self.is_supported(
                version=version, encoding=encoding, device_spec=device_spec
            )
        config = pipelines.PipelineConfig(
            architecture="MistralForCausalLM",
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            weight_path=[
                pipelines.HuggingFaceFile(
                    "mistralai/Mistral-Nemo-Instruct-2407", f
                ).download()
                for f in [
                    "model-00001-of-00005.safetensors",
                    "model-00002-of-00005.safetensors",
                    "model-00003-of-00005.safetensors",
                    "model-00004-of-00005.safetensors",
                    "model-00005-of-00005.safetensors",
                ]
            ],
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)

        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        hf_repo_id = "mistralai/Mistral-Nemo-Instruct-2407"
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_repo_id)
        config = transformers.AutoConfig.from_pretrained(hf_repo_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
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
    def supported_versions(self) -> Sequence[str]:
        return ["pixtral12b"]

    @property
    def supported_encodings(self) -> Sequence[str]:
        return [pipelines.SupportedEncoding.bfloat16]

    def is_supported(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> bool:
        assert version in self.supported_versions
        assert encoding in self.supported_encodings
        return device_spec.device_type == "gpu"

    def create_max_pipeline(
        self,
        *,
        version: str,
        encoding: str,
        device_specs: list[driver.DeviceSpec],
    ) -> MaxPipelineAndTokenizer:
        # TODO (AIPIPE-234): Implement MAX pipeline generation for Pixtral.
        for device_spec in device_specs:
            assert self.is_supported(
                version=version, encoding=encoding, device_spec=device_spec
            )
        hf_repo_id = "mistral-community/pixtral-12b"
        config = pipelines.PipelineConfig(
            device_specs=device_specs,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            huggingface_repo_id=hf_repo_id,
        )
        tokenizer, pipeline = pipelines.PIPELINE_REGISTRY.retrieve(config)

        assert isinstance(pipeline, pipelines.TextGenerationPipeline)
        return MaxPipelineAndTokenizer(
            model=pipeline._pipeline_model,
            generator=pipeline,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndDataProcessor:
        hf_repo_id = "mistral-community/pixtral-12b"
        processor = transformers.AutoProcessor.from_pretrained(hf_repo_id)
        config = transformers.AutoConfig.from_pretrained(hf_repo_id)
        model = transformers.LlavaForConditionalGeneration.from_pretrained(
            hf_repo_id,
            config=config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        return TorchModelAndDataProcessor(model=model, data_processor=processor)


PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    "llama": LlamaPipelineOracle(),
    "replit": ReplitPipelineOracle(),
    "mistral": MistralPipelineOracle(),
    "llama3-vision": LlamaVisionPipelineOracle(),
    "pixtral": PixtralPipelineOracle(),
}


@click.command()
@click.option(
    "--device",
    "device_type",
    type=click.Choice(["cpu", "gpu"]),
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
    type=click.Choice(list(PIPELINE_ORACLES.keys())),
    required=True,
    help="Pipeline to run",
)
@click.option(
    "--version",
    "version_name",
    required=True,
    help="Weight family and architecture variant to run pipeline with",
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
    version_name: str,
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
    if version_name not in pipeline_oracle.supported_versions:
        raise ValueError(
            f"Version {version_name!r} not one of supported versions "
            f"{pipeline_oracle.supported_versions}"
        )
    if encoding_name not in pipeline_oracle.supported_encodings:
        raise ValueError(
            f"Encoding {encoding_name!r} not one of supported encodings "
            f"{pipeline_oracle.supported_encodings}"
        )

    device_specs: list[driver.DeviceSpec]
    if device_type == "cpu":
        device_specs = [driver.DeviceSpec.cpu()]
    elif device_type == "gpu":
        device_specs = [driver.DeviceSpec.cuda()]
    else:
        raise ValueError(f"Unknown device type {device_type!r}")
    for device_spec in device_specs:
        if not pipeline_oracle.is_supported(
            version=version_name,
            encoding=encoding_name,
            device_spec=device_spec,
        ):
            raise ValueError(
                "Combination of version/encoding/device not supported"
            )

    if framework_name == "max":
        max_pipeline_and_tokenizer = pipeline_oracle.create_max_pipeline(
            version=version_name,
            encoding=encoding_name,
            device_specs=device_specs,
        )
        results = evaluate.run_model(
            max_pipeline_and_tokenizer.model,
            max_pipeline_and_tokenizer.tokenizer,
            pipeline_oracle.prompts,
        )
    elif framework_name == "torch":
        torch_device: torch.device
        if device_type == "cpu":
            torch_device = torch.device("cpu")
        elif device_type == "gpu":
            torch_device = torch.device("cuda:0")
        else:
            raise ValueError(
                f"Device type {device_type!r} not supported for Torch"
            )
        torch_pipeline_and_tokenizer = pipeline_oracle.create_torch_pipeline(
            version=version_name, encoding=encoding_name, device=torch_device
        )
        # Despite the name, run_torch_llama3 works for all transformers, not
        # just Llama.
        results = run_torch_llama.run_torch_llama3(
            model=torch_pipeline_and_tokenizer.model,
            data_processor=torch_pipeline_and_tokenizer.data_processor,
            device=torch_device,
            prompts=pipeline_oracle.prompts,
            images=pipeline_oracle.images
            if isinstance(pipeline_oracle, MultiModalPipelineOracle)
            else None,
        )
    else:
        raise NotImplementedError(
            f"Framework {framework_name!r} not implemented"
        )

    if print_output:
        print(f"Framework: {framework_name}")
        print(f"Pipeline:  {pipeline_name}")
        print(f"Version:   {version_name}")
        print(f"Encoding:  {encoding_name}")
        print(f"Device:    {device_type}")
        print("Results:")
        print(results)
    with open(output_path, "w") as f:
        f.write(numpy_encoder.NumpyEncoder().encode(results))


if __name__ == "__main__":
    main()
