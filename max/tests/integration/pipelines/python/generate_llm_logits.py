# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

# Standard library
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

# 3rd-party
import click
import huggingface_hub
import torch
import transformers

# MAX
from max import driver
from max import pipelines
from max.pipelines import interfaces, TextTokenizer
from max.pipelines import kv_cache

# Pipelines
import llama3
import replit
import replit.config
import mistral
import nn.tokenizer

# Tests
import replit_compat
import run_torch_llama
from test_common import evaluate
from test_common import numpy_encoder


@dataclass
class MaxPipelineAndTokenizer:
    """An instantiated MAX pipeline and pieces necessary to run it."""

    model: Any  # TODO(kathywu): Update to PipelineModel
    generator: interfaces.TokenGenerator
    tokenizer: interfaces.TokenGeneratorTokenizer


@dataclass
class TorchModelAndTokenizer:
    """An instantiated Torch LLM model and pieces necessary to run it."""

    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer


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
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> MaxPipelineAndTokenizer:
        """Instantiate a MAX pipeline for the given version/encoding/device."""
        raise NotImplementedError

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndTokenizer:
        """Instantiate a Torch pipeline for the given version/encoding/device.
        """
        raise NotImplementedError

    @property
    def prompts(self) -> Sequence[str]:
        """Prompts to run the model on.

        Should only be overridden if a pipeline has a particular reason the
        defaults are inappropriate.
        """
        return evaluate.PROMPTS


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
        elif device_spec.device_type == "cuda":
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
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> MaxPipelineAndTokenizer:
        assert self.is_supported(
            version=version, encoding=encoding, device_spec=device_spec
        )
        internal_version = self._map_to_internal_version(version)
        config = pipelines.PipelineConfig(
            architecture="llama",
            version=self._map_to_internal_version(version),
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            max_new_tokens=10,
            huggingface_repo_id=f"modularai/llama-{internal_version}",
            weight_path=self._weight_path_for(
                version=version, encoding=encoding
            ),
            device_spec=device_spec,
            cache_strategy=(
                kv_cache.KVCacheStrategy.CONTINUOUS if encoding
                in ["bfloat16", "float32"] else kv_cache.KVCacheStrategy.NAIVE
            ),
        )
        tokenizer = TextTokenizer(config)
        generator = llama3.Llama3TokenGenerator(
            config, tokenizer.eos, tokenizer.delegate.vocab_size
        )
        return MaxPipelineAndTokenizer(
            model=generator.model, generator=generator, tokenizer=tokenizer
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
    ) -> TorchModelAndTokenizer:
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
        return TorchModelAndTokenizer(model=model, tokenizer=tokenizer)


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
        elif device_spec.device_type == "cuda":
            if encoding != "bfloat16":
                return False
        else:
            return False
        return True

    def create_max_pipeline(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> MaxPipelineAndTokenizer:
        assert self.is_supported(
            version=version, encoding=encoding, device_spec=device_spec
        )
        config = pipelines.PipelineConfig(
            architecture="replit",
            device_spec=device_spec,
            quantization_encoding=pipelines.SupportedEncoding[encoding],
            cache_strategy=(
                kv_cache.KVCacheStrategy.CONTINUOUS if encoding
                in ["bfloat16", "float32"] else kv_cache.KVCacheStrategy.NAIVE
            ),
            huggingface_repo_id="modularai/replit-code-1.5",
            weight_path=replit.config.get_replit_huggingface_file(
                pipelines.SupportedEncoding[encoding]
            ).download(),
        )
        generator = replit.Replit(config)
        tokenizer = TextTokenizer(config)
        return MaxPipelineAndTokenizer(
            # Unlike the other pipelines, replit.Replit is both a model and a
            # generator at the same time.
            model=generator,
            generator=generator,
            tokenizer=tokenizer,
        )

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndTokenizer:
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
        return TorchModelAndTokenizer(model=model, tokenizer=tokenizer)

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
        return device_spec.device_type == "cuda"

    def create_max_pipeline(
        self, *, version: str, encoding: str, device_spec: driver.DeviceSpec
    ) -> MaxPipelineAndTokenizer:
        assert self.is_supported(
            version=version, encoding=encoding, device_spec=device_spec
        )
        config = mistral.InferenceConfig(
            weight_path=huggingface_hub.hf_hub_download(
                repo_id="mistralai/Mistral-Nemo-Instruct-2407",
                filename="consolidated.safetensors",
            ),
            device_spec=device_spec,
            max_new_tokens=10,
        )
        tokenizer = mistral.MistralTokenizer(config)
        generator = mistral.MistralTokenGenerator(config, tokenizer.eos)
        return MaxPipelineAndTokenizer(
            model=generator.model, generator=generator, tokenizer=tokenizer
        )

    def create_torch_pipeline(
        self, *, version: str, encoding: str, device: torch.device
    ) -> TorchModelAndTokenizer:
        hf_repo_id = "mistralai/Mistral-Nemo-Instruct-2407"
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_repo_id)
        config = transformers.AutoConfig.from_pretrained(hf_repo_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_repo_id, config=config, device_map=device
        )
        return TorchModelAndTokenizer(model=model, tokenizer=tokenizer)


PIPELINE_ORACLES: Mapping[str, PipelineOracle] = {
    "llama": LlamaPipelineOracle(),
    "replit": ReplitPipelineOracle(),
    "mistral": MistralPipelineOracle(),
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

    device_spec: driver.DeviceSpec
    if device_type == "cpu":
        device_spec = driver.DeviceSpec.cpu()
    elif device_type == "gpu":
        device_spec = driver.DeviceSpec.cuda()
    else:
        raise ValueError(f"Unknown device type {device_type!r}")

    if not pipeline_oracle.is_supported(
        version=version_name, encoding=encoding_name, device_spec=device_spec
    ):
        raise ValueError("Combination of version/encoding/device not supported")

    if framework_name == "max":
        max_pipeline_and_tokenizer = pipeline_oracle.create_max_pipeline(
            version=version_name,
            encoding=encoding_name,
            device_spec=device_spec,
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
            torch_pipeline_and_tokenizer.model,
            torch_pipeline_and_tokenizer.tokenizer,
            torch_device,
            pipeline_oracle.prompts,
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
