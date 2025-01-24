# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for evaluating and comparing Llama3 results.

Can also be used as a standalone binary to save out the golden values as a JSON.
"""

import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import click
from huggingface_hub import hf_hub_download
from llama3 import Llama3Model
from llama3.config import get_llama_huggingface_file
from max.driver import DeviceSpec
from max.engine import InferenceSession
from max.pipelines import PipelineConfig, SupportedEncoding, TextTokenizer
from max.pipelines.kv_cache import KVCacheStrategy
from test_common.evaluate import PROMPTS, run_model
from test_common.numpy_encoder import NumpyEncoder
from test_common.path import golden_data_fname


@dataclass(frozen=True)
class SupportedTestModels:
    """Standardizes config and golden filename based on the model and encoding.

    Usage:

    ```
    @pytest.mark.parametrize(...)
    def test_model(model, encoding):
        test_model = SupportedTestModels.get(model, encoding)
        config = test_model.build_config()
        golden_file = test_model.golden_data_fname()
    ```

    """

    model: str
    """Model Type. Can be `llama3`, `llama3_1`, or `tinyllama`"""

    encoding: SupportedEncoding
    """The supported dtype."""

    @classmethod
    def get(cls, model, encoding):
        """Initialize with type cast."""
        return cls(model, SupportedEncoding[encoding])

    def _tiny_llama_weights(
        self, testdata_directory: Optional[Path]
    ) -> list[Path]:
        if not testdata_directory:
            raise ValueError("Please pass `testdata_directory`")
        if self.encoding == SupportedEncoding.float32:
            return [Path(testdata_directory / "tiny_llama.gguf")]
        elif self.encoding == SupportedEncoding.bfloat16:
            return [Path(testdata_directory / "tiny_llama_bf16.gguf")]
        else:
            raise ValueError(
                f"Could not find tiny llama checkpoint for {self.encoding=}."
            )

    @property
    def version(self) -> str:
        if self.model == "tinyllama" or "3_1" in self.model:
            return "3.1"
        else:
            return "3"

    @property
    def hf_repo_id(self) -> str:
        return f"modularai/llama-{self.version}"

    @property
    def use_gpu(self) -> bool:
        return self.encoding == "bfloat16"

    def build_config(
        self,
        testdata_directory: Optional[Path] = None,
        **kwargs,
    ) -> PipelineConfig:
        if "max_new_tokens" not in kwargs and "max_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 10

        if "weight_path" not in kwargs:
            if self.model == "tinyllama":
                kwargs["weight_path"] = self._tiny_llama_weights(
                    testdata_directory
                )
            else:
                hf_file = get_llama_huggingface_file(
                    self.version, self.encoding
                )
                kwargs["weight_path"] = [hf_file.download()]
                kwargs["huggingface_repo_id"] = hf_file.repo_id

        if "device_specs" not in kwargs:
            kwargs["device_specs"] = (
                [DeviceSpec.accelerator()]
                if self.use_gpu
                else [DeviceSpec.cpu()]
            )

        if "cache_strategy" not in kwargs:
            kwargs["cache_strategy"] = (
                KVCacheStrategy.CONTINUOUS
                if self.encoding in ["bfloat16", "float32"]
                else KVCacheStrategy.NAIVE
            )

        if "huggingface_repo_id" not in kwargs:
            if self.version == "3.1":
                kwargs["huggingface_repo_id"] = "modularai/llama-3.1"
            elif self.version == "3":
                kwargs["huggingface_repo_id"] = "modularai/llama-3"
            else:
                raise ValueError(f"version {self.version} not supported.")

        if "device_memory_utilization" not in kwargs:
            kwargs["device_memory_utilization"] = 0.1

        config = PipelineConfig(
            quantization_encoding=self.encoding,
            **kwargs,
        )

        # # Temporary hack to load TinyLlama config until we migrate tests to
        # # SmolLM.
        if self.model == "tinyllama":
            config.huggingface_config.intermediate_size = 500
            config.huggingface_config.hidden_size = 16
            config.huggingface_config.num_hidden_layers = 1
            config.huggingface_config.num_key_value_heads = 1
            config.huggingface_config.num_attention_heads = 1

        return config

    def golden_data_fname(self, *, framework: Literal["max", "torch"] = "max"):
        return golden_data_fname(self.model, self.encoding, framework=framework)

    def hf_config_path(self, testdata_directory: Optional[Path] = None) -> Path:
        if self.model == "tinyllama":
            if not testdata_directory:
                raise ValueError(
                    "Need testdata_directory for tiny llama config path."
                )
            if self.encoding == SupportedEncoding.float32:
                return testdata_directory / "tiny_llama_config.json"
            elif self.encoding == SupportedEncoding.bfloat16:
                return testdata_directory / "tiny_llama_bf16_config.json"
            else:
                raise ValueError(
                    f"Could not find config path for tinyllama {self.encoding}."
                )
        else:
            return Path(
                hf_hub_download(repo_id=self.hf_repo_id, filename="config.json")
            )


ALL_SUPPORTED_MODELS = {"all", "tinyllama", "llama3", "llama3_1"}

ALL_SUPPORTED_ENCODINGS = {"all"}
for encoding in SupportedEncoding:
    ALL_SUPPORTED_ENCODINGS.add(encoding.name)


def supported_model_encodings(model, encoding, strict=False):
    """Yields all supported combination of model and encodings."""
    # TODO: Use driver to check if cuda available
    if encoding == "all":
        encodings = ALL_SUPPORTED_ENCODINGS - {"all"}
    else:
        encodings = set([encoding])
    if model == "all":
        models = ALL_SUPPORTED_MODELS - {"all"}
    else:
        models = set([model])

    for encoding, model in itertools.product(encodings, models):
        yield SupportedTestModels.get(model, encoding)


@click.command
@click.option(
    "--model",
    type=click.Choice(list(ALL_SUPPORTED_MODELS)),
    default="all",
)
@click.option(
    "--encoding",
    type=click.Choice(list(ALL_SUPPORTED_ENCODINGS)),
    default="all",
)
@click.option(
    "--verbose",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to print the results of the evaluated logits.",
)
def main(model, encoding, verbose):
    testdata_directory_str = os.getenv("PIPELINES_TESTDATA")
    if testdata_directory_str is None:
        raise ValueError("Environmental PIPELINES_TESTDATA not defined.")
    testdata_directory = Path(testdata_directory_str)
    encoder = NumpyEncoder()
    for model_encoding in supported_model_encodings(
        model, encoding, strict=False
    ):
        try:
            config = model_encoding.build_config(testdata_directory)
            tokenizer = TextTokenizer(config)

            session = InferenceSession(devices=config.devices)
            llama3 = Llama3Model(pipeline_config=config, session=session)
            results = run_model(llama3, tokenizer, PROMPTS)

            output_full_path = os.path.join(
                "/tmp", model_encoding.golden_data_fname()
            )
            if verbose:
                print(f"===Results for {model} {encoding}")
                print(results)
            with open(output_full_path, "w") as f:
                f.write(encoder.encode(results))
            print(
                f"Goldens for {model} {encoding} written to", output_full_path
            )
        except Exception as e:
            print(f"Failed to generate golden data for {model}_{encoding}: {e}")
            raise e


if __name__ == "__main__":
    main()
