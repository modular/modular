# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Library for evaluating and comparing Llama3 results.

Can also be used as a standalone binary to save out the golden values as a JSON.
"""

import asyncio
import base64
import itertools
import os
import subprocess
import uuid
from dataclasses import dataclass
from json import JSONDecoder, JSONEncoder
from pathlib import Path
from typing import Any, Literal, Optional

import click
import numpy as np
import numpy.typing as npt
from cpuinfo import get_cpu_info
from huggingface_hub import hf_hub_download
from llama3.config import (
    DeviceSpec,
    InferenceConfig,
    SupportedEncodings,
    SupportedVersions,
)
from llama3.llama3 import Llama3, Llama3Context
from max.driver import CPU
from nn.kv_cache import KVCacheStrategy


def find_runtime_path(
    fname: str,
    testdata_directory: Path,
    subdir: Path = Path("test_llama_golden"),
) -> Path:
    try:
        from python.runfiles import runfiles
    except ModuleNotFoundError:
        # Default to expecting data in the testdata directory when running
        # outside Bazel.
        return testdata_directory / fname

    r = runfiles.Create()
    path = r.Rlocation(str(subdir / fname))

    if path is None:
        raise Exception(f"Runtime path for {fname} was not found.")
    else:
        print(f"Runtime path for {fname} was located at {path}")
    return Path(path)


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__np__": base64.b64encode(obj.tobytes()).decode("ascii"),
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
        elif isinstance(obj, np.generic):
            return obj.item()
        return JSONEncoder.default(self, obj)


class NumpyDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )

    def object_hook(self, dct):
        if "__np__" in dct:
            shape = dct["shape"]
            dtype = np.dtype(dct["dtype"])
            return np.frombuffer(
                base64.b64decode(dct["__np__"]), dtype=dtype
            ).reshape(shape)
        return dct


NUM_STEPS = 10
PROMPTS = (
    """One of the most important aspects of performance benchmarking when it pertains to comparison of different implementations is making sure comparisons are fair. This is a place where most discussions occur, as deviation from best practices can make one’s performance claims easy to dismiss. For faster results of a given implementation (the Mojo implementation in our case) to be meaningful, the comparison needs to be apples-to-apples.
    * Make sure you use equivalent optimization flags across implementations; even though flags (like -O3 in C) that enable multiple optimizations at once cannot always be equivalent to another language’s -O3, make sure you don’t compare something like a debug build with an implementation that uses the fast optimization flag.
    * Make sure that if one implementation has auto-vectorization or automatic multithreading enabled the same applies to all implementations to be compared (unless for a given language one of these performs worse when turned-on, in which case one could keep the fastest implementation for comparison purposes).
    * Use the latest (or best) combination of compilers, libraries, etc. — an older compiler version (for example) may perform better for whatever reason; however it should be considered sufficient to test with the latest stable version. One can test with older or experimental versions if they are so inclined.
    * Use the same input file (if applicable) or same input data. Avoid random data generation that may stress different code paths.
    * Use the same algorithm (if applicable) across all your implementations.
    * Use equivalent error testing as it applies to different domains’ best practices (e.g., input sanitizing, corner case testing).
    * Remove any unnecessary I/O (e.g., writing to file/screen for debug purposes) and keep only what is practically necessary — make sure you do so in a manner that code is not optimized out (see #6)!
    * Try to apply the same level of manual optimization (within reason) — if you write multi-threaded/vectorized code in Mojo, you should try to compare it to an equivalent implementation of the other language. There is a case to be made here, however, if the other language does not have such capabilities or they are so difficult to use that implementing them is beyond what one can reasonably do. This can highlight the programmability aspect of Mojo (or one language against another more generally), but this fact should be listed so that people can take the performance claims under this light.""",
    "def is_prime(x):\n",
    "The meaning of life is ",
    """Translate the English text to Italian.
    Text: Sometimes, I've believed as many as six impossible things before breakfast.
    Translation:""",
)


def llama3_encode(llama3: Llama3, prompt: str):
    """Utility function to encode a string to list of token ids."""
    return llama3._encode(prompt)


def llama3_decode(llama3: Llama3, token_ids: list):
    """Utility function to decode a list of tokens to a string."""
    return llama3._tokenizer.decode(token_ids)


def run_llama3(llama3: Llama3, prompts=PROMPTS, num_steps=NUM_STEPS):
    results = []
    # Evaluate prompts individually (not batched).
    for prompt in prompts:
        context = asyncio.run(llama3.new_context(prompt))
        curr_req_id = str(uuid.uuid4())
        values: dict[str, list[Any]] = {curr_req_id: []}
        for _ in range(num_steps):
            next_token_with_logits(llama3, {curr_req_id: context}, values)
        results.append({"prompt": prompt, "values": values[curr_req_id]})
        asyncio.run(llama3.release(context))
    return results


def next_token_with_logits(
    llama3: Llama3,
    req_to_context_dict: dict[str, Llama3Context],
    update_values: dict[str, list[Any]],
):
    """Generates the next token and stores the logits.

    This method runs llama3.execute, stores the logits, and updates the context
    with the next token.

    Args:
        llama3: Llama3 model to execute.
        req_to_context_dict: Dictionary of request ids to Llama3Context.
        update_values: Dictionary of request ids to lists of next_token &
            logits. These lists are updated in this method.
    """
    logits = llama3._execute(req_to_context_dict).to(CPU())

    for req_id, logits in zip(req_to_context_dict, logits.to_numpy()):
        next_token = logits.argmax(axis=-1)
        update_values[req_id].append(
            {
                "next_token": next_token,
                "next_token_logits": logits[next_token],
                "logits": logits,
            }
        )
        # Update the context for the next input.
        req_to_context_dict[req_id].next_tokens = next_token.reshape(-1)


def compare_values(actual, expected, *, rtol=1e-2, atol=1e-5, compare_fn=None):
    """Compares the values between two computed logits.

    The data structure of the actual/expected logits should be:
    [
        {"prompt": "prompt 1", "values": [{"key": value, ...}],
        {"prompt": "prompt 2", "values": [{"key": value, ...}],
        ...
    ]

    The "values" list contains the logits at each step for the prompt.

    The `actual` logits structure must be a subset of the expected logits.
    E.g. if the `expected` values contains logits for 10 steps, the `actual`
    values can contain any number of steps between 1-10.

    Args:
        actual: Data structure containing computed values.
        expected: Data structure containing reference values.
        rtol: The relative tolerance (used if `compare_fn` is not provided).
        atol: The absolute tolerance (used if `compare_fn` is not provided).
        compare_fn: A callable that takes the arguments
            (actual, expected, description).
    """
    expected_prompts = {x["prompt"]: x["values"] for x in expected}
    actual_prompts = {x["prompt"]: x["values"] for x in actual}

    if expected_prompts.keys() < actual_prompts.keys():
        diff = actual_prompts.keys() - expected_prompts.keys()
        raise ValueError(
            f"Golden values for prompts {diff} not found. Please re-run"
            " `gen_golden_values`."
        )

    for prompt, values in actual_prompts.items():
        expected_values = expected_prompts[prompt]
        actual_steps = len(values)
        expected_steps = len(expected_values)
        assert actual_steps <= expected_steps

        for step in range(actual_steps):
            inference_results = values[step]
            expected_results = expected_values[step]

            for key, value in inference_results.items():
                expected_value = expected_results[key]
                short = f"{prompt[:15]}..." if len(prompt) > 15 else prompt
                description = f"'{key}' on step={step} for the prompt='{short}'"
                if compare_fn:
                    compare_fn(value, expected_value, description)
                else:
                    np.testing.assert_allclose(
                        value,
                        expected_value,
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"Got different values for {description}.",
                        verbose=True,
                    )


@dataclass(frozen=True)
class _SupportedModelEncoding:
    """Standardizes config and golden filename based on the model and encoding.

    This class should not be used directly, instead use SupportedTestModel.
    """

    model: str
    encoding: SupportedEncodings

    @classmethod
    def init(cls, model, encoding):
        """Initialize with type cast."""
        return cls(model, SupportedEncodings[encoding])

    def _tiny_llama_weights(self, testdata_directory: Optional[Path]) -> Path:
        if not testdata_directory:
            raise ValueError("Please pass `testdata_directory`")
        if self.encoding == SupportedEncodings.float32:
            return testdata_directory / "tiny_llama.gguf"
        elif self.encoding == SupportedEncodings.bfloat16:
            return testdata_directory / "tiny_llama_bf16.gguf"
        else:
            raise ValueError(
                f"Could not find tiny llama checkpoint for {self.encoding=}."
            )

    @property
    def version(self) -> SupportedVersions:
        if self.model == "tinyllama":
            return SupportedVersions.llama3_1
        else:
            return SupportedVersions[self.model]

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
    ) -> InferenceConfig:
        if "max_new_tokens" not in kwargs and "max_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 10
        if "weight_path" not in kwargs:
            if self.model == "tinyllama":
                kwargs["weight_path"] = self._tiny_llama_weights(
                    testdata_directory
                )
            else:
                version = SupportedVersions[self.model]
                repo_id = f"modularai/llama-{version}"
                kwargs["weight_path"] = hf_hub_download(
                    repo_id=repo_id,
                    filename=self.encoding.hf_model_name(version),
                )
        if "device_spec" not in kwargs:
            kwargs[
                "device_spec"
            ] = DeviceSpec.cuda() if self.use_gpu else DeviceSpec.cpu()
        if "max_new_tokens" not in kwargs and "max_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 10

        return InferenceConfig(
            version=self.version,
            quantization_encoding=self.encoding,
            **kwargs,
        )

    def golden_data_fname(self, *, framework: Literal["max", "torch"] = "max"):
        # TODO(MSDK-948): Actually support a distinction between device and encoding
        # instead of letting bfloat16 _imply_ GPU as is done multiple times in this file
        if self.encoding == "bfloat16":
            result = subprocess.run(
                [os.environ["MODULAR_CUDA_QUERY_PATH"]], stdout=subprocess.PIPE
            )
            hardware = (
                result.stdout.decode()
                .split("name:")[1]
                .split("\n")[0]
                .strip()
                .replace(" ", "")
            )
        else:
            # TODO: MSDK-968 address the hardware variance that makes
            # this untenable
            # info = _system_info()
            # hardware = info["arch"]
            hardware = "all"

        # This becomes a file path
        hardware = hardware.replace(" ", "")

        if framework == "max":
            return f"{self.model}_{self.encoding}_{hardware}_golden.json"
        else:
            return f"{framework}_{self.model}_{self.encoding}_{hardware}_golden.json"

    def hf_config_path(self, testdata_directory: Optional[Path] = None) -> Path:
        if self.model == "tinyllama":
            if not testdata_directory:
                raise ValueError(
                    "Need testdata_directory for tiny llama config path."
                )
            if self.encoding == SupportedEncodings.float32:
                return testdata_directory / "tiny_llama_config.json"
            elif self.encoding == SupportedEncodings.bfloat16:
                return testdata_directory / "tiny_llama_bf16_config.json"
            else:
                raise ValueError(
                    f"Could not find config path for tinyllama {self.encoding}."
                )
        else:
            return Path(
                hf_hub_download(repo_id=self.hf_repo_id, filename="config.json")
            )


def _system_info():
    result = subprocess.run(
        [os.getenv("MODULAR_SYSTEM_INFO_PATH")], stdout=subprocess.PIPE
    )
    system_info = {}
    for line in result.stdout.decode().split("\n"):
        try:
            k, v = line.split(": ")
            system_info[k.strip()] = v.strip()
        except:
            pass

    return system_info


class SupportedTestModels:
    """Models with supported golden data files.

    Usage: `SupportedTestModels.TINY_LLAMA_F32.golden_data_fname()`

    Also works with parametrized tests:

    ```
    @pytest.mark.parametrize(...)
    def test_model(model, encoding):
        test_model = SupportedTestModels.get(model, encoding)
        config = test_model.build_config()
        golden_file = test_model.golden_data_fname()
    ```

    """

    LLAMA_3_1_BF16 = _SupportedModelEncoding.init("llama3_1", "bfloat16")
    LLAMA_3_1_Q4_K = _SupportedModelEncoding.init("llama3_1", "q4_k")
    TINY_LLAMA_F32 = _SupportedModelEncoding.init("tinyllama", "float32")
    TINY_LLAMA_BF16 = _SupportedModelEncoding.init("tinyllama", "bfloat16")

    _supported_pairs: dict[tuple[str, str], _SupportedModelEncoding] = {}

    @staticmethod
    def get(model, encoding, strict=True):
        """Returns the supported model encoding object.

        Args:
            model: Name of model ("tinyllama" or "llama3_1")
            encoding: A SupportedEncoding or str ("float32", "bfloat16", etc.).
            strict: When strict mode is enabled, an error if the model and
                encoding isn't in the pre-defined pairs. You can disable this
                error by setting this option to False.

        Returns:
            A model encoding object that can be used to construct an
            InferenceConfig or get the golden data filename.
        """
        try:
            return SupportedTestModels._supported_pairs[(model, encoding)]
        except KeyError:
            if not strict:
                return _SupportedModelEncoding.init(model, encoding)
            raise ValueError(
                f"{model=} {encoding=} does not have golden values. If you're "
                "sure, please set `strict=False` when calling "
                "`SupportedTestModels.get`."
            )


# Supported models and encodings are defined in SupportedTestModels.
ALL_SUPPORTED_MODELS = {"all"}
ALL_SUPPORTED_ENCODINGS = {"all"}

for attr in dir(SupportedTestModels):
    value = getattr(SupportedTestModels, attr)
    if isinstance(value, _SupportedModelEncoding):
        SupportedTestModels._supported_pairs[
            (value.model, value.encoding)
        ] = value
        ALL_SUPPORTED_MODELS.add(value.model)
        ALL_SUPPORTED_ENCODINGS.add(str(value.encoding))


def supported_model_encodings(model, encoding, strict=False):
    # TODO: Use driver to check if cuda available
    if encoding == "all":
        encodings = ALL_SUPPORTED_ENCODINGS - {"all"}
    else:
        encodings = [encoding]
    if model == "all":
        models = ALL_SUPPORTED_MODELS - {"all"}
    else:
        models = [model]

    for encoding, model in itertools.product(encodings, models):
        try:
            yield SupportedTestModels.get(model, encoding, strict=False)
        except:
            print(
                "Skipping golden generation for"
                f" {model=} {encoding=} (combination not supported)."
            )
            continue


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
    testdata_directory = os.getenv("PIPELINES_TESTDATA")
    if testdata_directory is None:
        raise ValueError("Environmental PIPELINES_TESTDATA not defined.")
    testdata_directory = Path(testdata_directory)
    encoder = NumpyEncoder()
    for model_encoding in supported_model_encodings(
        model, encoding, strict=False
    ):
        try:
            config = model_encoding.build_config(testdata_directory)
            llama3 = Llama3(config)
            results = run_llama3(llama3, PROMPTS)

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
