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
from json import JSONDecoder, JSONEncoder
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
from cpuinfo import get_cpu_info
from huggingface_hub import hf_hub_download
from llama3 import (
    InferenceConfig,
    Llama3,
    SupportedEncodings,
    SupportedVersions,
)
from max.driver import CPU, CUDA, Device


def find_runtime_path(fname):
    from python.runfiles import runfiles

    r = runfiles.Create()
    path = r.Rlocation("test_llama_golden/" + fname)

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


def load_llama3(weight_path: str, **kwargs):
    config = InferenceConfig(
        weight_path=weight_path,
        version=SupportedVersions.llama3_1,
        quantization_encoding=SupportedEncodings.float32,
        **kwargs,
    )
    llama3 = Llama3(config)
    return llama3


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
    # More can be added later, such as:
    # "Tell me a story about a cat.",
    # "def is_prime(x):\n",
)


def run_llama3(llama3: Llama3, prompts=PROMPTS, num_steps=NUM_STEPS):
    results = []
    for prompt in prompts:
        llama3._reset_cache()
        context = asyncio.run(llama3.new_context(prompt))
        inference_results: list[dict[str, npt.NDArray]] = []

        curr_req_id = uuid.uuid4()
        for _ in range(num_steps):
            logits_dict = llama3._execute({curr_req_id: context})
            for req_id, logits in logits_dict.items():
                next_token = logits.argmax(axis=-1)[-1]
                inference_results.append(
                    {
                        "next_token": next_token,
                        "next_token_logits": logits[0, next_token],
                        "logits": logits.reshape(-1),
                    }
                )

                # Update the context for the next input.
                context.next_tokens = next_token.reshape(1, -1)

        results.append({"prompt": prompt, "values": inference_results})
    return results


def compare_values(actual, expected):
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
                # TODO(MSDK-1025): Add logits to A100 and A10G golden test files and
                # delete this.
                if (key == "logits") and (key not in expected_results):
                    continue

                expected_value = expected_results[key]
                np.testing.assert_allclose(
                    value,
                    expected_value,
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=(
                        f"Got different values for the computed {key} on step"
                        f" {step}."
                    ),
                    verbose=True,
                )


def build_config(version, weight_path, encoding):
    if encoding == "bfloat16":
        device = CUDA()
    else:
        device = CPU()

    config = InferenceConfig(
        weight_path=weight_path,
        device=device,
        version=SupportedVersions[version],
        quantization_encoding=SupportedEncodings[encoding],
        max_new_tokens=10,
    )

    if weight_path is None:
        repo_id = f"modularai/llama-{config.version}"
        config.weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=config.quantization_encoding.hf_model_name(config.version),
        )

    return config


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


def golden_data_fname(model, encoding):
    # TODO(MSDK-948): Actually support a distinction between device and encoding
    # instead of letting bfloat16 _imply_ GPU as is done multiple times in this file
    if encoding == "bfloat16":
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

    return f"{model}_{encoding}_{hardware}_golden.json"


SUPPORTED_PAIRS = [
    ("llama3_1", "bfloat16"),
    # ("llama3_1", "q4_k"),
    ("tinyllama", "float32"),
]


@click.command
@click.option("--modular-path", type=Path, required=True)
@click.option(
    "--model",
    type=click.Choice(["llama3_1", "tinyllama", "all"]),
    default="all",
)
@click.option(
    "--encoding",
    type=click.Choice(["bfloat16", "float32", "q4_k", "all"]),
    default="all",
)
def main(model, modular_path, encoding):
    testdata_path = modular_path / Path(os.getenv("PIPELINES_TESTDATA"))

    # There must be a slicker way to do this expansion
    encodings = [encoding]
    models = [model]
    if encoding == "all":
        encodings = ["bfloat16", "float32", "q4_k"]
    if model == "all":
        models = ["llama3_1", "tinyllama"]

    for encoding, model in itertools.product(encodings, models):
        if (model, encoding) in SUPPORTED_PAIRS:
            weight_path = (
                testdata_path / "tiny_llama.gguf" if model
                == "tinyllama" else None
            )
            version = "llama3_1" if model == "tinyllama" else model
            try:
                config = build_config(version, weight_path, encoding)

                encoding = config.quantization_encoding.value
                llama3 = Llama3(config)
                results = run_llama3(llama3, PROMPTS)
                encoder = NumpyEncoder()

                output_full_path = testdata_path / golden_data_fname(
                    model, encoding
                )

                with open(output_full_path, "w") as f:
                    f.write(encoder.encode(results))

                print("Golden file written to", output_full_path)
            except Exception as e:
                print(
                    "Failed to generate golden data for"
                    f" {model}_{encoding}: {e}"
                )
                raise e


if __name__ == "__main__":
    main()
