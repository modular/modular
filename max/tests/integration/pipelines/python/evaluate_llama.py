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
import json
from json import JSONEncoder
from pathlib import Path

import click
import numpy as np
from llama3 import (
    InferenceConfig,
    Llama3,
    SupportedEncodings,
    SupportedVersions,
)


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__np__": base64.b64encode(obj.tobytes()).decode("ascii"),
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
        return JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
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
    "What is the meaning of life?",
    # More can be added later, such as:
    # "Tell me a story about a cat.",
    # "def is_prime(x):\n",
)


def run_llama3(weight_path, prompts=PROMPTS, num_steps=NUM_STEPS):
    config = InferenceConfig(
        weight_path=weight_path,
        version=SupportedVersions.llama3_1,
        quantization_encoding=SupportedEncodings.float32,
        max_new_tokens=10,
    )
    llama3 = Llama3(config)

    results = []
    for prompt in prompts:
        context = asyncio.run(llama3.new_context(prompt))
        llama3._reset_cache()
        inference_results = []
        for _ in range(num_steps):
            logits, k_cache, v_cache = llama3._execute(context)
            next_token = logits.argmax(axis=-1)[-1]
            inference_results.append(
                {
                    "next_token": next_token,
                    # Only store `next_token_logits` otherwise the golden file
                    # gets too big.
                    "next_token_logits": logits[0, -1][next_token],
                    "kv_cache": k_cache,
                    "v_cache": v_cache,
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


@click.command
@click.option("--output", type=Path)
@click.option("--weight-path", type=Path, default="/tmp/tiny-llama.gguf")
def main(output, weight_path):
    results = run_llama3(weight_path, PROMPTS)
    encoder = NumpyEncoder()
    with open(output, "w") as f:
        f.write(encoder.encode(results))
    print("Golden file written to", output)


if __name__ == "__main__":
    main()
