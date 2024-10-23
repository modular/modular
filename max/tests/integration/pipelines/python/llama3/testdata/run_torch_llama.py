# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Executes the Torch Llama model and exports the logits."""
import os
from pathlib import Path
from typing import Iterable

import click
import torch
from evaluate_llama import (
    ALL_SUPPORTED_ENCODINGS,
    ALL_SUPPORTED_MODELS,
    PROMPTS,
    NumpyEncoder,
    supported_model_encodings,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)


def create_torch_llama3(
    config_path: Path, gguf_weight_path: Path, use_gpu: bool
):
    device = torch.device("cuda:0" if use_gpu else "cpu")
    config = AutoConfig.from_pretrained(config_path)
    return AutoModelForCausalLM.from_pretrained(
        "UNUSED", config=config, gguf_file=gguf_weight_path, device_map=device
    )


def create_tokenizer(tokenizer_directory: Path):
    return AutoTokenizer.from_pretrained(tokenizer_directory)


def run_torch_llama3(llama3, tokenizer, prompts: Iterable[str], num_steps=10):
    saved_logits = []

    def store_logits(input_ids: torch.LongTensor, scores: torch.FloatTensor):
        _ = input_ids  # Unused.
        scores = scores[0]  # Currently always passing in one batch at a time.
        scores_np = scores.cpu().detach().numpy()
        next_token = scores_np.argmax(axis=-1)
        saved_logits.append(
            {
                "next_token": next_token,
                "next_token_logits": scores_np[next_token],
                "logits": scores_np,
            }
        )
        return scores

    results = []
    for prompt in prompts:
        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
        mask = torch.ones_like(encoded_prompt)
        llama3.generate(
            input_ids=encoded_prompt,
            attention_mask=mask,
            max_new_tokens=num_steps,
            do_sample=False,
            logits_processor=LogitsProcessorList([store_logits]),
            num_return_sequences=1,
        )
        results.append({"prompt": prompt, "values": saved_logits[:]})
        saved_logits.clear()

    return results


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
    tokenizer = create_tokenizer(testdata_directory)
    for model_encoding in supported_model_encodings(
        model, encoding, strict=False
    ):
        if encoding == "bfloat16" and not torch.cuda.is_available():
            print("Skipping {model}")
            continue
        try:
            hf_config_path = model_encoding.hf_config_path(testdata_directory)
            max_config = model_encoding.build_config(testdata_directory)
            llama3 = create_torch_llama3(
                hf_config_path, max_config.weight_path, model_encoding.use_gpu
            )
            results = run_torch_llama3(llama3, tokenizer, PROMPTS)
            output_full_path = os.path.join(
                "/tmp", model_encoding.golden_data_fname(framework="torch")
            )
            if verbose:
                print(f"===Results for {model} {encoding}")
                print(results)
            with open(output_full_path, "w") as f:
                f.write(encoder.encode(results))
            print(
                f"Torch goldens for {model} {encoding} written to",
                output_full_path,
            )
        except Exception as e:
            print(f"Failed to generate golden data for {model}_{encoding}: {e}")
            raise e


if __name__ == "__main__":
    main()
