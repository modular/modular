# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging

from collections import defaultdict
from typing import Any, Optional, TextIO
from contextlib import ExitStack
from pathlib import Path

import click
from huggingface_hub import hf_hub_download

from max.driver import CPU
from max.pipelines import TokenGenerator

from llama3 import (
    InferenceConfig,
    Llama3,
)
from utils import config_to_flag

"""
Batch Sequence
start-time, max-new-tokens
"""
REQUEST_SEQUENCE = [
    [0, 16],
    [0, 24],
    [0, 32],
    [8, 16],  # Added to TokenGenBatch after ContextEncoding
    [8, 32],  # Stays in TokenGenQueue until first completes
    [16, 16],  # Stays in TokenGenQueue until second completes
]

# As opposed to continuous batching
# When set - the batch is executed until ALL items of the batch are completed.
FORCE_DYNAMIC_BATCHING = False


async def run_batch_scenario(
    model: TokenGenerator,
    prompts: list[str],
    output_path: str,
):
    logging.basicConfig(
        level=logging.INFO,
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting scenario with %d batches and %d prompts",
        len(REQUEST_SEQUENCE),
        len(prompts),
    )

    max_batch_size = model.config.batch_size
    num_prompts = len(prompts)

    batch_sequence_index = 0
    step = 0

    context_encoding_queue: list[tuple[str, str, int]] = []
    token_gen_queue: list[tuple[str, Any]] = []  # Queued batched
    token_gen_batch: dict[str, Any] = {}  # Active batch

    batch_configs: dict[str, tuple[str, int]] = {}
    batch_completions: dict[str, str] = defaultdict(str)
    batch_output_files: dict[str, TextIO] = {}

    exit_stack = ExitStack()

    resolved_output_path = Path(output_path).resolve() if output_path else None
    if resolved_output_path:
        resolved_output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output Path: %s", resolved_output_path)

    def get_batch_output_file(batch_id: str) -> Optional[TextIO]:
        if not resolved_output_path:
            return None
        file_path = resolved_output_path / f"{batch_id}.txt"
        file = open(file_path, "w")
        exit_stack.push(file)
        return file

    with exit_stack:
        while True:
            step_logger = logger.getChild(str(step))
            step_logger.info("")

            if (
                (batch_sequence_index == len(REQUEST_SEQUENCE))
                and (not context_encoding_queue)
                and (not token_gen_queue)
                and (not token_gen_batch)
            ):
                logger.info("Completed scenario")
                break

            # Check if we have new batches available.
            while (batch_sequence_index < len(REQUEST_SEQUENCE)) and (
                step == REQUEST_SEQUENCE[batch_sequence_index][0]
            ):
                # Reuse prompts as needed
                batch_id = str(batch_sequence_index)
                prompt = prompts[batch_sequence_index % num_prompts]
                max_new_tokens = REQUEST_SEQUENCE[batch_sequence_index][1]
                batch_configs[batch_id] = (prompt, max_new_tokens)
                step_logger.info(
                    "Received: Batch %s, '%s', %d",
                    batch_id,
                    prompt,
                    max_new_tokens,
                )
                context_encoding_queue.append(
                    (batch_id, prompt, max_new_tokens)
                )
                batch_sequence_index += 1

            # Perform context encoding for new batches
            if context_encoding_queue:
                # When set, defer context encoding until there are no active token generation batches
                if not FORCE_DYNAMIC_BATCHING or not token_gen_batch:
                    step_logger.info(
                        (
                            "Encoding Started: %d batches queued [%s], token"
                            " gen queue: %d"
                        ),
                        len(context_encoding_queue),
                        ",".join(b[0] for b in context_encoding_queue),
                        len(token_gen_queue),
                    )

                    encoded_count = 0
                    while context_encoding_queue:
                        context_encoding_batch = {}  # type: ignore
                        while (
                            context_encoding_queue
                            and len(context_encoding_batch) < max_batch_size
                        ):
                            batch_id, prompt, max_new_tokens = (
                                context_encoding_queue.pop(0)
                            )
                            context = await model.new_context(
                                prompt, max_new_tokens
                            )
                            context_encoding_batch[batch_id] = context
                            batch_out_file = get_batch_output_file(batch_id)
                            if batch_out_file:
                                batch_output_files[batch_id] = batch_out_file
                                batch_out_file.write(context.prompt)

                        step_logger.info(
                            "Encoding Batched: %d batches [%s], %d remaining",
                            len(context_encoding_batch),
                            ",".join(context_encoding_batch.keys()),
                            len(context_encoding_queue),
                        )

                        context_encoding_results = await model.next_token(
                            context_encoding_batch
                        )
                        assert (
                            context_encoding_results.keys()
                            == context_encoding_batch.keys()
                        )
                        encoded_count += len(context_encoding_results)
                        for batch_id, token in context_encoding_results.items():
                            assert token is not None
                            batch_completions[batch_id] = token
                            batch_context = context_encoding_batch[batch_id]
                            step_logger.info(
                                "Encoded: %s, %d/%d, Completion:\n%s",
                                batch_id,
                                len(batch_context.tokens),
                                batch_context.max_tokens,
                                batch_completions[batch_id],
                            )
                            batch_out_file = batch_output_files.get(batch_id)
                            if batch_out_file:
                                batch_out_file.write(token)

                            token_gen_queue.append((batch_id, batch_context))

                    step_logger.info(
                        (
                            "Encoding Completed: %d encoded, token gen queue:"
                            " %d, [%s]"
                        ),
                        encoded_count,
                        len(token_gen_queue),
                        ",".join(b[0] for b in token_gen_queue),
                    )

            # When set, only add new batches when there are no other ones active.
            if not FORCE_DYNAMIC_BATCHING or not token_gen_batch:
                # Add queued batches to active batch when there is capacity
                while token_gen_queue and (
                    len(token_gen_batch) - max_batch_size
                ):
                    batch = token_gen_queue.pop(0)
                    batch_id, batch_context = batch
                    token_gen_batch[batch_id] = batch_context
                    step_logger.info(
                        "Dequeued: %s, Total-Queue: %d, Active-Batch: [%s]",
                        batch_id,
                        len(token_gen_queue),
                        ",".join(token_gen_batch.keys()),
                    )

            # Run token generation on active batches
            if token_gen_batch:
                step_logger.info(
                    "Executing: Active-Batch: [%s]",
                    ",".join(token_gen_batch.keys()),
                )

                results = await model.next_token(token_gen_batch)
                for batch_id, token in results.items():
                    assert token is not None
                    batch_completions[batch_id] += token
                    step_logger.info(
                        "Executed: %s, %d/%d - Completion:\n%s",
                        batch_id,
                        len(batch_context.tokens),
                        batch_context.max_tokens,
                        batch_completions[batch_id],
                    )
                    batch_out_file = batch_output_files.get(batch_id)
                    if batch_out_file:
                        batch_out_file.write(token)

                # Remove completed batches
                completed_batch_ids = token_gen_batch.keys() - results.keys()
                if completed_batch_ids:
                    step_logger.info(
                        "Completed: [%s]", ",".join(completed_batch_ids)
                    )
                    for batch_id in completed_batch_ids:
                        del token_gen_batch[batch_id]

            step += 1

    summary_text = ""
    for batch_id, batch_completion in batch_completions.items():
        batch_prompt = batch_configs[batch_id][0]
        batch_max_tokens = batch_configs[batch_id][1]
        summary_text += (
            f"Batch: {batch_id}, MaxTokens: {batch_max_tokens}\nPrompt:"
            f" {batch_prompt}\nCompletion:\n{batch_completion}\n\n"
        )
    logger.info("\n%s", summary_text)


@click.command
@config_to_flag(InferenceConfig)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--prompt-count",
    type=int,
    default=0,
    help="Set to 1 or more to run a batching scenario.",
)
@click.option(
    "--output-path",
    type=str,
    default="",
    help="Optional path to write outputs for batching scenario.",
)
def main(
    prompt: str,
    prompt_count: int,
    output_path: str,
    **config_kwargs,
):
    config_kwargs.update({"device": CPU()})
    config = InferenceConfig(**config_kwargs)
    # By default, use the Modular HF repository as a reference for tokenizer
    # configuration, etc. when no repository is specified.
    repo_id = f"modularai/llama-{config.version}"
    if config.weight_path is None:
        if config.huggingface_weights is not None:
            components = config.huggingface_weights.split("/")
            assert len(components) == 3, (
                "invalid Hugging Face weight location:"
                f" {config.huggingface_weights}, "
            )
            repo_id = f"{components[0]}/{components[1]}"
            weight_filename = components[2]
        else:
            weight_filename = config.quantization_encoding.hf_model_name(
                config.version
            )

        config.weight_path = hf_hub_download(
            repo_id=repo_id,
            filename=weight_filename,
        )

    model = Llama3(config)
    print("Starting batch demo")
    prompts = [prompt.format(i + 1) for i in range(prompt_count)]
    asyncio.run(
        run_batch_scenario(
            model,
            prompts,
            output_path,
        )
    )


if __name__ == "__main__":
    main()
