# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, TextIO

import click
from architectures import register_all_models
from cli import DevicesOptionType, config_to_flag
from max.driver import DeviceSpec
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    PipelineTokenizer,
    SupportedEncoding,
    TokenGenerator,
    TokenGeneratorContext,
    TokenGeneratorRequest,
)
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)

logger = logging.getLogger(__name__)


@dataclass(order=True)
class RequestInstance:
    req_id: str
    max_token_len: int
    arrival_time: float


class BatchExecutionLog:
    """A utility container to log which requests were executed at each time step
    and print them out in a tabular form to validate manually."""

    time_to_request: list = []

    def append(self, timestamp: int, req_ids):
        """Append a set of request ids at a given time step."""
        self.time_to_request.append([timestamp, req_ids])

    def dump(self):
        """Dump the execution steps."""
        for t, r in self.time_to_request:
            req_list = ",".join(e for e in r)
            print(f"{t},{req_list}")


def generate_continuous_batch_requests(
    num_requests: int = 8,
) -> List[RequestInstance]:
    """Generates a sequence where the first request is much longer, and
    all the following requests are different lengths arriving at different times.
    We can measure this to make sure that the requests are being executed in
    a continuous batching manner.
    """
    request_list = []
    for i in range(num_requests):
        request_list.append(
            RequestInstance(
                req_id=f"req_{i}",
                max_token_len=20 if i == 0 else (i * 2),
                arrival_time=i * 2,
            )
        )
    return request_list


def generate_batch_requests(num_requests: int = 8) -> List[RequestInstance]:
    """Generates a set of request that have the arrive at the same time (t=0),
    and have the same max-token-length (10). Combined with prompt, we can get
    different execution behaviors.
    """
    request_list = []
    for i in range(num_requests):
        request_list.append(
            RequestInstance(req_id=f"req_{i}", max_token_len=10, arrival_time=0)
        )
    return request_list


def generate_simple_requests() -> List[RequestInstance]:
    """Generates a sequence of requests where 3 arrive at the same time but have
    different max-lengths, with one having a very long request (max-length). The
    second sequence of requests should overlap with this long request for
    continuous batching, otherwise it will wait for the first batch to complete.
    """
    _REQUEST_SEQUENCE = [
        [0, 16],
        [0, 24],
        [0, 32],
        [8, 16],  # Added to TokenGenBatch after ContextEncoding
        [8, 32],  # Stays in TokenGenQueue until first completes
        [16, 16],  # Stays in TokenGenQueue until second completes
    ]
    request_list = []
    for i, entry in enumerate(_REQUEST_SEQUENCE):
        request_list.append(
            RequestInstance(
                req_id=f"req_{i}",
                max_token_len=entry[1],
                arrival_time=entry[0],
            )
        )
    return request_list


async def run_batch_scenario(
    pipeline: TokenGenerator,
    tokenizer: PipelineTokenizer,
    model_name: str,
    prompts: list[str],
    batch_mode: str,
    batch_max_size: int,
    output_path: str,
    request_list: list[RequestInstance],
):
    logger.info(
        "Starting %s batching scenario with %d batches and %d prompts",
        batch_mode,
        len(request_list),
        len(prompts),
    )
    for r in request_list:
        logger.info("Request  : %s", r)

    num_prompts = len(prompts)
    batch_sequence_index = 0

    # Sort the entries by arrival time.
    ordered_request_list = request_list
    ordered_request_list.sort(key=lambda x: x.arrival_time)

    step_current: int = 0
    step_duration: int = 1
    batch_log = BatchExecutionLog()

    context_encoding_queue: list[tuple[str, str, int]] = []
    token_gen_queue: list[tuple[str, Any]] = []  # Queued batched
    token_gen_batch: dict[str, Any] = {}  # Active batch

    # Simulates dynamic batching. When set, the batch continues to be
    # executed until ALL items of the batch are completed.
    batch_mode_dynamic = batch_mode == "dynamic"
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
            step_logger = logger.getChild(str(step_current))
            step_logger.info("")

            if (
                (not ordered_request_list)
                and (not context_encoding_queue)
                and (not token_gen_queue)
                and (not token_gen_batch)
            ):
                # all queues are empty and all batches have been processed
                logger.info("Completed scenario")
                break

            # Check if we have new batches available.
            while (
                ordered_request_list
                and step_current >= ordered_request_list[0].arrival_time
            ):
                current_request = ordered_request_list.pop(0)
                # Reuse prompts as needed
                batch_id = current_request.req_id
                prompt = prompts[batch_sequence_index % max(num_prompts, 1)]
                max_new_tokens = current_request.max_token_len
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
                if not batch_mode_dynamic or not token_gen_batch:
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
                    start_idx = 0
                    while context_encoding_queue:
                        context_encoding_batch: Dict[  # type: ignore
                            str, TokenGeneratorContext
                        ] = {}
                        while (
                            context_encoding_queue
                            and (
                                len(context_encoding_batch)
                                + len(token_gen_queue)
                                + len(token_gen_batch)
                            )
                            < batch_max_size
                        ):
                            batch_id, prompt, max_new_tokens = (
                                context_encoding_queue.pop(0)
                            )
                            request = TokenGeneratorRequest(
                                id=batch_id,
                                index=start_idx,
                                prompt=prompt,
                                model_name=model_name,
                            )
                            context = await tokenizer.new_context(request)
                            start_idx += 1
                            context_encoding_batch[batch_id] = context
                            batch_out_file = get_batch_output_file(batch_id)
                            if batch_out_file:
                                batch_output_files[batch_id] = batch_out_file
                                batch_out_file.write(context.prompt)

                        if not context_encoding_batch:
                            break

                        step_logger.info(
                            "Encoding Batched: %d batches [%s], %d remaining",
                            len(context_encoding_batch),
                            ",".join(context_encoding_batch.keys()),
                            len(context_encoding_queue),
                        )

                        context_encoding_results = pipeline.next_token(
                            context_encoding_batch, num_steps=1
                        )[0]
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
                                batch_context.current_length,
                                batch_context.max_length,
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
            if not batch_mode_dynamic or not token_gen_batch:
                # Add queued batches to active batch when there is capacity
                while token_gen_queue and (
                    len(token_gen_batch) - batch_max_size
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
                batch_log.append(step_current, [x for x in token_gen_batch])
                results = pipeline.next_token(token_gen_batch, num_steps=1)[0]
                for batch_id, token in results.items():
                    assert token is not None
                    batch_completions[batch_id] += token
                    step_logger.info(
                        "Executed: %s, %d/%d - Completion:\n%s",
                        batch_id,
                        batch_context.current_length,
                        batch_context.max_length,
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
                        # TODO: What ever were we referring to here?  "model"
                        # is not a variable that actually exists.
                        await model.release(token_gen_batch[batch_id])  # type: ignore
                        del token_gen_batch[batch_id]

            step_current += step_duration  # assumes that token generation takes one time step.

    summary_text = ""
    for batch_id, batch_completion in batch_completions.items():
        batch_prompt = batch_configs[batch_id][0]
        batch_max_tokens = batch_configs[batch_id][1]
        summary_text += (
            f"Batch: {batch_id}, MaxTokens: {batch_max_tokens}\nPrompt:"
            f" {batch_prompt}\nCompletion:\n{batch_completion}\n\n"
        )
    logger.info("\n%s", summary_text)
    logger.info("COMPLETED_AT %s", step_current)
    batch_log.dump()


@click.command
@config_to_flag(PipelineConfig)
@click.option(
    "--prompt",
    type=str,
    multiple=True,
    default=["I believe the meaning of life is"],
    help="The text prompt to use for further generation.",
)
@click.option(
    "--prompt-count",
    type=int,
    default=4,
    help="Set to 1 or more to run a batching scenario.",
)
@click.option(
    "--use-gpu",
    is_flag=False,
    type=DevicesOptionType(),
    show_default=True,
    default="",
    flag_value="0",
    help=(
        "Whether to run the model on the available GPU. An ID value can be"
        " provided optionally to indicate the device ID to target."
    ),
)
@click.option(
    "--model-name",
    type=click.Choice(["llama3", "rev-echo"]),
    default="rev-echo",
    help="the model to use for evaluation",
)
@click.option(
    "--batch-mode",
    type=click.Choice(["dynamic", "continuous"]),
    default="dynamic",
    help="Configures the servers batching scheme",
)
@click.option(
    "--output-path",
    type=str,
    default="",
    help="Optional path to write outputs for batching scenario.",
)
@click.option(
    "--scenario",
    type=click.Choice(["A", "B", "C"]),
    default="A",
    help=(
        "Request scenarios. A=simple_requests, B=batch_requests,"
        " C=continuous_batch_requests"
    ),
)
def main(
    prompt: List[str],
    prompt_count: int,
    use_gpu: List[int],
    model_name: str,
    batch_mode: str,
    output_path: str,
    scenario: str,
    **config_kwargs,
):
    register_all_models()

    logging.basicConfig(
        level=logging.INFO,
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    if use_gpu:
        config_kwargs.update(
            {
                "device_specs": [
                    DeviceSpec.accelerator(id=gpu_id) for gpu_id in use_gpu
                ],
                "quantization_encoding": SupportedEncoding.bfloat16,
            }
        )
    else:
        config_kwargs.update({"device_specs": [DeviceSpec.cpu()]})

    if config_kwargs["architecture"] is None:
        config_kwargs["architecture"] = "LlamaForCausalLM"

    if config_kwargs["huggingface_repo_id"] is None:
        config_kwargs["huggingface_repo_id"] = "modularai/llama-3.1"

    config = PipelineConfig(**config_kwargs)

    if model_name == "rev-echo":
        tokenizer: PipelineTokenizer = EchoPipelineTokenizer()
        pipeline: TokenGenerator = EchoTokenGenerator()
    else:
        tokenizer, _pipeline = PIPELINE_REGISTRY.retrieve(config)
        # Do the following extra steps to resolve mypy type checking.
        assert isinstance(_pipeline, TokenGenerator)
        pipeline = _pipeline

    logger.info(
        "Loaded model %s, %s on %s",
        config.huggingface_repo_id,
        config.quantization_encoding,
        config.devices[0],
    )
    logger.info(
        "- Using weights %s",
        config.weight_path,
    )
    logger.info(
        "- MaxLength %d, MaxNewTokens %d",
        config.max_length,
        config.max_new_tokens,
    )
    logger.info(
        "- KVCache %s, MaxSize %d",
        config.cache_strategy,
        config.max_cache_batch_size,
    )

    logger.info("Starting batch demo")
    requests: list = []

    if scenario == "A":
        requests = generate_simple_requests()
    elif scenario == "B":
        requests = generate_batch_requests()
    elif scenario == "C":
        requests = generate_continuous_batch_requests()
    else:
        raise ValueError("Invalid scenario specified")

    assert config.max_cache_batch_size is not None
    asyncio.run(
        run_batch_scenario(
            pipeline=pipeline,
            tokenizer=tokenizer,
            model_name=model_name,
            prompts=prompt,
            batch_mode=batch_mode,
            batch_max_size=config.max_cache_batch_size,
            output_path=output_path,
            request_list=requests,
        )
    )


if __name__ == "__main__":
    """README
    ** Dynamic Batching Simulation **
    CPU
        Produces valid outputs until the first request in the batch completes - then garbage.
        bazelw run //SDK/integration-test/pipelines/python:evaluate_batch_scenarios -- --batch-mode dynamic --max-cache-batch-size 4 --model-name llama3 --quantization-encoding q4_k
    GPU
        Produces valid outputs until the first request in the batch completes - then garbage.
        bazelw run //SDK/integration-test/pipelines/python:evaluate_batch_scenarios -- --batch-mode dynamic --max-cache-batch-size 4 --model-name llama3

    ** Continuous Batching Simulation **
    CPU
        bazelw run //SDK/integration-test/pipelines/python:evaluate_batch_scenarios -- --batch-mode continuous --max-cache-batch-size 4 --model-name llama3 --quantization-encoding float32 --cache-strategy continuous
    GPU
        bazelw run //SDK/integration-test/pipelines/python:evaluate_batch_scenarios -- --batch-mode continuous --max-cache-batch-size 4 --model-name llama3 --cache-strategy continuous
    """
    main()
