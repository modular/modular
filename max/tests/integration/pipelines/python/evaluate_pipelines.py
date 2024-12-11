# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Evaluates pipelines on tasks in the Language Model Evaluation Harness.

List of all tasks:
https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.5/lm_eval/tasks

"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dataclasses import dataclass
import click
import lm_eval
import numpy as np

from generate_llm_logits import PIPELINE_ORACLES, MaxPipelineAndTokenizer
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from max import driver
from max.pipelines import interfaces
from scipy.special import log_softmax
from test_common import evaluate
from tqdm import tqdm


@dataclass(frozen=True)
class Example:
    context: str
    """Context string."""

    continuation: str
    """The continuation over which log likelihood will be calculated."""

    encoded_context: list[int]
    """Encoded context. Must be <= max sequence length that can be handled by
    the model."""

    truncated_context: str
    """Decoded `encoded_context`. Is generally the same as `context`, unless
    the `encode_context` has been truncated to fit the max seq length."""

    encoded_continuation: int
    """Encoded continuation token. Currently we only support a single
    continuation token."""

    @property
    def cache_key(self) -> tuple[str, str]:
        return (self.context, self.continuation)

    @classmethod
    async def from_request(
        cls,
        context: str,
        continuation: str,
        tokenizer: interfaces.TokenGeneratorTokenizer,
        max_length: int,
    ):
        context = context or tokenizer.eos

        # Move all right spaces from the context to the continuation.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # To avoid the beginning-of-text tokens in the continuation encoding,
        # the context and continuation should be encoded together.
        # For example, when encoded separately:
        #    "my dog is" =  [128000, 2465, 5679, 374]
        #    " cute" = [128000, 19369]
        # But together, this is encoded as [128000, 2465, 5679, 374, 19369]
        encoded_text = await tokenizer.encode(context + continuation)
        encoded_context = await tokenizer.encode(context)
        encoded_continuation = encoded_text[len(encoded_context) :]
        if len(encoded_continuation) != 1:
            raise ValueError(
                "Evaluations do not support continuations with more than 1"
                f" token. Got {continuation}, which is encoded as"
                f" {encoded_continuation}."
            )

        if len(encoded_context) > max_length:
            encoded_context = encoded_context[-max_length:]
            truncated_context = await tokenizer.decode(encoded_context)
        else:
            truncated_context = context

        return cls(
            context=context,
            continuation=continuation,
            encoded_context=encoded_context,
            truncated_context=truncated_context,
            encoded_continuation=encoded_continuation[0],
        )


class PipelineLM(LM):
    model: Any  # TODO(kathywu): Update to PipelineModel
    generator: interfaces.TokenGenerator
    tokenizer: interfaces.TokenGeneratorTokenizer

    def __init__(self, max_pipeline_and_tokenizer: MaxPipelineAndTokenizer):
        self.model = max_pipeline_and_tokenizer.model
        self.generator = max_pipeline_and_tokenizer.generator
        self.tokenizer = max_pipeline_and_tokenizer.tokenizer
        self.config = self.model.config
        self.max_batch_size = self.config.max_cache_batch_size
        self.max_length = self.config.huggingface_config.max_seq_len
        super().__init__()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        """
        return asyncio.run(self.tokenizer.encode(string))

    def tok_decode(self, encoded: np.ndarray, **kwargs) -> list[int]:
        """
        Tokenize a string using the model's tokenizer and return a list of token IDs.
        """
        return self.tokenizer.delegate.decode(encoded)

    def loglikelihood(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        async def read_requests():
            examples = []
            for context, continuation in [req.args for req in requests]:
                examples.append(
                    await Example.from_request(
                        context, continuation, self.tokenizer, self.max_length
                    )
                )
            return examples

        examples = asyncio.run(read_requests())
        return self._loglikelihood_tokens(examples)

    def _loglikelihood_tokens(
        self,
        requests: list[Example],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        # Reorder requests by length, then group into batches <  max batch size.

        def _collate(x):
            """Defines the key for the sorted method"""
            toks = x.encoded_context + [x.encoded_continuation]
            return -len(toks), tuple(toks)

        reorderer = utils.Reorderer(requests, _collate)

        example_batches = list(
            lm_eval.models.utils.chunks(
                reorderer.get_reordered(), self.max_batch_size
            )
        )
        responses = []
        for batch in tqdm(example_batches, disable=disable_tqdm):
            # Each batch contains <= self.max_batch_size examples.
            prompts = [x.truncated_context for x in batch]
            results = evaluate.run_model(
                self.model, self.tokenizer, prompts, num_steps=1
            )
            for result, example in zip(results, batch):
                # Compute the log softmax of the logits. This is what the
                # Huggingface LM model does.
                # https://github.com/EleutherAI/lm-evaluation-harness/blob/4155ec7fb9695d73d8301bc0323972fcb0f0d31e/lm_eval/models/huggingface.py#L1140
                logits = log_softmax(result["values"][0]["logits"])
                greedy_token = logits.argmax()
                max_equal = greedy_token == example.encoded_continuation

                # Answer: (log prob, is-exact-match)
                answer = (logits[example.encoded_continuation], max_equal)
                responses.append(answer)

                self.cache_hook.add_partial(
                    "loglikelihood", example.cache_key, answer
                )

        return reorderer.get_original(responses)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """For perplexity computation (how well a model has learned the training
        set).
        """
        # This is not implemented because:
        # - Requires logits for ALL tokens provided in the input. We currently
        #   only return the logits for the last token.
        # - Most tasks use `loglikelihood`.
        raise NotImplementedError("Rolling loglikelihood is not implemented.")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        # TODO: Implement this
        raise NotImplementedError("Not sure if needed.")


@click.command()
@click.option(
    "-o",
    "--output",
    "output_path",
    type=str,
    required=False,
    default=None,
    help="Path to save the results.",
)
@click.option(
    "--log_samples",
    "log_samples",
    type=bool,
    default=False,
    is_flag=True,
    help="Whether to log the samples",
)
def main(output_path: str, log_samples: bool):
    if workspace_dir := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace_dir)

    # TODO: Move hardcoded params to click arguments
    pipeline_name = "llama"
    version_name = "llama3_1"
    encoding_name = "bfloat16"
    device_specs = [driver.DeviceSpec.cuda()]

    pipeline_oracle = PIPELINE_ORACLES[pipeline_name]
    max_pipeline_and_tokenizer = pipeline_oracle.create_max_pipeline(
        version=version_name,
        encoding=encoding_name,
        device_specs=device_specs,
    )
    lm_model = PipelineLM(max_pipeline_and_tokenizer)

    results = simple_evaluate(
        model=lm_model,
        tasks=["leaderboard_bbh_boolean_expressions"],
        num_fewshot=0,
        log_samples=log_samples,
    )

    evaluation_tracker = EvaluationTracker(output_path=output_path)
    if results is not None:
        samples = results.pop("samples") if log_samples else None
        evaluation_tracker.general_config_tracker.model_name_sanitized = (
            "max-llama3.1"
        )
        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples
        )
        if log_samples:
            for task_name in results["configs"]:
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )
        print(utils.make_table(results))
        if "groups" in results:
            print(utils.make_table(results, "groups"))


if __name__ == "__main__":
    main()
