# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import dataclasses
import json
import logging
import threading
from os import environ
from time import sleep, time
from typing import Any, Callable, Iterable, List, Literal, Optional, Union

import numpy as np
from max.pipelines.interfaces import TokenGenerator, TokenGeneratorRequest
from max.pipelines.response import TextResponse
from max.pipelines.tokenizer import PreTrainedPipelineTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclasses.dataclass
class PerformanceFakingContext:
    # simulation attributes
    prompt_len: int
    context_len: int
    seq_len: int
    max_tokens: int

    # correctness attributes
    prompt: str
    encoded_prompt: np.ndarray


@dataclasses.dataclass
class BatchInfo:
    """Information about a batch of requests passed to the pipeline"""

    past_seq_lens: List[int]
    """Coordinated list of past sequence lengths (i.e. context lengths)"""

    seq_lens: List[int]
    """Coordinated list of sequence lengths, i.e. prompt_len or 1"""

    num_steps: int
    """Number of steps to do in the pipeline"""


class PerformanceFakingPipelineTokenizer(
    PreTrainedPipelineTokenizer[PerformanceFakingContext]
):
    def __init__(
        self, delegate: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    ):
        super().__init__(delegate)
        self.logger = logging.getLogger(self.__class__.__name__)
        # amount of time spent in the tokenizer
        self.tokenizer_secs = 0.0

    async def new_context(
        self, request: TokenGeneratorRequest
    ) -> PerformanceFakingContext:
        prompt: str
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")
        encoded_prompt = await self.encode(prompt)
        prompt_length = len(encoded_prompt)
        num_tokens = request.max_new_tokens or prompt_length
        return PerformanceFakingContext(
            prompt_length,
            0,
            prompt_length,
            num_tokens,
            prompt,
            np.array(encoded_prompt),
        )

    async def encode(self, prompt: str) -> np.ndarray:
        start = time()
        encoded = await super().encode(prompt)
        self.tokenizer_secs += time() - start
        return encoded

    async def decode(
        self,
        context: PerformanceFakingContext,
        logits: np.ndarray,
    ) -> str:
        # Not actually an np.ndarray!
        return logits  # type: ignore

    def __del__(self):
        self.logger.info(
            "PerformanceFake: tokenized for"
            f" {self.tokenizer_secs:.4f} sec total"
        )


class PerformanceFakingTokenGenerator(TokenGenerator[PerformanceFakingContext]):
    def __init__(
        self,
        ce_baseline: float,
        ce_rate: float,
        ce_padding: bool,
        tg_baseline: float,
        tg_rate_no_context: float,
        tg_rate_per_context_token: float,
        tg_padding: bool,
        busy_wait: bool,
        failure_predicate: Optional[Callable[[], bool]] = None,
    ):
        """Parameterize a performance fake

        Args:
            ce_baseline: The minimum amount of time context encoding can take
            ce_rate: The context encoding time per prompt token
            ce_padding: Whether or not prompts are padding to equal length
            tg_baseline: The minimum amount of time token generation can take
            tg_rate_no_context: The token generation time per request with no context
            tg_rate_per_context_token: The additional token generation time per context token
            tg_padding: Whether or not contexts are padded to the same length
            busy_wait: Whether to wait on the fake GPU as a busy-loop, vs sleep
        """
        super().__init__()
        self.ce_baseline = ce_baseline
        self.ce_rate = ce_rate
        self.ce_padding = ce_padding
        self.tg_baseline = tg_baseline
        self.tg_rate_no_context = tg_rate_no_context
        self.tg_rate_per_context_token = tg_rate_per_context_token
        self.tg_padding = tg_padding
        self.busy_wait = busy_wait
        self.failure_predicate = failure_predicate

        self.batch_info_output_fname: Optional[str] = environ.get(
            "MAX_BATCH_INFO_FILENAME"
        )

        self.logger: logging.Logger = logging.getLogger(__name__)

        # lock to prevent concurrent usage of the fake GPU
        self.wait_lock = threading.Lock()
        # amount of time waited in the fake GPU
        self.wait_secs = 0
        # number of times waited in the fake GPU
        self.wait_count = 0
        # timestamp of the end of the last GPU wait
        self.last_wait_end: Optional[float] = None
        # amount of time spent in between waiting
        self.non_wait_secs: float = 0
        # record of batches
        self.batch_infos: List[BatchInfo] = []

    def next_token(
        self, batch: dict[str, PerformanceFakingContext], num_steps: int = 1
    ) -> list[dict[str, Any]]:
        if self.batch_info_output_fname is not None:
            self._record_batch_info(batch.values(), num_steps)

        return [self.step(batch) for _ in range(num_steps)]

    def step(self, batch: dict[str, PerformanceFakingContext]):
        context_lengths = [x.context_len for x in batch.values()]
        if sum(context_lengths) == 0:
            self.logger.debug(
                f"PerformanceFake: CE with batch_size = {len(batch)}"
            )
            # context encoding mode
            wait_time = self._ce_time_ms([x.prompt_len for x in batch.values()])
            for _, ctx in batch.items():
                if self.failure_predicate and self.failure_predicate():
                    raise Exception("performance fake simulated failure in CE")
                ctx.context_len += ctx.prompt_len
                ctx.seq_len = 1
        else:
            # token generation mode
            self.logger.debug(
                f"PerformanceFake: TG with batch_size = {len(batch)}"
            )
            wait_time = self._tg_time_ms(
                [x.context_len for x in batch.values()]
            )
            for _, ctx in batch.items():
                if self.failure_predicate and self.failure_predicate():
                    raise Exception("performance fake simulated failure in TG")
                ctx.context_len += 1

        # actually wait here
        self._wait(wait_time)

        # We return the reversed prompt, repeated as many times necessary
        # to satisfy the max_tokens
        return {
            rid: TextResponse(
                next_token=ctx.prompt[-((ctx.context_len + 1) % ctx.prompt_len)]
            )
            for rid, ctx in batch.items()
            if ctx.context_len - ctx.prompt_len < ctx.max_tokens
        }

    def release(self, context: PerformanceFakingContext):
        pass

    def _wait(self, wait_time_ms):
        self.logger.debug(f"PerformanceFake: waiting {wait_time_ms} ms")
        self.wait_secs += wait_time_ms * 0.001
        self.wait_count += 1
        with self.wait_lock:
            start = time()
            if self.last_wait_end is not None:
                self.logger.debug(
                    "PerformanceFake: waiting after"
                    f" {start - self.last_wait_end:.4f} sec"
                )
                self.non_wait_secs += start - self.last_wait_end
            if self.busy_wait:
                while (time() - start) * 1000 < wait_time_ms:
                    pass
            else:
                sleep(wait_time_ms * 0.001)
            self.last_wait_end = time()

    def _ce_time_ms(self, prompt_sizes):
        if self.ce_padding:
            N = len(prompt_sizes) * max(prompt_sizes)
        else:
            N = sum(prompt_sizes)
        return max(self.ce_rate * N, self.ce_baseline)

    def _tg_time_ms(self, context_sizes):
        if self.tg_padding:
            N = len(context_sizes) * max(context_sizes)
        else:
            N = sum(context_sizes)

        return (
            max(self.tg_baseline, self.tg_rate_no_context * len(context_sizes))
            + N * self.tg_rate_per_context_token
        )

    def _record_batch_info(
        self, contexts: Iterable[PerformanceFakingContext], num_steps: int
    ):
        self.batch_infos.append(
            BatchInfo(
                past_seq_lens=[x.context_len for x in contexts],
                seq_lens=[x.seq_len for x in contexts],
                num_steps=num_steps,
            )
        )

    def __del__(self):
        # print the total wait time for benchmarking/debugging purposes
        self.logger.info(
            f"PerformanceFake: waited {self.wait_count} times for"
            f" {self.wait_secs:.4f} sec total"
        )
        self.logger.info(
            "PerformanceFake: not waiting for"
            f" {self.non_wait_secs:.4f} sec total"
        )
        if self.batch_info_output_fname is not None:
            output = {
                "batch_data": [dataclasses.asdict(x) for x in self.batch_infos]
            }
            with open(self.batch_info_output_fname, "w") as f:
                json.dump(output, f, indent=2)


def get_performance_fake(
    mode: Literal["no-op", "speed-of-light", "vllm"],
    failure_predicate: Optional[Callable[[], bool]] = None,
) -> PerformanceFakingTokenGenerator:
    """Construct a performance fake for the given performance mode."""
    if mode == "no-op":
        return PerformanceFakingTokenGenerator(
            ce_baseline=0,
            ce_rate=0,
            ce_padding=False,
            tg_baseline=0,
            tg_rate_no_context=0,
            tg_rate_per_context_token=0,
            tg_padding=False,
            busy_wait=False,
            failure_predicate=failure_predicate,
        )
    elif mode == "speed-of-light":
        # current defaults are speed-of-light on A100-80GB
        return PerformanceFakingTokenGenerator(
            ce_baseline=6.85,
            ce_rate=54043.08 / 1024 / 1024,
            ce_padding=False,
            tg_baseline=6.85,
            tg_rate_no_context=12.67 / 256,
            tg_rate_per_context_token=(21.11 / 256 - 12.67 / 256) / 512,
            tg_padding=False,
            busy_wait=False,
            failure_predicate=failure_predicate,
        )
    elif mode == "vllm":
        # this is for A100-80GB
        return PerformanceFakingTokenGenerator(
            ce_baseline=11.95,
            ce_rate=19487 / 1024 / 256,
            ce_padding=False,
            tg_baseline=11.95,
            tg_rate_no_context=33.66 / 256,
            tg_rate_per_context_token=(59.79 - 33.66) / 256 / 1024,
            tg_padding=False,
            busy_wait=False,
            failure_predicate=failure_predicate,
        )
    else:
        raise ValueError(f"Unexpected mode: {mode}")
