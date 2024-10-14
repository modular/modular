# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import threading
from dataclasses import dataclass
from time import sleep, time
from typing import Literal, Optional

from transformers import AutoTokenizer


@dataclass
class PerformanceFakingContext:
    # simulation attributes
    prompt_len: int
    context_len: int
    max_tokens: int
    # correctness attributes
    prompt: str


@dataclass
class PerformanceFakingTokenGenerator:
    _tokenizer: Optional[AutoTokenizer] = None

    # ttft (ms) for prompt_length = batch_size = 1
    ce_baseline: float = 6.85
    # ttft (ms) / batch size / prompt size
    ce_rate: float = 54043.08 / 1024 / 1024
    # padding mode for context encoding
    ce_padding: bool = False

    # 1st token TPOT (ms) for prompt_length = batch_size = 1
    tg_baseline: float = 6.85
    # 1st token TPOT (ms) / batch_size with prompt_length = 1
    tg_rate_no_context: float = 12.67 / 256
    # 1st token TPOT (ms) / batch_size with prompt_length = 512
    tg_rate_per_context_token: float = (21.11 / 256 - 12.67 / 256) / 512
    # padding mode for context encoding
    tg_padding: bool = False

    # our pipelines are well behaved asyncio citizen and offload
    # expensive work to the asyncio threadpool
    use_executor: bool = True

    # whether to busy wait or to sleep
    # the model execute call in our pipelines does release the GIL
    busy_wait: bool = False

    # TODO@gaz: We can't use __class__ here because this is currently setup as a dataclass
    logger: logging.Logger = logging.getLogger(__name__)

    # lock to prevent concurrent usage of the fake GPU
    wait_lock = threading.Lock()
    # amount of time waited in the fake GPU
    wait_secs = 0
    # number of times waited in the fake GPU
    wait_count = 0
    # amount of time spent in the tokenizer
    tokenizer_secs = 0
    # timestamp of the end of the last GPU wait
    last_wait_end = None
    # amount of time spent in between waiting
    non_wait_secs = 0

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> PerformanceFakingContext:
        if self.last_wait_end is None:
            self.last_wait_end = time()

        if self._tokenizer:
            if self.use_executor:
                loop = asyncio.get_running_loop()
                encoded = await loop.run_in_executor(
                    None, self._tokenize, prompt
                )
            else:
                encoded = self._tokenize(prompt)
            prompt_length = len(encoded)
        else:
            prompt_length = len(prompt)
        num_tokens = max_new_tokens or prompt_length
        return PerformanceFakingContext(prompt_length, 0, num_tokens, prompt)

    async def next_token(
        self, batch: dict[str, PerformanceFakingContext]
    ) -> dict[str, str]:
        context_lengths = [x.context_len for x in batch.values()]
        if sum(context_lengths) == 0:
            self.logger.info(
                f"PerformanceFake: CE with batch_size = {len(batch)}"
            )
            # context encoding mode
            wait_time = self._ce_time_ms([x.prompt_len for x in batch.values()])
            for _, ctx in batch.items():
                ctx.context_len += ctx.prompt_len
        else:
            # token generation mode
            self.logger.info(
                f"PerformanceFake: TG with batch_size = {len(batch)}"
            )
            wait_time = self._tg_time_ms(
                [x.context_len for x in batch.values()]
            )
            for _, ctx in batch.items():
                ctx.context_len += 1

        # actually wait here
        if self.use_executor:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._wait, wait_time)
        else:
            self._wait(wait_time)

        # We return the reversed prompt, repeated as many times necessary
        # to satisfy the max_tokens
        return {
            rid: ctx.prompt[-((ctx.context_len + 1) % ctx.prompt_len)]
            for rid, ctx in batch.items()
            if ctx.context_len - ctx.prompt_len < ctx.max_tokens
        }

    async def release(self, context: PerformanceFakingContext):
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

    def _tokenize(self, prompt):
        if self._tokenizer is None:
            return prompt
        else:
            start = time()
            encoded = self._tokenizer.encode(prompt)
            self.tokenizer_secs += time() - start
            return encoded

    def __del__(self):
        # print the total wait time for benchmarking/debugging purposes
        self.logger.info(
            f"PerformanceFake: waited {self.wait_count} times for"
            f" {self.wait_secs:.4f} sec total"
        )
        self.logger.info(
            "PerformanceFake: tokenized for"
            f" {self.tokenizer_secs:.4f} sec total"
        )
        self.logger.info(
            "PerformanceFake: not waiting for"
            f" {self.non_wait_secs:.4f} sec total"
        )


def get_performance_fake(
    mode: Literal["no-op", "speed-of-light", "vllm"],
    tokenizer: Optional[AutoTokenizer] = None,
) -> PerformanceFakingTokenGenerator:
    """Construct a performance fake for the given performance mode."""
    if mode == "no-op":
        return PerformanceFakingTokenGenerator(
            tokenizer, 0, 0, False, 0, 0, 0, False, False
        )
    elif mode == "speed-of-light":
        # current defaults are speed-of-light on A100-80GB
        return PerformanceFakingTokenGenerator(tokenizer)
    elif mode == "vllm":
        # this is for A100-80GB
        return PerformanceFakingTokenGenerator(
            _tokenizer=tokenizer,
            ce_baseline=11.95,
            ce_rate=19487 / 1024 / 256,
            ce_padding=False,
            tg_baseline=11.95,
            tg_rate_no_context=33.66 / 256,
            tg_rate_per_context_token=(59.79 - 33.66) / 256 / 1024,
            tg_padding=False,
        )
    else:
        raise ValueError(f"Unexpected mode: {mode}")
