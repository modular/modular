# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from time import sleep, time
from typing import List, Literal, Optional

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
    tokenizer: Optional[AutoTokenizer] = None

    # ttft (ms) for prompt_length = batch_size = 1
    ce_baseline: float = 6.85
    # ttft (ms) / batch size / prompt size
    ce_rate: float = 54043.08 / 1024 / 1024
    # padding mode for context encoding
    ce_padding: bool = True

    # 1st token TPOT (ms) for prompt_length = batch_size = 1
    tg_baseline: float = 6.85
    # 1st token TPOT (ms) / batch_size with prompt_length = 1
    tg_rate_no_context: float = 12.67 / 256
    # 1st token TPOT (ms) / batch_size with prompt_length = 512
    tg_rate_256_context: float = 21.11 / 256
    # padding mode for context encoding
    tg_padding: bool = True

    # whether to busy wait or to sleep
    busy_wait: bool = False

    async def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> PerformanceFakingContext:
        if self.tokenizer:
            prompt_length = len(self.tokenizer.encode(prompt))
        else:
            prompt_length = len(prompt)
        num_tokens = max_new_tokens or prompt_length
        return PerformanceFakingContext(prompt_length, 0, num_tokens, prompt)

    async def next_token(
        self, batch: dict[str, PerformanceFakingContext]
    ) -> dict[str, str]:
        context_lengths = [x.context_len for x in batch.values()]
        if sum(context_lengths) == 0:
            # context encoding mode
            wait_time = self._ce_time_ms([x.prompt_len for x in batch.values()])
            for _, ctx in batch.items():
                ctx.context_len += ctx.prompt_len
        else:
            # token generation mode
            wait_time = self._tg_time_ms(
                [x.context_len for x in batch.values()]
            )
            for _, ctx in batch.items():
                ctx.context_len += 1

        # actually wait here
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
        if self.busy_wait:
            start = time()
            while (time() - start) * 1000 < wait_time_ms:
                pass
        else:
            sleep(wait_time_ms * 0.001)

    def _ce_time_ms(self, prompt_sizes):
        if self.ce_padding:
            N = len(prompt_sizes) * max(prompt_sizes)
        else:
            N = sum(prompt_sizes)
        return max(self.ce_rate * N, self.ce_baseline)

    def _tg_time_ms(self, context_sizes):
        context_term = (
            self.tg_rate_256_context - self.tg_rate_no_context
        ) / 512

        if self.tg_padding:
            N = len(context_sizes) * max(context_sizes)
        else:
            N = sum(context_sizes)

        return (
            max(self.tg_baseline, self.tg_rate_no_context * len(context_sizes))
            + N * context_term
        )


def get_performance_fake(
    tokenizer: AutoTokenizer, mode: Literal["no-op", "speed-of-light"]
) -> PerformanceFakingTokenGenerator:
    """Construct a performance fake for the given performance mode."""
    if mode == "no-op":
        return PerformanceFakingTokenGenerator(
            tokenizer, 0, 0, False, 0, 0, 0, False, False
        )
    elif mode == "speed-of-light":
        # current defaults are speed-of-light
        return PerformanceFakingTokenGenerator(tokenizer)
    else:
        raise ValueError(f"Unexpected mode: {mode}")
