# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Shared helpers for xgrammar token-mask cost and exactness tests."""

from __future__ import annotations

import itertools
import string
import time
from collections.abc import Callable, Sequence

import numpy as np
from max import _xgrammar as xgr

# One fill per generated token; a fill over this budget is visible as decode
# latency.
BUDGET_S = 1e-3

EOS = "<eos>"


def synthetic_vocab(markers: Sequence[str], size: int) -> list[str]:
    """Builds a RAW vocabulary that holds each marker as one token.

    About half the tokens start with a space, like a BPE vocabulary. Some
    parser states leave every space-led token uncertain, which is what makes
    a fill cost scale with the vocabulary.
    """
    alphanum = string.ascii_letters + string.digits
    vocab = [EOS, *markers, *string.printable]
    for n in (2, 3):
        for chars in itertools.product(alphanum, repeat=n):
            word = "".join(chars)
            vocab += [word, " " + word]
            if len(vocab) >= size:
                return vocab[:size]
    return vocab


def encode(text: str, ids: dict[str, int], markers: Sequence[str]) -> list[int]:
    """Encodes text greedily: a marker, else the longest token of 3 to 1."""
    out, i = [], 0
    while i < len(text):
        word = next((m for m in markers if text.startswith(m, i)), None)
        if word is None:
            word = next(
                text[i : i + n] for n in (3, 2, 1) if text[i : i + n] in ids
            )
        out.append(ids[word])
        i += len(word)
    return out


def cpu_seconds_per_call(fn: Callable[[], object]) -> float:
    """Returns the CPU seconds of one call to fn.

    Takes the best of several samples, so host scheduling noise cannot
    inflate a measurement.
    """
    best = float("inf")
    for _ in range(5):
        start = time.process_time()
        for _ in range(10):
            fn()
        best = min(best, time.process_time() - start)
    return best / 10


def fill_costs_along(
    matcher: xgr.GrammarMatcher, vocab_size: int, tokens: Sequence[int]
) -> list[float]:
    """Returns the CPU seconds of one fill at each position along tokens.

    The matcher accepts each token after its position is timed, and is left
    after the last one.
    """
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    costs = []
    for i in range(len(tokens) + 1):
        costs.append(
            cpu_seconds_per_call(
                lambda: matcher.fill_next_token_bitmask(bitmask, 0)
            )
        )
        if i < len(tokens):
            assert matcher.accept_token(tokens[i]), f"rejected at {i}"
    return costs


def assert_mask_is_exact(
    matcher: xgr.GrammarMatcher, vocab_size: int, where: str = ""
) -> None:
    """Asserts that the filled mask equals token-by-token acceptance."""
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(bitmask, 0)
    bits = np.unpackbits(bitmask[0].view(np.uint8), bitorder="little")
    filled = set(np.flatnonzero(bits[:vocab_size]).tolist())
    accepted = set()
    for token in range(vocab_size):
        if matcher.accept_token(token):
            accepted.add(token)
            matcher.rollback(1)
    assert filled == accepted, f"mask differs {where}"
