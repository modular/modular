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

"""Context-managed spawn process pool for slow tokenizer work.

Some benchmark tokenizers (Kimi K2.5) are pure-Python and hold the GIL on
every `encode`/`decode` call. The dataset-generation code paths perform many
thousands of these calls per benchmark, so we offload them to a process pool.
Workers re-load the tokenizer once each via `_init_encoder` and then service
encode/decode/session-build tasks for the lifetime of the `TokenizerPool`.

Use as a context manager so the pool is closed and joined deterministically
on exit. Worker processes are created lazily on the first encode/decode/map
call, so constructing a `TokenizerPool` you never use (e.g. when a dry-run
test mocks out the dataset method) is free.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from typing import TypeVar

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_T = TypeVar("_T")
_R = TypeVar("_R")

# Per-worker tokenizer instance, populated by `_init_encoder` when each
# spawned worker process starts. Reads via `worker_tokenizer()` from inside
# worker callables (e.g. the multiturn session builder).
_WORKER_TOK: PreTrainedTokenizerBase | None = None


def _default_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    """Production worker loader: re-loads the real tokenizer by name."""
    from max.benchmark.benchmark_shared.utils import get_tokenizer

    return get_tokenizer(
        name_or_path,
        model_max_length=model_max_length,
        trust_remote_code=trust_remote_code,
    )


_LoaderFn = Callable[[str, "int | None", bool], PreTrainedTokenizerBase]


def _init_encoder(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    loader: _LoaderFn,
) -> None:
    """Worker initializer: build a tokenizer and stash it for the worker."""
    global _WORKER_TOK
    _WORKER_TOK = loader(name_or_path, model_max_length, trust_remote_code)


def _encode_len(text: str) -> int:
    assert _WORKER_TOK is not None
    return len(_WORKER_TOK.encode(text, add_special_tokens=False))


def _decode_text(ids: list[int]) -> str:
    assert _WORKER_TOK is not None
    return _WORKER_TOK.decode(ids, skip_special_tokens=False)


def worker_tokenizer() -> PreTrainedTokenizerBase:
    """Return the tokenizer loaded inside the current worker process.

    Worker callables that need direct tokenizer access (e.g. the multiturn
    session builder) call this to read the worker-local tokenizer instance.
    """
    assert _WORKER_TOK is not None
    return _WORKER_TOK


class TokenizerPool:
    """Spawn process pool bound to a specific tokenizer.

    Use as a context manager so the pool is closed cleanly on exit:

        with TokenizerPool(tokenizer) as pool:
            lens = pool.encode_lens(texts)
            sessions = pool.map(_build_session, args_list)

    The tokenizer is captured at construction; mismatch is structurally
    impossible because callers receive only the pool and read
    `pool.tokenizer` for parent-side work.

    Worker processes are spawned lazily on the first encode/decode/map
    call. Constructing a pool you never use is free.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        workers: int | None = None,
        loader: _LoaderFn | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        nproc = workers if workers is not None else max(1, os.cpu_count() or 8)
        # Tests (and other constrained environments) can cap worker count
        # via env var without threading `workers=` through every call site.
        env_cap = os.environ.get("MAX_BENCHMARK_MAX_TOKENIZER_THREADS")
        if env_cap:
            try:
                nproc = min(nproc, max(1, int(env_cap)))
            except ValueError:
                pass
        self._workers = nproc
        self._loader = loader if loader is not None else _default_loader
        self._pool: mp.pool.Pool | None = None

    def __enter__(self) -> TokenizerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the pool and wait for workers to exit cleanly."""
        if self._pool is None:
            return
        # `close()` stops the pool from accepting new tasks; `join()` waits
        # for in-flight tasks to drain and workers to exit. Unlike
        # `terminate()`, this gives workers a chance to release any
        # Rayon/transformers C-extension state without leaking.
        self._pool.close()
        self._pool.join()
        self._pool = None

    def _ensure_pool(self) -> mp.pool.Pool:
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self._workers,
                initializer=_init_encoder,
                initargs=(
                    self.tokenizer.name_or_path,
                    self.tokenizer.model_max_length,
                    True,
                    self._loader,
                ),
            )
        return self._pool

    def _chunksize(self, n: int) -> int:
        return max(1, n // (self._workers * 4))

    def encode_lens(self, texts: list[str]) -> list[int]:
        """Encode each text and return only its token count.

        Workload is fanned out across the pool's workers.
        """
        return self.map(_encode_len, texts)

    def decode_texts(self, id_lists: list[list[int]]) -> list[str]:
        """Decode each token-id list to text.

        Workload is fanned out across the pool's workers.
        """
        return self.map(_decode_text, id_lists)

    def map(self, fn: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
        """Run `fn` across the pool for every element of `items`.

        Used by coarser per-task workers (e.g. session builders) that
        access their tokenizer via `worker_tokenizer()`.
        """
        items_list = list(items)
        if not items_list:
            return []
        pool = self._ensure_pool()
        return pool.map(
            fn, items_list, chunksize=self._chunksize(len(items_list))
        )
