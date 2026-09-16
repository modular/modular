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

"""Per-request RNG seeds for token sampling."""

from __future__ import annotations

import functools
import hashlib

from max.pipelines.context import TextContext

_U64_MASK = (1 << 64) - 1


@functools.lru_cache(maxsize=8192)
def request_id_key(request_id: str) -> int:
    """Returns the 64-bit seed contribution of ``request_id``.

    BLAKE2b rather than :func:`hash`: the builtin's string hash is salted per
    process unless ``PYTHONHASHSEED`` is pinned, so keying off it would give a
    seeded run different output on every restart. Request ids are usually
    UUID4 hex but callers may supply any string, so the id is hashed rather
    than parsed.
    """
    digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def request_row_seed(context: TextContext) -> int:
    """Returns the Philox key for ``context``'s row of the sampling batch.

    The key is the request's own seed, advanced by its generated-token count
    so consecutive decoding steps draw fresh randomness, mixed with a stable
    hash of its request id.

    The id is what ties the key to the request rather than to the batch slot
    it happens to occupy. The sampling kernel keys its RNG on this value
    alone, so a request that is preempted and re-admitted into a different
    slot still draws the token its seed called for; the id survives that round
    trip because ``TextBatchConstructor._preempt_request`` returns the same
    context object to the queue. It is also what keeps two co-resident
    requests that were handed the same ``sampling_params.seed`` from drawing
    in lock step.

    Both the ordinary decode sampler and the speculative-decoding mirror in
    ``overlap_text_generation.py`` build their per-row seeds from here. A row
    whose key disagreed between the two would change token mid-stream when a
    request moved between those paths.
    """
    return (
        context.sampling_params.seed
        + len(context.tokens)
        + request_id_key(context.request_id.value)
    ) & _U64_MASK
