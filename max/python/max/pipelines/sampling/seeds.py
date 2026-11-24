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

from max.pipelines.context import TextContext

_U64_MASK = (1 << 64) - 1


def request_row_seed(context: TextContext) -> int:
    """Returns the Philox key for ``context``'s row of the sampling batch.

    The key is the request's own seed, advanced by its generated-token count
    so consecutive decoding steps draw fresh randomness. Both terms belong to
    the request rather than to the batch slot it occupies, which is what lets
    a request that is preempted and re-admitted into a different slot carry
    on drawing the tokens its seed called for.

    Nothing identifying the request is mixed in, and that is deliberate. Two
    requests handed the same ``sampling_params.seed`` that have generated the
    same number of tokens draw in lock step, which is what the seed means: a
    client that pins one is asking to be able to reproduce a result. Salting
    the key per request would decorrelate co-resident rows, but a
    server-minted id in the key makes a pinned seed unreproducible even
    against itself, so the seed has to be the whole of it.

    Both the ordinary decode sampler and the speculative-decoding mirror in
    ``overlap_text_generation.py`` build their per-row seeds from here. A row
    whose key disagreed between the two would change token mid-stream when a
    request moved between those paths.
    """
    return (context.sampling_params.seed + len(context.tokens)) & _U64_MASK
