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

"""A request's block hashes must not depend on how it grew.

A request's hash chain is extended as it decodes past block boundaries or
prefills in chunks. The keys that come out must equal what hashing the whole
sequence in one call produces: they are the prefix-cache and external-tier keys,
so any drift is a silent cache miss.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from max.nn.kv_cache import KVCacheGroupId, PagedKVLeafRegion
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.paged_kv_cache.block_manager import BlockManager
from max.pipelines.kv_cache.paged_kv_cache.block_utils import (
    KVHashAlgo,
    hash_request_tokens,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_manager import (
    JengaBlockManager,
    KVLeafInfo,
    create_groups,
    create_pools,
)
from max.pipelines.modeling.types import RequestID

BLOCK_SIZE = 4
SEED = bytes(range(32))

# (algo, seed, cache_salt): every algo, seeded and unseeded, with and without a
# per-request salt.
HASH_CONFIGS = [
    pytest.param(algo, seed, salt, id=f"{algo}-{seed_id}-{salt_id}")
    for algo in ("ahash64", "sha256")
    for seed, seed_id in ((None, "unseeded"), (SEED, "seeded"))
    for salt, salt_id in ((None, "nosalt"), ("tenant-A", "salt"))
]


def one_shot_hashes(
    tokens: np.ndarray,
    algo: KVHashAlgo,
    seed: bytes | None,
    salt: str | None,
) -> list[bytes]:
    """Hashes every full block of ``tokens[:-1]`` in a single call.

    The last token is never hashed: a request that hit the cache on all of its
    tokens would have none left to run.
    """
    return hash_request_tokens(
        tokens[:-1].copy(), BLOCK_SIZE, algo=algo, seed=seed, salt=salt
    )


@pytest.mark.parametrize(("algo", "seed", "salt"), HASH_CONFIGS)
def test_block_manager_hashes_match_one_shot_as_the_chain_grows(
    algo: KVHashAlgo, seed: bytes | None, salt: str | None
) -> None:
    """Feeds a BlockManager a request that gains tokens between hash calls.

    The sizes stay under one block, land one token past a boundary, skip
    several blocks at once (chunked prefill), and stop mid-block. Every
    intermediate chain is checked against a one-shot hash.
    """
    bm = BlockManager(
        total_num_blocks=32,
        block_size=BLOCK_SIZE,
        connector=NullConnector(),
        enable_prefix_caching=True,
        kv_hash_algo=algo,
        kv_hash_seed=seed,
    )
    all_tokens = np.arange(100, 140, dtype=np.int32)
    request_id = RequestID("req-grow")
    for num_tokens in (3, 5, 6, 9, 10, 22, 23, 40):
        ctx = cast(
            TextContext,
            SimpleNamespace(
                request_id=request_id,
                tokens=all_tokens[:num_tokens],
                cache_salt=salt,
            ),
        )
        bm.compute_hashes_for_request(ctx)
        assert bm.req_to_hashes[request_id] == one_shot_hashes(
            all_tokens[:num_tokens], algo, seed, salt
        ), f"diverged after {num_tokens} tokens"
    assert len(bm.req_to_hashes[request_id]) == 9


@pytest.mark.parametrize(("algo", "seed", "salt"), HASH_CONFIGS)
def test_jenga_hashes_match_one_shot_as_the_chain_grows(
    algo: KVHashAlgo, seed: bytes | None, salt: str | None
) -> None:
    leaf_id = "full"
    leaf_infos = {leaf_id: KVLeafInfo(1, KVCacheGroupId.full())}
    pools = create_pools(leaf_infos, 32, 1)
    bm = JengaBlockManager(
        pools=pools,
        groups=create_groups(
            leaf_infos, pools, BLOCK_SIZE, enable_prefix_caching=True
        ),
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        kv_hash_algo=algo,
        kv_hash_seed=seed,
        leaves={
            leaf_id: PagedKVLeafRegion(
                leaf_id=leaf_id,
                group_id=KVCacheGroupId.full(),
                bytes_per_page=1,
                page_size=BLOCK_SIZE,
            )
        },
    )

    def make_ctx(tokens: np.ndarray) -> TextContext:
        return TextContext(
            request_id=RequestID(),
            max_length=4096,
            tokens=TokenBuffer(tokens.astype(np.int64)),
            cache_salt=salt,
        )

    # Decode across several block boundaries, committing a block's hashes the
    # step after it fills.
    prompt = np.arange(100, 106, dtype=np.int64)
    ctx = make_ctx(prompt)
    bm.claim(ctx)
    generated = list(range(1000, 1019))
    for token in generated:
        bm.alloc(ctx)
        ctx.update(token)
        bm.step(ctx)
    bm.release(ctx)

    # A request with the same tokens hashes them in one call. It finds every
    # block the decoding request committed only if the two sets of keys agree.
    full = np.concatenate([prompt, np.array(generated, dtype=np.int64)])
    num_hashable_blocks = (len(full) - 1) // BLOCK_SIZE
    assert num_hashable_blocks >= 5
    hits = bm.get_prefix_cache_hit_counts(make_ctx(full))
    assert hits[0].device_blocks == num_hashable_blocks
