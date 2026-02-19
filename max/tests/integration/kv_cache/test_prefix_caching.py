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

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.interfaces import ImageMetadata, RequestID, TokenBuffer
from max.kv_cache import InsufficientBlocksError, PagedKVCacheManager
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    RaggedKVCacheInputs,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.support.image import hash_image
from test_common.context_utils import create_text_context


def gen_prompt(length: int) -> np.ndarray:
    """Generate a random binary string with given length."""
    return np.random.randint(0, 2, size=length)


def get_blocks_from_kv_tuple(kv_tuple: RaggedKVCacheInputs) -> list[list[int]]:
    return kv_tuple[2].to_numpy().tolist()


def get_uncommitted_and_committed_block_counts(
    kv_tuple: RaggedKVCacheInputs,
) -> list[list[int]]:
    return kv_tuple[3].to_numpy().tolist()


def get_cache_lengths_from_kv_tuple(kv_tuple: RaggedKVCacheInputs) -> list[int]:
    return kv_tuple[1].to_numpy().tolist()


def create_kv_cache(num_blocks: int, page_size: int = 1) -> PagedKVCacheManager:
    kv_params = KVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=1,
        head_dim=1,
        cache_strategy="paged",
        enable_prefix_caching=True,
        page_size=page_size,
        devices=[DeviceRef.CPU()],
    )

    session = InferenceSession(devices=[CPU()])

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=num_blocks,
        session=session,
        enable_runtime_checks=True,
    )

    return kv_manager


@pytest.mark.asyncio
async def test_prefix_caching_basic() -> None:
    kv_manager = create_kv_cache(num_blocks=128)

    # Reserve a slot in the KV cache manager.
    initial_prompt_1 = [10, 11, 12, 13, 14]
    context_1 = create_text_context(np.array(initial_prompt_1, dtype=np.int64))
    kv_manager.claim(context_1.request_id, replica_idx=0)
    kv_manager.alloc(context_1, replica_idx=0, num_steps=6)

    # Seq 1: Prefill 10 - 14
    batch = [context_1]
    kv_tuple_list = kv_manager.get_runtime_inputs([batch])
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        len(initial_prompt_1),
        len(initial_prompt_1),
    ]
    batch[0].update(15)
    kv_manager.step([batch])

    # Check that we got new blocks
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3, 4]

    # Seq 1: Token gen 15 - 18
    toks = [15, 16, 17, 18, 19]
    for i in range(len(toks) - 1):
        kv_tuple_list = kv_manager.get_runtime_inputs([batch])
        assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[
            0
        ] == [1, 5 + i + 1]
        batch[0].update(toks[i + 1])
        kv_manager.step([batch])

    # Seq 2: Claim
    initial_prompt_2 = [10, 11, 12, 13]
    context_2 = create_text_context(np.array(initial_prompt_2, dtype=np.int64))
    batch = [context_2]
    kv_manager.claim(context_2.request_id, replica_idx=0)
    kv_manager.alloc(context_2, replica_idx=0, num_steps=5)

    # Seq 2: Prefill 10 - 13
    kv_tuple_list = kv_manager.get_runtime_inputs([batch])
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        1,
        len(initial_prompt_2),
    ]
    batch[0].update(14)
    kv_manager.step([batch])

    # Check that we got cached blocks, except for last token in prompt
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:3] == [0, 1, 2]
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][3] != [3]

    # Seq 2: Token gen 14 - 17
    toks = [14, 15, 99, 100, 101]
    for i in range(len(toks) - 1):
        kv_tuple_list = kv_manager.get_runtime_inputs([batch])
        assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[
            0
        ] == [1, len(initial_prompt_2) + i + 1]
        assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:4] == [0, 1, 2, 3]
        batch[0].update(toks[i + 1])
        kv_manager.step([batch])

    # first and second ce have 5 + 4 tokens
    metrics = kv_manager.get_metrics(replica_idx=0)
    assert metrics.prompt_tokens == 9
    # second ce gets cache hit on 3 tokens
    assert metrics.cache_tokens == 3
    # cache hit rate is = 3 / 9
    assert metrics.cache_hit_rate >= 0.333


@pytest.mark.asyncio
async def test_prefix_caching_reset_prefix_cache() -> None:
    kv_manager = create_kv_cache(num_blocks=128)
    # This is a noop
    kv_manager.reset_prefix_cache()

    prompt = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    context_1 = create_text_context(prompt)
    context_2 = create_text_context(prompt)
    context_3 = create_text_context(prompt)

    # Get cache hit of 0 tokens since the prefix cache is empty
    with kv_manager.reserve([context_1], replica_idx=0, num_steps=1):
        kv_manager.get_runtime_inputs([[context_1]])
        context_1.update(15)
        kv_manager.step([[context_1]])
    assert kv_manager.get_metrics(replica_idx=0).cache_tokens == 0

    # Get cache hit of 4 tokens
    with kv_manager.reserve([context_2], replica_idx=0, num_steps=1):
        pass
    assert kv_manager.get_metrics(replica_idx=0).cache_tokens == 4

    # Get cache hit of 0 tokens since we reset the prefix cache
    kv_manager.reset_prefix_cache()
    with kv_manager.reserve([context_3], replica_idx=0, num_steps=1):
        pass
    assert kv_manager.get_metrics(replica_idx=0).cache_tokens == 4


@pytest.mark.asyncio
async def test_prefix_caching_with_repeating_prompt() -> None:
    kv_manager = create_kv_cache(num_blocks=128)

    available_blocks = 128

    # Try to assign and release more than 128 blocks.
    for i in range(1000):
        # We reuse the same prompt each time, allowing for prefix sharing.
        prompt = np.array([100, 101, 102, 103, 104], dtype=np.int64)
        batch = [create_text_context(prompt)]
        context = batch[0]
        with kv_manager.reserve([context], replica_idx=0, num_steps=1):
            _ = kv_manager.get_runtime_inputs([batch])

            if i == 0:
                # During first fetch, we do not get a cache hit so we use 5 blocks.
                available_blocks -= 5
            else:
                # During later fetches, we get a cache hit so we use 1 block.
                available_blocks -= 1
            assert available_blocks >= 0

            context.update(42)
            kv_manager.step([batch])

            if i != 0:
                # During later fetches, we will just release the block we wrote to
                # since a different block already exists for the same token.
                available_blocks += 1

    # cache hit rate is ~= 4 / 5 tokens
    assert kv_manager.get_metrics(replica_idx=0).cache_hit_rate > 0.79


@pytest.mark.asyncio
async def test_prefix_caching_with_no_release() -> None:
    np.random.seed(12345)

    kv_manager = create_kv_cache(num_blocks=128)

    # Try to allocate more than 128 blocks.
    # We expect to run out of blocks here.
    with pytest.raises(InsufficientBlocksError):
        for _ in range(1000):
            prompt = gen_prompt(16)
            batch = [create_text_context(prompt)]
            kv_manager.claim(batch[0].request_id, replica_idx=0)
            kv_manager.alloc(batch[0], replica_idx=0, num_steps=1)
            _ = kv_manager.get_runtime_inputs([batch])
            batch[0].update(42)
            kv_manager.step([batch])

            # We intentionally do not release the sequence here!

    assert kv_manager.get_metrics(replica_idx=0).cache_hit_rate > 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page_size, num_steps",
    [
        (1, 1),
        (1, 4),
        (3, 4),
        (4, 1),
        (4, 3),
        (4, 4),
        (64, 1),
        (64, 10),
        (128, 1),
        (128, 10),
    ],
)
async def test_prefix_caching_with_random_prompts(
    page_size: int, num_steps: int
) -> None:
    np.random.seed(12345)

    num_blocks = 128
    kv_manager = create_kv_cache(num_blocks=num_blocks, page_size=page_size)
    available_slots = num_blocks * page_size

    # Try to assign and release more than 128 blocks.
    for _ in range(1000):
        slots_used = 0
        # Picking random prompts.
        prompt_len = np.random.randint(1, 64)
        prompt = gen_prompt(prompt_len)
        batch = [create_text_context(prompt)]
        context = batch[0]
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=num_steps)
        # This fetch can trigger evictions from the tree.
        _ = kv_manager.get_runtime_inputs([batch], num_steps=num_steps)
        new_tokens = gen_prompt(num_steps)
        for tok in new_tokens:
            context.update(tok)
        kv_manager.step([batch])

        slots_used_in_curr_iter = prompt_len + num_steps - 1
        slots_used += slots_used_in_curr_iter

        # Perform some number of token generation steps.
        num_of_tg_steps = np.random.randint(0, 20)
        for _ in range(num_of_tg_steps):
            prompt_len = np.random.randint(1, 4)

            # If this single sequence will exceed the total number of slots,
            # break out of the loop.
            slots_used_in_curr_iter = prompt_len + num_steps - 1
            if slots_used + slots_used_in_curr_iter > available_slots:
                break

            prompt = gen_prompt(prompt_len - 1)

            orig_start_idx = context.tokens.processed_length
            for tok in prompt:
                context.update(tok)

            context.tokens.rewind_processing(
                context.tokens.processed_length - orig_start_idx
            )

            # This fetch can trigger evictions from the tree.
            for ctx in batch:
                kv_manager.alloc(ctx, replica_idx=0, num_steps=num_steps)

            _ = kv_manager.get_runtime_inputs([batch], num_steps=num_steps)
            new_tokens = gen_prompt(num_steps)
            for tok in new_tokens:
                context.update(tok)
            kv_manager.step([batch])

            slots_used += slots_used_in_curr_iter

        kv_manager.release(context.request_id, replica_idx=0)

    assert kv_manager.get_num_used_pages(replica_idx=0) == 0


@pytest.mark.asyncio
async def test_prefix_caching_with_num_steps_gt_1() -> None:
    kv_manager = create_kv_cache(num_blocks=128)

    # Reserve a slot in the KV cache manager.
    initial_prompt_1 = [10, 11, 12, 13, 14]

    # Seq 1: Prefill 10 - 14 and generate 15 - 17 in one pass
    batch = [create_text_context(np.array(initial_prompt_1))]
    for context in batch:
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=3)

    kv_tuple_list = kv_manager.get_runtime_inputs([batch], num_steps=3)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
        [1, 6],
        [1, 7],
    ]

    for tok in [15, 16, 17]:
        batch[0].update(tok)
    kv_manager.step([batch])

    # Seq 1: Token gen 18 - 19 in one pass
    for ctx in batch:
        kv_manager.alloc(ctx, replica_idx=0, num_steps=2)
    kv_tuple_list = kv_manager.get_runtime_inputs([batch], num_steps=2)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 8],
        [1, 9],
    ]

    for tok in [18, 19]:
        batch[0].update(tok)
    kv_manager.step([batch])

    assert kv_manager.get_metrics(replica_idx=0).cache_hit_rate == 0.0


@pytest.mark.asyncio
async def test_prefix_caching_with_page_size_gt_1() -> None:
    kv_manager = create_kv_cache(num_blocks=128, page_size=2)

    # Seq 1: Prefill 10 - 14
    batch = [create_text_context(np.array([10, 11, 12, 13, 14]))]
    kv_manager.claim(batch[0].request_id, replica_idx=0)
    kv_manager.alloc(batch[0], replica_idx=0, num_steps=5)
    kv_tuple_list = kv_manager.get_runtime_inputs([batch])
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
    ]

    batch[0].update(15)
    kv_manager.step([batch])

    # Seq 1: Token gen 15
    kv_tuple_list = kv_manager.get_runtime_inputs([batch])
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 6],
    ]

    batch[0].update(16)
    kv_manager.step([batch])

    # Seq 1: Token gen 16
    kv_tuple_list = kv_manager.get_runtime_inputs([batch])
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 7],
    ]

    batch[0].update(17)
    kv_manager.step([batch])

    assert kv_manager.get_metrics(replica_idx=0).cache_hit_rate == 0.0


@pytest.mark.asyncio
async def test_prefix_caching_with_page_size_gt_1_and_num_steps_gt_1() -> None:
    kv_manager = create_kv_cache(num_blocks=128, page_size=2)

    # Seq 1: Prefill 10 - 14 and generate 15 - 17 in one pass
    batch = [create_text_context(np.array([10, 11, 12, 13, 14]))]
    kv_manager.claim(batch[0].request_id, replica_idx=0)
    kv_manager.alloc(batch[0], replica_idx=0, num_steps=5)
    kv_tuple_list = kv_manager.get_runtime_inputs([batch], num_steps=3)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
        [1, 6],
        [1, 7],
    ]

    for tok in [15, 16, 17]:
        batch[0].update(tok)
    kv_manager.step([batch])

    # Seq 1: Token gen 18 - 19 in one pass
    for ctx in batch:
        kv_manager.alloc(ctx, replica_idx=0, num_steps=2)
    kv_tuple_list = kv_manager.get_runtime_inputs([batch], num_steps=2)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3, 4]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 8],
        [1, 9],
    ]

    for tok in [18, 19]:
        batch[0].update(tok)
    kv_manager.step([batch])


class FakeModel:
    """Create a fake model that can be used to test prefix caching."""

    def __init__(self, kv_manager: PagedKVCacheManager) -> None:
        self.page_size = kv_manager.page_size
        self.total_num_pages = kv_manager.get_num_pages(replica_idx=0)
        # block_projections maps from bid -> offset -> prefix tokens
        self.block_projections: dict[int, dict[int, np.ndarray]] = defaultdict(
            dict
        )
        # request_ids_and_all_tokens accumulates tokens for each request
        # Empty arrays are used as initial state for concatenation
        self.request_ids_and_all_tokens: dict[RequestID, np.ndarray] = {}
        self._empty_token_array = np.array([], dtype=np.int64)

    def run(
        self,
        request_ids_and_prompts: dict[RequestID, np.ndarray],
        kv_inputs: Sequence[RaggedKVCacheInputs],
        num_steps: int,
        request_ids_and_new_tokens: dict[RequestID, np.ndarray] | None = None,
    ) -> dict[RequestID, np.ndarray]:
        """Given a batch and the kv_inputs, we `run` the model and check that
        the paged manager gave us valid blocks that contain the appropriate KV
        projections.

        This function returns the new tokens that were 'generated' by the model.
        """
        # generate some new tokens
        if request_ids_and_new_tokens is None:
            request_ids_and_new_tokens = {}
            for request_id in request_ids_and_prompts:
                new_toks = gen_prompt(num_steps)
                request_ids_and_new_tokens[request_id] = new_toks

        # update all tokens to contain the tokens which should have a KV
        # projection in the cache after this forward step
        for request_id in request_ids_and_prompts:
            self.request_ids_and_all_tokens[request_id] = np.concatenate(
                [
                    self.request_ids_and_all_tokens.get(
                        request_id, self._empty_token_array
                    ),
                    request_ids_and_prompts[request_id],
                    request_ids_and_new_tokens[request_id][:-1],
                ]
            )

        # read the blocks and cache lengths returned by the paged manager
        all_blocks = get_blocks_from_kv_tuple(kv_inputs[0])
        cache_lengths = get_cache_lengths_from_kv_tuple(kv_inputs[0])

        for batch_idx, request_id in enumerate(request_ids_and_prompts):
            blocks = all_blocks[batch_idx]
            cache_len = cache_lengths[batch_idx]
            tokens = self.request_ids_and_all_tokens[request_id]
            # count the number of unassigned blocks by checking for blocks with
            # id equal to self.total_num_pages.
            num_unassigned_blocks = blocks.count(self.total_num_pages)
            num_assigned_blocks = len(blocks) - num_unassigned_blocks
            cache_space = num_assigned_blocks * self.page_size
            # check that we have enough cache space to store all tokens which
            # need a KV projection
            assert len(tokens) <= cache_space

            for idx in range(len(tokens)):
                prefix = tokens[: idx + 1]
                bidx = idx // self.page_size
                block = blocks[bidx]
                block_offset = idx % self.page_size

                if idx < cache_len:
                    # if the token is cached, ensure that the block + offset
                    # map to the appropriate prefix tokens
                    # specifying key otherwise assertion printing of `block_projections` would be too loud
                    blocks_in_cache = self.block_projections.keys()
                    assert block in blocks_in_cache
                    assert block_offset in self.block_projections[block]
                    actual = self.block_projections[block][block_offset]
                    expected = prefix
                    assert (
                        actual.shape == expected.shape
                        and (actual == expected).all()
                    ), (
                        f"KV projection mismatch for block {block} at offset {block_offset}. Expected {expected}, got {actual}"
                    )
                else:
                    # if the token is not cached, update the block_projections
                    self.block_projections[block][block_offset] = prefix

        return request_ids_and_new_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "batch_size, num_steps, shared_prefix_len, page_size",
    [
        (1, 1, 0, 1),
        (3, 3, 33, 3),
        (4, 5, 29, 3),
        (5, 1, 33, 5),
        (6, 2, 29, 5),
        (7, 3, 33, 6),
        (8, 9, 29, 13),
        (30, 1, 75, 128),
        (30, 10, 75, 128),
    ],
)
async def test_prefix_caching_grouped_prefixes(
    batch_size: int, num_steps: int, shared_prefix_len: int, page_size: int
) -> None:
    """
    Test e2e prefix caching, ensuring that we do not leak memory.
    """
    np.random.seed(12345)

    # evictions will not happen since we allocate so many blocks
    kv_manager = create_kv_cache(num_blocks=10000, page_size=page_size)
    model = FakeModel(kv_manager)

    # generate a number of grouped prefixes:
    shared_prefix = gen_prompt(shared_prefix_len)
    group_prefixes: list[np.ndarray] = []
    num_prompts = 20
    for _ in range(num_prompts):
        random_len = np.random.randint(0, 100)
        group_prefix = np.concatenate([shared_prefix, gen_prompt(random_len)])
        group_prefixes.append(group_prefix)

    # run CE on 15 batches of batch_size requests each
    num_batches = 15
    batch: dict[RequestID, TextContext] = {}
    for b in range(num_batches):
        for r in range(batch_size):
            request_index = b * batch_size + r
            group_prefix = group_prefixes[request_index % len(group_prefixes)]
            random_len = np.random.randint(0, 10)
            prompt = np.concatenate([group_prefix, gen_prompt(random_len)])
            ctx = create_text_context(prompt)
            kv_manager.claim(ctx.request_id, replica_idx=0)
            batch[ctx.request_id] = ctx

        ctxs = list(batch.values())
        request_ids_and_prompts = {
            request_id: batch[request_id].tokens.active for request_id in batch
        }
        for ctx in ctxs:
            kv_manager.alloc(ctx, replica_idx=0, num_steps=num_steps)
        kv_inputs = kv_manager.get_runtime_inputs([ctxs], num_steps=num_steps)
        request_ids_and_new_tokens_batch = model.run(
            request_ids_and_prompts, kv_inputs, num_steps=num_steps
        )
        for request_id in request_ids_and_new_tokens_batch:
            ctx = batch[request_id]
            for tok in request_ids_and_new_tokens_batch[request_id]:
                ctx.update(tok)
        kv_manager.step([ctxs])

    # Since our prompts have large grouped prefixes, we should have a high cache
    # hit rate.
    cache_hit_rate = kv_manager.get_metrics(replica_idx=0).cache_hit_rate
    if shared_prefix_len > 0:
        assert cache_hit_rate > 0.45

    # run TG on all requests for num_tg_steps steps
    # we terminate requests with probability 10% each iteration
    num_tg_steps = 100
    for _ in range(num_tg_steps):
        for ctx in batch.values():
            extended = gen_prompt(np.random.randint(0, 10))
            orig_start_idx = ctx.tokens.processed_length
            for tok in extended:
                ctx.update(tok)
            ctx.tokens.rewind_processing(
                ctx.tokens.processed_length - orig_start_idx
            )

        ctxs = list(batch.values())
        orig_request_ids_and_prompts = {
            request_id: ctx.tokens.active for request_id, ctx in batch.items()
        }
        for ctx in ctxs:
            kv_manager.alloc(ctx, replica_idx=0, num_steps=num_steps)
        kv_inputs = kv_manager.get_runtime_inputs([ctxs], num_steps=num_steps)
        request_ids_and_new_tokens_subset = model.run(
            orig_request_ids_and_prompts, kv_inputs, num_steps=num_steps
        )

        for request_id in request_ids_and_new_tokens_subset:
            ctx = batch[request_id]
            for tok in request_ids_and_new_tokens_subset[request_id]:
                ctx.update(tok)
        kv_manager.step([ctxs])

        # copying keys so we don't iterate over dict while deleting things
        copied_request_ids = list(batch.keys())
        for request_id in copied_request_ids:
            # terminate requests with probability 10%
            if len(batch) > 1 and np.random.rand() < 0.1:
                kv_manager.release(batch[request_id].request_id, replica_idx=0)
                del batch[request_id]

    for request_id in batch:
        kv_manager.release(request_id, replica_idx=0)


def run_forward(
    model: FakeModel,
    kv_manager: PagedKVCacheManager,
    ctx: TextContext,
    prompt: np.ndarray,
    next_tok: int,
    run_fetch: bool = True,
    run_step: bool = True,
) -> None:
    """Process a forward pass with the given prompt and next token.

    If the context has unprocessed tokens, they are included in the prompt
    passed to the model to ensure proper KV projection tracking.
    """
    orig_start_idx = ctx.tokens.processed_length
    num_unprocessed = len(ctx.tokens.all) - ctx.tokens.processed_length

    for tok in prompt:
        ctx.update(tok)
    ctx.tokens.rewind_processing(ctx.tokens.processed_length - orig_start_idx)

    # Include any unprocessed tokens from context in the prompt for the model
    if num_unprocessed > 0:
        all_tokens_array = np.array(ctx.tokens.all)
        unprocessed_tokens = all_tokens_array[:num_unprocessed]
        full_prompt = np.concatenate([unprocessed_tokens, prompt])
    else:
        full_prompt = prompt

    batch = [ctx]
    request_ids_and_prompts = {ctx.request_id: full_prompt}
    orig_request_ids_and_prompts = request_ids_and_prompts.copy()
    new_toks = {ctx.request_id: np.array([next_tok])}
    if run_fetch:
        kv_manager.alloc(ctx, replica_idx=0, num_steps=1)

    kv_inputs = kv_manager.get_runtime_inputs([batch], num_steps=1)
    _ = model.run(
        orig_request_ids_and_prompts,
        kv_inputs,
        num_steps=1,
        request_ids_and_new_tokens=new_toks,
    )
    if run_step:
        ctx.update(next_tok)
        kv_manager.step([batch])


@pytest.mark.asyncio
async def test_prefix_caching_chunked_prefill() -> None:
    """Test chunked prefill where prompts are processed in multiple forward passes.

    This test verifies that prefix caching works correctly when:
    1. Two sequences share a common prefix
    2. The prompts are processed in chunks rather than all at once
    3. Sequences diverge at different points
    """
    kv_manager = create_kv_cache(num_blocks=128, page_size=3)
    model = FakeModel(kv_manager)

    # Define prompts for chunked prefill - use non-empty arrays
    # Start with initial token, then process rest in chunks
    initial_token = np.array([10], dtype=np.int64)
    prompt_1_part_1_rest = np.array(
        [11, 12, 13, 14, 15, 16, 17], dtype=np.int64
    )
    prompt_1_part_2 = np.array([18, 19, 20, 21, 22], dtype=np.int64)

    prompt_2_part_1_rest = np.array(
        [11, 12, 13, 14, 15, 16, 17], dtype=np.int64
    )
    prompt_2_part_2 = np.array([16, 17, 18, 998, 999], dtype=np.int64)

    # === Create contexts with initial token (non-empty) ===
    ctx_1 = create_text_context(initial_token)
    kv_manager.claim(ctx_1.request_id, replica_idx=0)

    ctx_2 = create_text_context(initial_token)
    kv_manager.claim(ctx_2.request_id, replica_idx=0)

    # Process the remaining tokens in chunks - run_forward will include initial unprocessed token
    run_forward(
        model, kv_manager, ctx_1, prompt_1_part_1_rest, prompt_1_part_2[0]
    )
    run_forward(
        model, kv_manager, ctx_2, prompt_2_part_1_rest, prompt_2_part_2[0]
    )
    run_forward(model, kv_manager, ctx_1, prompt_1_part_2, 42)

    # Make sure that we don't return block 2 for seq_id_2 since its last KV
    # projection differs.
    # block 2 holds projections for [..., 16, 17, 18]
    # seq_id_2 needs projections for [..., 16, 17, 16]
    run_forward(model, kv_manager, ctx_2, prompt_2_part_2, 42)
    blocks = kv_manager.get_req_blocks(ctx_2.request_id, replica_idx=0)
    assert 2 not in blocks

    assert kv_manager.get_metrics(replica_idx=0).cache_tokens == 6
    assert kv_manager.get_metrics(replica_idx=0).cache_hit_rate > 0.2


def run_and_check_num_cached_tokens(
    kv_manager: PagedKVCacheManager,
    ctx: TextAndVisionContext,
    do_step: bool = True,
) -> int:
    # reset cache_tokens to 0
    kv_manager.reset_metrics()
    kv_manager.claim(ctx.request_id, replica_idx=0)
    kv_manager.alloc(ctx, replica_idx=0, num_steps=1)
    kv_manager.get_runtime_inputs([[ctx]])
    magic_token_value = 42  # this is arbitrary
    if do_step:
        ctx.update(magic_token_value)
        kv_manager.step([[ctx]])
    return kv_manager.get_metrics(replica_idx=0).cache_tokens


@pytest.mark.asyncio
async def test_prefix_caching_with_images() -> None:
    IMG = 99
    kv_manager = create_kv_cache(num_blocks=128, page_size=1)

    vision_token_ids = [IMG]

    img1_pixels = np.array([[777], [777], [777]])
    # This differs from img1_pixels and should result in a different hash
    img3_pixels = np.array([[777], [777], [888]])

    #                              |<-     img     ->|
    tokens1 = np.array([51, 52, 53, IMG, IMG, IMG, IMG, 54, 55])
    #                      |<-     img      ->|
    tokens4 = np.array([51, IMG, IMG, IMG, IMG, 52, 53, 54, 55, 56])

    # ctx1 and ctx2 are exactly the same
    ctx1 = TextAndVisionContext(
        max_length=100,
        tokens=TokenBuffer(tokens1),
        images=[
            ImageMetadata(
                start_idx=3,
                end_idx=7,
                pixel_values=img1_pixels,
                image_hash=hash_image(img1_pixels),
            ),
        ],
        vision_token_ids=vision_token_ids,
    )
    ctx2 = TextAndVisionContext(
        max_length=100,
        tokens=TokenBuffer(tokens1),
        images=[
            ImageMetadata(
                start_idx=3,
                end_idx=7,
                pixel_values=img1_pixels,
                image_hash=hash_image(img1_pixels),
            ),
        ],
        vision_token_ids=vision_token_ids,
    )

    # We should not get any cache hits since cache is empty
    assert run_and_check_num_cached_tokens(kv_manager, ctx1) == 0

    # We should get cache hits on all 8 eligible tokens
    assert run_and_check_num_cached_tokens(kv_manager, ctx2) == 8

    # ctx3 has different image pixels from ctx1 and ctx2
    ctx3 = TextAndVisionContext(
        max_length=100,
        tokens=TokenBuffer(tokens1),
        images=[
            ImageMetadata(
                start_idx=3,
                end_idx=7,
                pixel_values=img3_pixels,
                image_hash=hash_image(img3_pixels),
            ),
        ],
        vision_token_ids=vision_token_ids,
    )

    # We should only get cache hits on the first 3 tokens prior to the image
    assert run_and_check_num_cached_tokens(kv_manager, ctx3) == 3

    # ctx4 also has img1 but it is in a different position
    ctx4 = TextAndVisionContext(
        max_length=100,
        tokens=TokenBuffer(tokens4),
        images=[
            ImageMetadata(
                start_idx=1,
                end_idx=5,
                pixel_values=img1_pixels,
                image_hash=hash_image(img1_pixels),
            ),
        ],
        vision_token_ids=vision_token_ids,
    )

    # Because the image is in a different position, we should only get a cache
    # hit on the first token
    assert run_and_check_num_cached_tokens(kv_manager, ctx4) == 1


@pytest.mark.asyncio
async def test_prefix_caching_with_images_and_page_size_gt_1() -> None:
    IMG = 99
    kv_manager = create_kv_cache(num_blocks=128, page_size=4)
    #   |    block 0     |      block 1     |      block 2     |
    #               |<--------- img0 ----------->|     |<-- img1 -->|
    tokens = np.array(
        [51, 52, 53, IMG, IMG, IMG, IMG, IMG, IMG, 54, IMG, IMG, IMG, 56]
    )

    img0_pixels = np.array([[777], [777], [777]])
    img1_pixels = np.array([[888], [888], [888]])

    ctx0 = TextAndVisionContext(
        max_length=100,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=3,
                end_idx=9,
                pixel_values=img0_pixels,
                image_hash=hash_image(img0_pixels),
            ),
            ImageMetadata(
                start_idx=10,
                end_idx=13,
                pixel_values=img1_pixels,
                image_hash=hash_image(img1_pixels),
            ),
        ],
        vision_token_ids=[IMG],
    )

    # Duplicate of above
    ctx1 = TextAndVisionContext(
        max_length=100,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=3,
                end_idx=9,
                pixel_values=img0_pixels,
                image_hash=hash_image(img0_pixels),
            ),
            ImageMetadata(
                start_idx=10,
                end_idx=13,
                pixel_values=img1_pixels,
                image_hash=hash_image(img1_pixels),
            ),
        ],
        vision_token_ids=[IMG],
    )

    assert run_and_check_num_cached_tokens(kv_manager, ctx0) == 0
    # Notice that this is NOT rounded down to the nearest image boundary
    assert (
        run_and_check_num_cached_tokens(kv_manager, ctx1, do_step=False) == 12
    )

    assert ctx1.tokens.processed_length == 12
    assert (
        kv_manager._replica_managers[0].block_manager.req_to_committed_idx[
            ctx1.request_id
        ]
        == 12
    )
