# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.interfaces import InputContext, RequestID
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheManager,
    RaggedKVCacheInputs,
)
from max.nn.kv_cache.paged_cache import block_utils
from max.pipelines.core import TextContext
from test_common.context_utils import create_text_context

block_utils.ENABLE_MOJO_BLOCK_HASHER = True


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


def create_paged_manager(
    num_blocks: int, page_size: int = 1
) -> PagedKVCacheManager:
    # Setting kv_heads, head_dim, and num_layers to 1 so it is easy to compute
    # memory usage. Now we know each block is 1 byte.
    NUM_KV_HEADS = 1
    HEAD_DIM = 1
    NUM_LAYERS = 1

    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=True,
        page_size=page_size,
    )

    session = InferenceSession()

    cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * page_size
        * num_blocks
        * kv_params.dtype.size_in_bytes
    )
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        max_batch_size=512,
        max_seq_len=4096,
        num_layers=NUM_LAYERS,
        devices=[CPU()],
        session=session,
        cache_memory=cache_memory,
        page_size=page_size,
        enable_runtime_checks=True,
    )

    return kv_manager


@pytest.mark.asyncio
async def test_prefix_caching_basic() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    # Reserve a slot in the KV cache manager.
    initial_prompt_1 = [10, 11, 12, 13, 14]
    context_1 = create_text_context(np.array(initial_prompt_1))
    kv_manager.external_claim(context_1.request_id)

    # Seq 1: Prefill 10 - 14
    batch = [context_1]
    kv_tuple_list = kv_manager.fetch(batch)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        len(initial_prompt_1),
        len(initial_prompt_1),
    ]
    batch[0].update(15)
    kv_manager.step(batch)

    # Check that we got new blocks
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3, 4]

    # Seq 1: Token gen 15 - 18
    toks = [15, 16, 17, 18, 19]
    for i, tok in enumerate(toks[:-1]):  # noqa: B007
        kv_tuple_list = kv_manager.fetch(batch)
        assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[
            0
        ] == [1, 5 + i + 1]
        batch[0].update(toks[i + 1])
        kv_manager.step(batch)

    # Seq 2: Claim
    initial_prompt_2 = [10, 11, 12, 13]
    context_2 = create_text_context(np.array(initial_prompt_2))
    batch = [context_2]
    kv_manager.external_claim(context_2.request_id)

    # Seq 2: Prefill 10 - 13
    kv_tuple_list = kv_manager.fetch(batch)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        1,
        len(initial_prompt_2),
    ]
    batch[0].update(14)
    kv_manager.step(batch)

    # Check that we got cached blocks, except for last token in prompt
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:3] == [0, 1, 2]
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][3] != [3]

    # Seq 2: Token gen 14 - 17
    toks = [14, 15, 99, 100, 101]
    for i, tok in enumerate(toks[:-1]):  # noqa: B007
        kv_tuple_list = kv_manager.fetch(batch)
        assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[
            0
        ] == [1, len(initial_prompt_2) + i + 1]
        assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:4] == [0, 1, 2, 3]
        batch[0].update(toks[i + 1])
        kv_manager.step(batch)

    # first and second ce have 4 + 3 tokens
    assert kv_manager.block_manager.prompt_tokens == 7
    # second ce gets cache hit on 3 tokens
    assert kv_manager.block_manager.cached_prompt_tokens == 3
    # cache hit rate is = 3 / 7
    assert kv_manager.cache_hit_rate > 0.42


@pytest.mark.asyncio
async def test_prefix_caching_with_repeating_prompt() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    available_blocks = 128

    # Try to assign and release more than 128 blocks.
    for i in range(1000):
        # We reuse the same prompt each time, allowing for prefix sharing.
        prompt = np.array([100, 101, 102, 103, 104])
        batch = [create_text_context(prompt)]
        context = batch[0]
        kv_manager.external_claim(context.request_id)
        _ = kv_manager.fetch(batch)

        if i == 0:
            # During first fetch, we do not get a cache hit so we use 5 blocks.
            available_blocks -= 5
        else:
            # During later fetches, we get a cache hit so we use 1 block.
            available_blocks -= 1

        context.update(42)
        kv_manager.step(batch)

        if i != 0:
            # During later fetches, we will just release the block we wrote to
            # since a different block already exists for the same token.
            available_blocks += 1

        kv_manager.release(context.request_id)

    assert kv_manager.cache_hit_rate > 0.99


@pytest.mark.asyncio
async def test_prefix_caching_with_no_release() -> None:
    np.random.seed(12345)

    kv_manager = create_paged_manager(num_blocks=128)

    # Try to assign and release more than 128 blocks.
    # We expect to run out of blocks here.
    with pytest.raises(RuntimeError):
        for _ in range(1000):
            prompt = gen_prompt(16)
            batch = [create_text_context(prompt)]
            kv_manager.external_claim(batch[0].request_id)
            _ = kv_manager.fetch(batch)
            batch[0].update(42)
            kv_manager.step(batch)

            # We intentionally do not release the sequence here!

    assert kv_manager.cache_hit_rate > 0.1


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
    kv_manager = create_paged_manager(
        num_blocks=num_blocks, page_size=page_size
    )
    available_slots = num_blocks * page_size

    # Try to assign and release more than 128 blocks.
    for _ in range(1000):
        slots_used = 0
        # Picking random prompts.
        prompt_len = np.random.randint(1, 64)
        prompt = gen_prompt(prompt_len)
        batch = [create_text_context(prompt)]
        context = batch[0]
        kv_manager.external_claim(context.request_id)
        # This fetch can trigger evictions from the tree.
        _ = kv_manager.fetch(batch, num_steps=num_steps)
        new_tokens = gen_prompt(num_steps)
        for tok in new_tokens:
            context.update(tok)
        kv_manager.step(batch)

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

            orig_start_idx = context.start_idx
            for tok in prompt:
                context.update(tok)
            context.set_token_indices(start_idx=orig_start_idx)

            # This fetch can trigger evictions from the tree.
            _ = kv_manager.fetch(batch, num_steps=num_steps)
            new_tokens = gen_prompt(num_steps)
            for tok in new_tokens:
                context.update(tok)
            kv_manager.step(batch)

            slots_used += slots_used_in_curr_iter

        kv_manager.release(context.request_id)

    assert (
        len(kv_manager.block_manager.device_block_pool.free_block_queue)
        == kv_manager.total_num_pages
    )


@pytest.mark.asyncio
async def test_prefix_caching_with_num_steps_gt_1() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    # Reserve a slot in the KV cache manager.
    initial_prompt_1 = [10, 11, 12, 13, 14]

    # Seq 1: Prefill 10 - 14 and generate 15 - 17 in one pass
    batch = [create_text_context(np.array(initial_prompt_1))]
    kv_manager.external_claim(batch[0].request_id)
    kv_tuple_list = kv_manager.fetch(batch, num_steps=3)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
        [1, 6],
        [1, 7],
    ]

    for tok in [15, 16, 17]:
        batch[0].update(tok)
    kv_manager.step(batch)

    # Seq 1: Token gen 18 - 19 in one pass
    kv_tuple_list = kv_manager.fetch(batch, num_steps=2)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 8],
        [1, 9],
    ]

    for tok in [18, 19]:
        batch[0].update(tok)
    kv_manager.step(batch)

    assert kv_manager.cache_hit_rate == 0.0


@pytest.mark.asyncio
async def test_prefix_caching_with_page_size_gt_1() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=2)

    # Seq 1: Prefill 10 - 14
    batch = [create_text_context(np.array([10, 11, 12, 13, 14]))]
    kv_manager.external_claim(batch[0].request_id)
    kv_tuple_list = kv_manager.fetch(batch)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
    ]

    batch[0].update(15)
    kv_manager.step(batch)

    # Seq 1: Token gen 15
    kv_tuple_list = kv_manager.fetch(batch)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 6],
    ]

    batch[0].update(16)
    kv_manager.step(batch)

    # Seq 1: Token gen 16
    kv_tuple_list = kv_manager.fetch(batch)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 7],
    ]

    batch[0].update(17)
    kv_manager.step(batch)

    assert kv_manager.cache_hit_rate == 0.0


@pytest.mark.asyncio
async def test_prefix_caching_with_page_size_gt_1_and_num_steps_gt_1() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=2)

    # Seq 1: Prefill 10 - 14 and generate 15 - 17 in one pass
    batch = [create_text_context(np.array([10, 11, 12, 13, 14]))]
    kv_manager.external_claim(batch[0].request_id)
    kv_tuple_list = kv_manager.fetch(batch, num_steps=3)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
        [1, 6],
        [1, 7],
    ]

    for tok in [15, 16, 17]:
        batch[0].update(tok)
    kv_manager.step(batch)

    # Seq 1: Token gen 18 - 19 in one pass
    kv_tuple_list = kv_manager.fetch(batch, num_steps=2)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3, 4]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 8],
        [1, 9],
    ]

    for tok in [18, 19]:
        batch[0].update(tok)
    kv_manager.step(batch)


class FakeModel:
    """Create a fake model that can be used to test prefix caching."""

    def __init__(self, kv_manager: PagedKVCacheManager) -> None:
        self.page_size = kv_manager.page_size
        self.total_num_pages = kv_manager.total_num_pages
        # block_projections maps from bid -> offset -> prefix tokens
        self.block_projections: dict[int, dict[int, np.ndarray]] = defaultdict(
            lambda: defaultdict(lambda: np.array([]))
        )
        self.request_ids_and_all_tokens: dict[RequestID, np.ndarray] = (
            defaultdict(lambda: np.array([]))
        )

        fake_model = self

        # Monkey patch the memcpy_d2d method to use our mock method.
        block_copy_engine = kv_manager.block_copy_engine
        assert block_copy_engine is not None
        orig_memcpy_d2d = block_copy_engine.memcpy_d2d

        def mock_memcpy_d2d(
            dst: int,
            src: int,
            num_tokens: int,
        ) -> Any:
            assert src in fake_model.block_projections
            assert 0 < num_tokens < fake_model.page_size
            for token in range(num_tokens):
                fake_model.block_projections[dst][token] = (
                    fake_model.block_projections[src][token]
                )

            orig_memcpy_d2d(dst, src, num_tokens)

        block_copy_engine.memcpy_d2d = mock_memcpy_d2d  # type: ignore[method-assign]

    def run(
        self,
        request_ids_and_prompts: dict[RequestID, np.ndarray],
        fetch_kv_tuple: Sequence[RaggedKVCacheInputs],
        num_steps: int,
        request_ids_and_new_tokens: Optional[
            dict[RequestID, np.ndarray]
        ] = None,
    ) -> dict[RequestID, np.ndarray]:
        """Given a batch and the fetch_kv_tuple, we `run` the model and check that
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
                    self.request_ids_and_all_tokens[request_id],
                    request_ids_and_prompts[request_id],
                    request_ids_and_new_tokens[request_id][:-1],
                ]
            )

        # read the blocks and cache lengths returned by the paged manager
        all_blocks = get_blocks_from_kv_tuple(fetch_kv_tuple[0])
        cache_lengths = get_cache_lengths_from_kv_tuple(fetch_kv_tuple[0])

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
    kv_manager = create_paged_manager(num_blocks=10000, page_size=page_size)
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
            kv_manager.external_claim(ctx.request_id)
            batch[ctx.request_id] = ctx

        ctxs = list(batch.values())
        request_ids_and_prompts = {
            request_id: batch[request_id].next_tokens for request_id in batch
        }
        fetch_kv_tuple = kv_manager.fetch(ctxs, num_steps=num_steps)
        request_ids_and_new_tokens_batch = model.run(
            request_ids_and_prompts, fetch_kv_tuple, num_steps=num_steps
        )
        for request_id in request_ids_and_new_tokens_batch:
            ctx = batch[request_id]
            for tok in request_ids_and_new_tokens_batch[request_id]:
                ctx.update(tok)
        kv_manager.step(ctxs)

    # Since our prompts have large grouped prefixes, we should have a high cache
    # hit rate.
    cache_hit_rate = kv_manager.cache_hit_rate
    if shared_prefix_len > 0:
        assert cache_hit_rate > 0.45

    # run TG on all requests for num_tg_steps steps
    # we terminate requests with probability 10% each iteration
    num_tg_steps = 100
    for _ in range(num_tg_steps):
        for ctx in batch.values():
            extended = gen_prompt(np.random.randint(0, 10))
            orig_start_idx = ctx.start_idx
            for tok in extended:
                ctx.update(tok)
            ctx.set_token_indices(start_idx=orig_start_idx)

        ctxs = list(batch.values())
        orig_request_ids_and_prompts = {
            request_id: ctx.next_tokens for request_id, ctx in batch.items()
        }
        fetch_kv_tuple = kv_manager.fetch(ctxs, num_steps=num_steps)
        request_ids_and_new_tokens_subset = model.run(
            orig_request_ids_and_prompts, fetch_kv_tuple, num_steps=num_steps
        )

        for request_id in request_ids_and_new_tokens_subset:
            ctx = batch[request_id]
            for tok in request_ids_and_new_tokens_subset[request_id]:
                ctx.update(tok)
        kv_manager.step(ctxs)

        # copying keys so we don't iterate over dict while deleting things
        copied_request_ids = list(batch.keys())
        for request_id in copied_request_ids:
            # terminate requests with probability 10%
            if len(batch) > 1 and np.random.rand() < 0.1:
                kv_manager.release(batch[request_id].request_id)
                del batch[request_id]

    for request_id in batch:
        kv_manager.release(request_id)


def run_forward(
    model: FakeModel,
    kv_manager: PagedKVCacheManager,
    ctx: InputContext,
    prompt: np.ndarray,
    next_tok: int,
    run_fetch: bool = True,
    run_step: bool = True,
) -> None:
    orig_start_idx = ctx.start_idx
    for tok in prompt:
        ctx.update(tok)
    ctx.set_token_indices(start_idx=orig_start_idx)
    batch = [ctx]
    request_ids_and_prompts = {ctx.request_id: prompt}
    orig_request_ids_and_prompts = request_ids_and_prompts.copy()
    new_toks = {ctx.request_id: np.array([next_tok])}
    if run_fetch:
        scheduled = kv_manager.prefetch(ctx, num_steps=1)
        assert scheduled

        fetch_kv_tuple = kv_manager.fetch(batch, num_steps=1)
        _ = model.run(
            orig_request_ids_and_prompts,
            fetch_kv_tuple,
            num_steps=1,
            request_ids_and_new_tokens=new_toks,
        )
    if run_step:
        ctx.update(next_tok)
        kv_manager.step(batch)


@pytest.mark.asyncio
async def test_prefix_caching_chunked_prefill() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=3)
    model = FakeModel(kv_manager)

    ctx_1 = create_text_context(np.array([]))
    ctx_2 = create_text_context(np.array([]))
    kv_manager.external_claim(ctx_1.request_id)
    kv_manager.external_claim(ctx_2.request_id)

    prompt_1_part_1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    prompt_1_part_2 = np.array([18, 19, 20, 21, 22])

    prompt_2_part_1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    prompt_2_part_2 = np.array([16, 17, 18, 998, 999])

    run_forward(model, kv_manager, ctx_1, prompt_1_part_1, prompt_1_part_2[0])
    run_forward(model, kv_manager, ctx_2, prompt_2_part_1, prompt_2_part_2[0])
    run_forward(model, kv_manager, ctx_1, prompt_1_part_2, 42)

    # Make sure that we don't return block 2 for seq_id_2 since its last KV
    # projection differs.
    # block 2 holds projections for [..., 16, 17, 18]
    # seq_id_2 needs projections for [..., 16, 17, 16]
    run_forward(model, kv_manager, ctx_2, prompt_2_part_2, 42)
    blocks = kv_manager.get_req_blocks(ctx_2.request_id)
    assert 2 not in blocks

    assert kv_manager.block_manager.cached_prompt_tokens == 6
    assert kv_manager.cache_hit_rate > 0.2


@pytest.mark.asyncio
async def test_prefix_caching_cow() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=3)
    model = FakeModel(kv_manager)

    ctx = create_text_context(np.array([]))
    kv_manager.external_claim(ctx.request_id)
    prompt_1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    run_forward(model, kv_manager, ctx, prompt_1, 42)

    def run_forward_cow(prompt: np.ndarray, cache_idx: int) -> None:
        ctx = create_text_context(np.array([]))
        kv_manager.external_claim(ctx.request_id)
        run_forward(
            model,
            kv_manager,
            ctx,
            prompt,
            42,
            run_fetch=True,
            run_step=False,
        )
        assert ctx.start_idx == cache_idx

    run_forward_cow(prompt=np.array([10, 11, 12, 13, 14, 22]), cache_idx=5)
    assert kv_manager.num_blocks_copied.d2d == 1
    run_forward_cow(prompt=np.array([10, 11, 22]), cache_idx=2)
    assert kv_manager.num_blocks_copied.d2d == 2
    run_forward_cow(prompt=np.array([10, 11, 12]), cache_idx=2)
    assert kv_manager.num_blocks_copied.d2d == 3
