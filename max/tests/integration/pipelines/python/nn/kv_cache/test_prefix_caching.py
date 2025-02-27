# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.kv_cache import KVCacheParams, KVCacheStrategy
from max.pipelines.kv_cache.paged_cache import PagedKVCacheManager

FAKE_TOKEN = 999


def gen_prompt(length: int) -> np.ndarray:
    """Generate a random binary string with given length."""
    return np.random.randint(0, 2, size=length)


def get_blocks_from_kv_tuple(kv_tuple) -> list[list[int]]:
    return kv_tuple[2].to_numpy().tolist()


def get_uncommitted_and_committed_block_counts(
    kv_tuple,
) -> list[list[int]]:
    return kv_tuple[3].to_numpy().tolist()


def get_cache_lengths_from_kv_tuple(kv_tuple) -> list[int]:
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

    assert len(kv_manager.available_blocks) == num_blocks
    return kv_manager


@pytest.mark.asyncio
async def test_prefix_caching() -> None:
    kv_manager = create_paged_manager(num_blocks=128)
    assert kv_manager.prefix_cache is not None

    # Reserve a slot in the KV cache manager.
    seq_id_1 = 1
    initial_prompt_1 = [10, 11, 12, 13, 14]
    kv_manager.external_claim([seq_id_1])

    # Seq 1: Prefill 10 - 14
    seq_ids_and_prompts = {seq_id_1: np.array(initial_prompt_1)}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        len(initial_prompt_1),
        len(initial_prompt_1),
    ]
    seq_ids_and_new_tokens = {seq_id_1: np.array([FAKE_TOKEN])}
    kv_manager.step(seq_ids_and_new_tokens)

    # Check that we got new blocks
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3, 4]

    # Seq 1: Token gen 15 - 18
    for i, tok in enumerate([15, 16, 17, 18]):
        seq_ids_and_prompts = {seq_id_1: np.array([tok])}
        kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
        assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[
            0
        ] == [1, 5 + i + 1]
        seq_ids_and_new_tokens = {seq_id_1: np.array([FAKE_TOKEN])}
        kv_manager.step(seq_ids_and_new_tokens)

    # Seq 2: Claim
    seq_id_2 = 2
    initial_prompt_2 = [10, 11, 12, 13]
    kv_manager.external_claim([seq_id_2])

    # Seq 2: Prefill 10 - 13
    seq_ids_and_prompts = {seq_id_2: np.array(initial_prompt_2)}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        1,
        len(initial_prompt_2),
    ]
    seq_ids_and_new_tokens = {seq_id_2: np.array([FAKE_TOKEN])}
    kv_manager.step(seq_ids_and_new_tokens)

    # Check that we got cached blocks, except for last token in prompt
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:3] == [0, 1, 2]
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][3] != [3]

    # Seq 2: Token gen 14 - 17
    for i, tok in enumerate([14, 15, 99, 100]):
        seq_ids_and_prompts = {seq_id_2: np.array([tok])}
        kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
        assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[
            0
        ] == [
            1,
            len(initial_prompt_2) + i + 1,
        ]
        assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:4] == [0, 1, 2, 3]
        seq_ids_and_new_tokens = {seq_id_2: np.array([FAKE_TOKEN])}
        kv_manager.step(seq_ids_and_new_tokens)

    # Validate final trie
    assert kv_manager.prefix_cache.radix_trie.pretty_format() == [
        "[10, 11, 12]",
        "--[13]",
        "----[14]",
        "------[15]",
        "--------[16]",
        "----------[17]",
        "------------[18]",
        "--------[99]",
        "----------[100]",
    ]

    # first and second ce have 4 + 3 tokens
    assert kv_manager.prefix_cache.all_tokens == 7
    # second ce gets cache hit on 3 tokens
    assert kv_manager.prefix_cache.cache_hit_tokens == 3
    # cache hit rate is = 3 / 7
    assert kv_manager.cache_hit_rate > 0.42


@pytest.mark.asyncio
async def test_prefix_caching_with_repeating_prompt() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    available_blocks = 128

    # Try to assign and release more than 128 blocks.
    for i in range(1000):
        if i == 0:
            seq_id = 0
        else:
            seq_id = random.randint(1, kv_manager.max_batch_size - 1)
        kv_manager.external_claim([seq_id])
        # We reuse the same prompt each time, allowing for prefix sharing.
        prompt = np.array([100, 101, 102, 103, 104])
        seq_ids_and_prompts = {seq_id: prompt}
        _ = kv_manager.fetch(seq_ids_and_prompts)

        if seq_id == 0:
            # During first fetch, we do not get a cache hit so we use 5 blocks.
            available_blocks -= 5
        else:
            # During later fetches, we get a cache hit so we use 1 block.
            available_blocks -= 1
        assert len(kv_manager.available_blocks) == available_blocks

        seq_ids_and_new_tokens = {seq_id: np.array([FAKE_TOKEN])}
        kv_manager.step(seq_ids_and_new_tokens)

        if i != 0:
            # During later fetches, we will just release the block we wrote to
            # since a different block already exists for the same token.
            available_blocks += 1
        assert len(kv_manager.available_blocks) == available_blocks

        kv_manager.release(seq_id)

    assert kv_manager.cache_hit_rate > 0.99


@pytest.mark.asyncio
async def test_prefix_caching_with_no_release() -> None:
    np.random.seed(12345)

    kv_manager = create_paged_manager(num_blocks=128)

    # Try to assign and release more than 128 blocks.
    # We expect to run out of blocks here.
    with pytest.raises(RuntimeError):
        for i in range(1000):
            seq_id = i % kv_manager.max_batch_size
            kv_manager.external_claim([seq_id])
            prompt = gen_prompt(16)
            seq_ids_and_prompts = {seq_id: prompt}
            _ = kv_manager.fetch(seq_ids_and_prompts)
            seq_ids_and_new_tokens = {seq_id: np.array([FAKE_TOKEN])}
            kv_manager.step(seq_ids_and_new_tokens)

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
async def test_prefix_caching_with_random_prompts(page_size, num_steps) -> None:
    np.random.seed(12345)

    num_blocks = 128
    kv_manager = create_paged_manager(
        num_blocks=num_blocks, page_size=page_size
    )
    available_slots = num_blocks * page_size

    # Try to assign and release more than 128 blocks.
    for i in range(1000):
        seq_id = i % kv_manager.max_batch_size
        kv_manager.external_claim([seq_id])
        slots_used = 0
        # Picking random prompts.
        prompt_len = np.random.randint(1, 64)
        prompt = gen_prompt(prompt_len)
        # This fetch can trigger evictions from the tree.
        _ = kv_manager.fetch({seq_id: prompt}, num_steps=num_steps)
        new_tokens = gen_prompt(num_steps)
        kv_manager.step({seq_id: new_tokens})

        slots_used_in_curr_iter = prompt_len + num_steps - 1
        slots_used += slots_used_in_curr_iter

        # Perform some number of token generation steps.
        num_of_tg_steps = np.random.randint(0, 20)
        for _ in range(num_of_tg_steps):
            last_token = new_tokens[-1]
            prompt_len = np.random.randint(1, 4)

            # If this single sequence will exceed the total number of slots,
            # break out of the loop.
            slots_used_in_curr_iter = prompt_len + num_steps - 1
            if slots_used + slots_used_in_curr_iter > available_slots:
                break

            prompt = np.concatenate([[last_token], gen_prompt(prompt_len - 1)])
            # This fetch can trigger evictions from the tree.
            _ = kv_manager.fetch({seq_id: prompt}, num_steps=num_steps)
            new_tokens = gen_prompt(num_steps)
            kv_manager.step({seq_id: new_tokens})

            slots_used += slots_used_in_curr_iter

        kv_manager.release(seq_id)

    # Evict all blocks from the trie.
    assert kv_manager.prefix_cache is not None
    kv_manager.purge_prefix_cache()
    # Check that all blocks are either in the trie or available.
    assert len(kv_manager.available_blocks) == kv_manager.total_num_pages


@pytest.mark.asyncio
async def test_prefix_caching_with_num_steps_gt_1() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    # Reserve a slot in the KV cache manager.
    seq_id_1 = 1
    initial_prompt_1 = [10, 11, 12, 13, 14]
    kv_manager.external_claim([seq_id_1])

    # Seq 1: Prefill 10 - 14 and generate 15 - 17 in one pass
    seq_ids_and_prompts = {seq_id_1: np.array(initial_prompt_1)}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts, num_steps=3)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
        [1, 6],
        [1, 7],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([15, 16, 17])}
    kv_manager.step(seq_ids_and_new_tokens)

    # Seq 1: Token gen 18 - 19 in one pass
    seq_ids_and_prompts = {seq_id_1: np.array([17])}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts, num_steps=2)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 8],
        [1, 9],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([18, 19])}
    kv_manager.step(seq_ids_and_new_tokens)

    assert kv_manager.cache_hit_rate == 0.0


@pytest.mark.asyncio
async def test_prefix_caching_with_page_size_gt_1() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=2)

    # Reserve a slot in the KV cache manager.
    seq_id_1 = 1
    initial_prompt_1 = [10, 11, 12, 13, 14]
    kv_manager.external_claim([seq_id_1])

    # Seq 1: Prefill 10 - 14
    seq_ids_and_prompts = {seq_id_1: np.array(initial_prompt_1)}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([15])}
    kv_manager.step(seq_ids_and_new_tokens)

    # Seq 1: Token gen 15
    seq_ids_and_prompts = {seq_id_1: np.array([15])}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 6],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([16])}
    kv_manager.step(seq_ids_and_new_tokens)

    # Seq 1: Token gen 16
    seq_ids_and_prompts = {seq_id_1: np.array([16])}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 7],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([17])}
    kv_manager.step(seq_ids_and_new_tokens)

    assert kv_manager.cache_hit_rate == 0.0


@pytest.mark.asyncio
async def test_prefix_caching_with_page_size_gt_1_and_num_steps_gt_1() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=2)

    # Reserve a slot in the KV cache manager.
    seq_id_1 = 1
    initial_prompt_1 = [10, 11, 12, 13, 14]
    kv_manager.external_claim([seq_id_1])

    # Seq 1: Prefill 10 - 14 and generate 15 - 17 in one pass
    seq_ids_and_prompts = {seq_id_1: np.array(initial_prompt_1)}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts, num_steps=3)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 5],
        [1, 6],
        [1, 7],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([15, 16, 17])}
    kv_manager.step(seq_ids_and_new_tokens)

    # Seq 1: Token gen 18 - 19 in one pass
    seq_ids_and_prompts = {seq_id_1: np.array([17])}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts, num_steps=2)
    assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0] == [0, 1, 2, 3, 4]
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 8],
        [1, 9],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([18, 19])}
    kv_manager.step(seq_ids_and_new_tokens)


class FakeModel:
    """Create a fake model that can be used to test prefix caching."""

    def __init__(self, kv_manager: PagedKVCacheManager):
        self.page_size = kv_manager.page_size
        self.total_num_pages = kv_manager.total_num_pages
        # block_projections maps from block_id -> offset -> prefix tokens
        self.block_projections: dict[int, dict[int, np.ndarray]] = defaultdict(
            lambda: defaultdict(lambda: np.array([]))
        )
        self.seq_ids_and_all_tokens: dict[int, np.ndarray] = defaultdict(
            lambda: np.array([])
        )
        # Monkey patch the cow_strided_memcpy_graph to use our mock graph so that
        # when COW occurs, we can update the block_projections object.
        assert kv_manager.prefix_cache is not None
        kv_manager.prefix_cache.cow_strided_memcpy_graph = self.mock_cow_graph()

    def run(
        self,
        seq_ids_and_prompts: dict[int, np.ndarray],
        fetch_kv_tuple,
        num_steps: int,
        seq_ids_and_new_tokens: Optional[dict[int, np.ndarray]] = None,
    ) -> dict[int, np.ndarray]:
        """Given a batch and the fetch_kv_tuple, we `run` the model and check that
        the paged manager gave us valid blocks that contain the appropriate KV
        projections.

        This function returns the new tokens that were 'generated' by the model.
        """
        # generate some new tokens
        if seq_ids_and_new_tokens is None:
            seq_ids_and_new_tokens = {}
            for seq_id in seq_ids_and_prompts:
                new_toks = gen_prompt(num_steps)
                seq_ids_and_new_tokens[seq_id] = new_toks

        # update all tokens to contain the tokens which should have a KV
        # projection in the cache after this forward step
        for seq_id in seq_ids_and_prompts:
            self.seq_ids_and_all_tokens[seq_id] = np.concatenate(
                [
                    self.seq_ids_and_all_tokens[seq_id],
                    seq_ids_and_prompts[seq_id],
                    seq_ids_and_new_tokens[seq_id][:-1],
                ]
            )

        # read the blocks and cache lengths returned by the paged manager
        all_blocks = get_blocks_from_kv_tuple(fetch_kv_tuple[0])
        cache_lengths = get_cache_lengths_from_kv_tuple(fetch_kv_tuple[0])

        for batch_idx, seq_id in enumerate(seq_ids_and_prompts):
            blocks = all_blocks[batch_idx]
            cache_len = cache_lengths[batch_idx]
            tokens = self.seq_ids_and_all_tokens[seq_id]
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
                block_idx = idx // self.page_size
                block = blocks[block_idx]
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

        return seq_ids_and_new_tokens

    def mock_cow_graph(self) -> Any:
        class MockGraph:
            def __init__(
                self,
                block_projections: dict[int, dict[int, np.ndarray]],
                page_size: int,
            ):
                self.block_projections = block_projections
                self.page_size = page_size

            def execute(
                self, block_dst: int, block_src: int, num_tokens: int, *_
            ) -> None:
                # when the kv_manager attempts to execute the cow strided memcpy
                # graph, we intercept the call and update the block_projections
                # object instead.
                assert block_src in self.block_projections
                assert 0 < num_tokens < self.page_size
                for token in range(num_tokens):
                    self.block_projections[block_dst][token] = (
                        self.block_projections[block_src][token]
                    )

        return MockGraph(self.block_projections, self.page_size)


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
    seq_ids_and_new_tokens: dict[int, np.ndarray] = {}
    for b in range(num_batches):
        seq_ids_and_prompts: dict[int, np.ndarray] = {}
        for r in range(batch_size):
            seq_id = b * batch_size + r
            kv_manager.external_claim([seq_id])
            group_prefix = group_prefixes[seq_id % len(group_prefixes)]
            random_len = np.random.randint(0, 10)
            prompt = np.concatenate([group_prefix, gen_prompt(random_len)])
            seq_ids_and_prompts[seq_id] = prompt

        orig_seq_ids_and_prompts = seq_ids_and_prompts.copy()
        fetch_kv_tuple = kv_manager.fetch(
            seq_ids_and_prompts, num_steps=num_steps
        )
        seq_ids_and_new_tokens_batch = model.run(
            orig_seq_ids_and_prompts, fetch_kv_tuple, num_steps=num_steps
        )
        seq_ids_and_new_tokens |= seq_ids_and_new_tokens_batch
        kv_manager.step(seq_ids_and_new_tokens_batch)

    # Since our prompts have large grouped prefixes, we should have a high cache
    # hit rate.
    cache_hit_rate = kv_manager.cache_hit_rate
    if shared_prefix_len > 0:
        assert cache_hit_rate > 0.45

    # run TG on all requests for num_tg_steps steps
    # we terminate requests with probability 10% each iteration
    num_tg_steps = 100
    for _ in range(num_tg_steps):
        seq_ids_and_prompts = seq_ids_and_new_tokens

        for seq_id in seq_ids_and_prompts:
            seq_ids_and_prompts[seq_id] = seq_ids_and_prompts[seq_id][-1:]
            extended = gen_prompt(np.random.randint(0, 10))
            seq_ids_and_prompts[seq_id] = np.concatenate(
                [seq_ids_and_prompts[seq_id], extended]
            )

        orig_seq_ids_and_prompts = seq_ids_and_prompts.copy()
        fetch_kv_tuple = kv_manager.fetch(
            seq_ids_and_prompts, num_steps=num_steps
        )
        seq_ids_and_new_tokens_subset = model.run(
            orig_seq_ids_and_prompts, fetch_kv_tuple, num_steps=num_steps
        )

        kv_manager.step(seq_ids_and_new_tokens_subset)

        terminated_seq_ids = []
        for seq_id in list(seq_ids_and_new_tokens.keys()):
            # terminate requests with probability 10%
            if len(seq_ids_and_new_tokens) > 1 and np.random.rand() < 0.1:
                terminated_seq_ids.append(seq_id)
                kv_manager.release(seq_id)
                del seq_ids_and_new_tokens[seq_id]

    for seq_id in seq_ids_and_new_tokens:
        kv_manager.release(seq_id)


def run_forward(
    model: FakeModel,
    kv_manager: PagedKVCacheManager,
    seq_id: int,
    prompt: np.ndarray,
    next_tok: int,
    run_fetch: bool = True,
    run_step: bool = True,
) -> None:
    seq_ids_and_prompts = {seq_id: prompt}
    orig_seq_ids_and_prompts = seq_ids_and_prompts.copy()
    new_toks = {seq_id: np.array([next_tok])}
    if run_fetch:
        fetch_kv_tuple = kv_manager.fetch(seq_ids_and_prompts, num_steps=1)
        _ = model.run(
            orig_seq_ids_and_prompts,
            fetch_kv_tuple,
            num_steps=1,
            seq_ids_and_new_tokens=new_toks,
        )
    if run_step:
        kv_manager.step(new_toks)


@pytest.mark.asyncio
async def test_prefix_caching_chunked_prefill() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=3)
    model = FakeModel(kv_manager)
    assert kv_manager.prefix_cache is not None

    seq_id_1 = 1
    seq_id_2 = 2
    kv_manager.external_claim([seq_id_1, seq_id_2])

    prompt_1_part_1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    prompt_1_part_2 = np.array([18, 19, 20, 21, 22])

    prompt_2_part_1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    prompt_2_part_2 = np.array([16, 17, 18, 998, 999])

    run_forward(
        model, kv_manager, seq_id_1, prompt_1_part_1, prompt_1_part_2[0]
    )
    run_forward(
        model, kv_manager, seq_id_2, prompt_2_part_1, prompt_2_part_2[0]
    )
    run_forward(model, kv_manager, seq_id_1, prompt_1_part_2, FAKE_TOKEN)

    assert kv_manager.prefix_cache.radix_trie.pretty_format(
        print_blocks=True
    ) == [
        "[10, 11, 12, 13, 14, 15] : [0, 1]",
        "--[16, 17, 18, 19, 20, 21] : [2, 4]",
    ]

    # Make sure that we don't return block 2 for seq_id_2 since its last KV
    # projection differs.
    # block 2 holds projections for [..., 16, 17, 18]
    # seq_id_2 needs projections for [..., 16, 17, 16]
    run_forward(model, kv_manager, seq_id_2, prompt_2_part_2, FAKE_TOKEN)
    metadata = kv_manager.active_requests[seq_id_2]
    assert 2 not in metadata.blocks

    assert kv_manager.prefix_cache.radix_trie.pretty_format() == [
        "[10, 11, 12, 13, 14, 15]",
        "--[16, 17, 18, 19, 20, 21]",
        "--[16, 17, 16, 17, 18, 998]",
    ]

    assert kv_manager.prefix_cache.cache_hit_tokens == 6
    assert kv_manager.cache_hit_rate > 0.2


@pytest.mark.asyncio
async def test_prefix_caching_cow() -> None:
    kv_manager = create_paged_manager(num_blocks=128, page_size=3)
    model = FakeModel(kv_manager)
    assert kv_manager.prefix_cache is not None

    seq_id_1 = 1
    seq_id_2 = 2
    kv_manager.external_claim([seq_id_1, seq_id_2])

    prompt_1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    run_forward(model, kv_manager, seq_id_1, prompt_1, FAKE_TOKEN)
    prompt_2 = np.array([10, 11, 12, 13, 14, 25])
    run_forward(
        model,
        kv_manager,
        seq_id_2,
        prompt_2,
        FAKE_TOKEN,
        run_fetch=True,
        run_step=False,
    )
    assert kv_manager.active_requests[seq_id_2].cached_idx == 5
    assert kv_manager.prefix_cache.cow_count == 1
