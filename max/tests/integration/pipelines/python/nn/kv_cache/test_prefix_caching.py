# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.kv_cache import (
    KVCacheManager,
    KVCacheParams,
    KVCacheStrategy,
    load_kv_manager,
)

FAKE_TOKEN = 999


def get_blocks_from_kv_tuple(kv_tuple) -> list[list[int]]:
    return kv_tuple[2].to(CPU()).to_numpy().tolist()


def get_uncommitted_and_committed_block_counts(
    kv_tuple,
) -> list[list[int]]:
    return kv_tuple[3].to(CPU()).to_numpy().tolist()


def create_paged_manager(num_blocks: int) -> KVCacheManager:
    # Setting kv_heads, head_dim, and num_layers to 1 so it is easy to compute
    # memory usage. Now we know each block is 1 byte.
    NUM_KV_HEADS = 1
    HEAD_DIM = 1
    NUM_LAYERS = 1
    PAGE_SIZE = 1

    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cache_strategy=KVCacheStrategy.PAGED,
        enable_prefix_caching=True,
    )

    session = InferenceSession()

    available_cache_memory = (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * num_blocks
        * kv_params.dtype.size_in_bytes
    )
    kv_manager = load_kv_manager(
        params=kv_params,
        max_cache_batch_size=9999,
        max_seq_len=512,
        num_layers=NUM_LAYERS,
        devices=[CPU()],
        session=session,
        available_cache_memory=available_cache_memory,
        page_size=PAGE_SIZE,
    )

    assert len(kv_manager.available_blocks) == num_blocks  # type: ignore
    return kv_manager


@pytest.mark.asyncio
async def test_prefix_caching() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    # Reserve a slot in the KV cache manager.
    seq_id_1 = 1
    initial_prompt_1 = [10, 11, 12, 13, 14]
    kv_manager.external_claim([seq_id_1])

    # Seq 1: Prefill 10 - 14
    seq_ids_and_prompts = {seq_id_1: np.array(initial_prompt_1)}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0])[0] == [
        len(initial_prompt_1),
        0,
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
        ] == [
            1,
            5 + i,
        ]
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
        len(initial_prompt_2) - 1,
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
            len(initial_prompt_2) + i,
        ]
        assert get_blocks_from_kv_tuple(kv_tuple_list[0])[0][:4] == [0, 1, 2, 3]
        seq_ids_and_new_tokens = {seq_id_2: np.array([FAKE_TOKEN])}
        kv_manager.step(seq_ids_and_new_tokens)

    # Validate final trie
    assert kv_manager.radix_trie.pretty_format() == [  # type: ignore
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


@pytest.mark.asyncio
async def test_prefix_caching_with_repeating_prompt() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    available_blocks = 128

    # Try to assign and release more than 128 blocks.
    for seq_id in range(1000):
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
        assert len(kv_manager.available_blocks) == available_blocks  # type: ignore

        seq_ids_and_new_tokens = {seq_id: np.array([FAKE_TOKEN])}
        kv_manager.step(seq_ids_and_new_tokens)

        if seq_id != 0:
            # During later fetches, we will just release the block we wrote to
            # since a different block already exists for the same token.
            available_blocks += 1
        assert len(kv_manager.available_blocks) == available_blocks  # type: ignore

        kv_manager.release(seq_id)


@pytest.mark.asyncio
async def test_prefix_caching_with_no_release() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    # Try to assign and release more than 128 blocks.
    # We expect to run out of blocks here.
    with pytest.raises(RuntimeError):
        for seq_id in range(1000):
            kv_manager.external_claim([seq_id])
            prompt = np.random.randint(0, 8, size=16)
            seq_ids_and_prompts = {seq_id: prompt}
            _ = kv_manager.fetch(seq_ids_and_prompts)
            seq_ids_and_new_tokens = {seq_id: np.array([FAKE_TOKEN])}
            kv_manager.step(seq_ids_and_new_tokens)

            # We intentionally do not release the sequence here!


@pytest.mark.asyncio
async def test_prefix_caching_with_random_prompt() -> None:
    kv_manager = create_paged_manager(num_blocks=128)

    # Try to assign and release more than 128 blocks.
    for seq_id in range(1000):
        kv_manager.external_claim([seq_id])
        # Picking random prompts.
        prompt_len = np.random.randint(1, 64)
        prompt = np.random.randint(0, 8, size=prompt_len)
        seq_ids_and_prompts = {seq_id: prompt}
        # This fetch can trigger evictions from the tree.
        _ = kv_manager.fetch(seq_ids_and_prompts)
        seq_ids_and_new_tokens = {seq_id: np.array([FAKE_TOKEN])}
        kv_manager.step(seq_ids_and_new_tokens)
        kv_manager.release(seq_id)

    # Evict all blocks from the trie.
    total_num_blocks = kv_manager.total_num_blocks  # type: ignore
    blocks_in_trie = kv_manager.radix_trie.evict_blocks(  # type: ignore
        desired_num_evicted=total_num_blocks
    )
    available_blocks = len(kv_manager.available_blocks)  # type: ignore

    # Check that all blocks are either in the trie or available.
    assert available_blocks + len(blocks_in_trie) == total_num_blocks


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
    seq_ids_and_new_tokens = {seq_id_1: np.array([15, 16, 17])}
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [5, 0],
        [1, 5],
        [1, 6],
    ]

    kv_manager.step(seq_ids_and_new_tokens)

    # Seq 1: Token gen 18 - 19 in one pass
    seq_ids_and_prompts = {seq_id_1: np.array([17])}
    kv_tuple_list = kv_manager.fetch(seq_ids_and_prompts, num_steps=2)
    assert get_uncommitted_and_committed_block_counts(kv_tuple_list[0]) == [
        [1, 7],
        [1, 8],
    ]

    seq_ids_and_new_tokens = {seq_id_1: np.array([18, 19])}
    kv_manager.step(seq_ids_and_new_tokens)
