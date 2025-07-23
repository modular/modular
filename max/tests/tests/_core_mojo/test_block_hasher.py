# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import hashlib

import numpy as np
from max._core_mojo import block_hasher
from pytest_benchmark.fixture import BenchmarkFixture


def test_block_hasher() -> None:
    block_size = 128
    num_tokens = 3000
    tokens = np.arange(num_tokens)

    hashes = block_hasher(tokens, block_size, hash("None"))

    assert isinstance(hashes, list)
    assert isinstance(hashes[0], int)
    assert len(hashes) == num_tokens // block_size

    # It is very unlikely (but not impossible) that a valid hasher will return 0
    # Usually a 0 is indicative of a bug of some sort.
    assert 0 not in hashes

    # It is unlikely (but not impossible) that a hasher will return the same value twice.
    # Usually a duplicate is indicative of a bug of some sort.
    seen = set()
    for h in hashes:
        assert h not in seen
        seen.add(h)


def mojo_block_hasher(tokens: np.ndarray, block_size: int) -> list[int]:
    return block_hasher(tokens, block_size, hash("None"))


def tensor_block_hasher(tokens: np.ndarray, block_size: int) -> list[int]:
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    # Initial hash seed value
    prev_hash = hash("None")

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        digest = hashlib.blake2b(block.tobytes()).hexdigest()
        pair_to_hash = (prev_hash, digest)
        curr_hash = hash(pair_to_hash)

        results.append(curr_hash)

    return results


def naive_block_hasher(tokens: np.ndarray, block_size: int) -> list[int]:
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    # Initial hash seed value
    prev_hash = hash("None")

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        pair_to_hash = (prev_hash, tuple(block))
        curr_hash = hash(pair_to_hash)
        results.append(curr_hash)
        prev_hash = curr_hash

    return results


def test_benchmark_mojo(benchmark: BenchmarkFixture) -> None:
    block_size = 128
    tokens = np.arange(30000)

    _ = benchmark.pedantic(
        mojo_block_hasher,
        args=(tokens, block_size),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )


def test_benchmark_tensor(benchmark: BenchmarkFixture) -> None:
    block_size = 128
    tokens = np.arange(30000)

    _ = benchmark.pedantic(
        tensor_block_hasher,
        args=(tokens, block_size),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )


def test_benchmark_naive(benchmark: BenchmarkFixture) -> None:
    block_size = 128
    tokens = np.arange(30000)

    _ = benchmark.pedantic(
        naive_block_hasher,
        args=(tokens, block_size),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )
