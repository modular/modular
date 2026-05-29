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

# Hack-to-own'ed from
# https://github.com/AdamGleave/pytest-shard/blob/64610a08dac6b0511b6d51cf895d0e1040d162ad/pytest_shard/pytest_shard.py

"""Shard tests to support parallelism across multiple machines."""

from collections.abc import Iterable, Sequence
from typing import Any

from _pytest import nodes
from pytest import Config, Parser


def positive_int(x: Any) -> int:
    x = int(x)
    if x < 0:
        raise ValueError(f"Argument {x} must be positive")
    return x


def pytest_addoption(parser: Parser) -> None:
    """Add pytest-shard specific configuration parameters."""
    group = parser.getgroup("shard")
    group.addoption(
        "--shard-id",
        dest="shard_id",
        type=positive_int,
        default=0,
        help="Number of this shard.",
    )
    group.addoption(
        "--num-shards",
        dest="num_shards",
        type=positive_int,
        default=1,
        help="Total number of shards.",
    )


def pytest_report_collectionfinish(
    config: Config, items: Sequence[nodes.Node]
) -> str:
    """Log how many and, if verbose, which items are tested in this shard."""
    msg = f"Running {len(items)} items in this shard"
    if config.option.verbose > 0 and config.getoption("num_shards") > 1:
        msg += ": " + ", ".join([item.nodeid for item in items])
    return msg


def filter_items_by_shard(
    items: Iterable[nodes.Node], shard_id: int, num_shards: int
) -> Sequence[nodes.Node]:
    """Computes `items` that should be tested in `shard_id` out of `num_shards` total shards."""

    # 2 special markers to support
    # unique_shard - Gets its own dedicated shard
    # shard_group(name) - All tests with the same group name will be in a single dedicated shard
    # Everything else is round-robined into the remaining shards.

    # Step 1: Pull out anything marked with either
    unique: list[nodes.Node] = []
    groups: dict[str, list[nodes.Node]] = {}
    rest: list[nodes.Node] = []

    for item in items:
        print(item.name)
        for mark in item.iter_markers():
            if mark.name == "unique_shard":
                unique.append(item)
                break
            elif mark.name == "shard_group":
                group_name = mark.args[0]
                if group_name not in groups:
                    groups[group_name] = [item]
                else:
                    groups[group_name].append(item)
                break
        else:
            rest.append(item)

    min_shards = len(unique) + len(groups) + 1

    assert min_shards <= num_shards, (
        f"not enough shards for tests (minimum {min_shards} shards, {num_shards} specified)"
    )

    if shard_id < len(unique):
        return [unique[shard_id]]

    # Make the math easier
    shard_id -= len(unique)

    if shard_id < len(groups):
        return list(groups.values())[shard_id]

    shard_id -= len(groups)

    num_remaining_shards = num_shards - len(unique) - len(groups)

    # Use round-robin over hashing, for two reasons:
    # 1. More predictable
    # 2. Avoids some shards not getting any work (which results in an error).
    #    This allows a suite with N tests to always work with <=N shards.
    return rest[shard_id::num_remaining_shards]


def pytest_collection_modifyitems(
    config: Config, items: list[nodes.Node]
) -> None:
    """Mutate the collection to consist of just items to be tested in this shard."""
    shard_id = config.getoption("shard_id")
    shard_total = config.getoption("num_shards")
    if shard_id >= shard_total:
        raise ValueError(
            "shard_num = f{shard_num} must be less than shard_total = f{shard_total}"
        )

    items[:] = filter_items_by_shard(items, shard_id, shard_total)
