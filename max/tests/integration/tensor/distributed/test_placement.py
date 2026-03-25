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
"""Tests for Replicated, Sharded, Partial, and ReduceOp placement types."""

from __future__ import annotations

import dataclasses

import pytest
from max.experimental.distributed import (
    Partial,
    ReduceOp,
    Replicated,
    Sharded,
)


class TestReplicated:
    def test_repr(self) -> None:
        assert repr(Replicated()) == "Replicated()"

    def test_equality(self) -> None:
        assert Replicated() == Replicated()

    def test_frozen(self) -> None:
        r = Replicated()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.dummy = 1  # type: ignore[attr-defined]


class TestSharded:
    def test_repr(self) -> None:
        assert repr(Sharded(axis=0)) == "Sharded(axis=0)"

    def test_equality(self) -> None:
        assert Sharded(axis=0) == Sharded(axis=0)
        assert Sharded(axis=0) != Sharded(axis=1)

    def test_frozen(self) -> None:
        s = Sharded(axis=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.axis = 1  # type: ignore[misc]


class TestPartial:
    def test_repr(self) -> None:
        assert repr(Partial()) == "Partial(reduce_op='sum')"

    def test_default_reduce_op(self) -> None:
        assert Partial().reduce_op == ReduceOp.SUM

    def test_equality(self) -> None:
        assert Partial() == Partial(reduce_op=ReduceOp.SUM)

    def test_frozen(self) -> None:
        p = Partial()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.reduce_op = ReduceOp.SUM  # type: ignore[misc]


class TestReduceOp:
    def test_sum_value(self) -> None:
        assert ReduceOp.SUM.value == "sum"

    def test_is_string_enum(self) -> None:
        assert isinstance(ReduceOp.SUM, str)
