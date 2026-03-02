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
"""Tests for max.nn.Sequential."""

from __future__ import annotations

import re

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Module, Sequential, module_dataclass


@module_dataclass
class TestModule(Module[[Tensor], Tensor]):
    a: int

    def forward(self, x: Tensor) -> Tensor:
        return x + self.a


def strip_margin(s: str, margin_character: str = "|"):  # noqa: ANN201
    return re.sub(
        rf"^\s*\{margin_character}", "", s.strip(), flags=re.MULTILINE
    )


def test_repr() -> None:
    s = Sequential(
        Sequential(
            TestModule(1),
            TestModule(2),
        ),
        Sequential(
            TestModule(3),
            TestModule(4),
        ),
    )
    expected_repr = strip_margin("""
    |Sequential(
    |    Sequential(TestModule(a=1), TestModule(a=2)),
    |    Sequential(TestModule(a=3), TestModule(a=4))
    |)
    """)
    assert expected_repr == repr(s)


def test_children() -> None:
    c1 = Sequential(TestModule(1), TestModule(2))
    c2 = Sequential(TestModule(3), TestModule(4))
    s = Sequential(c1, c2)
    assert dict(s.children) == {"0": c1, "1": c2}


def test_descendants() -> None:
    t1 = TestModule(1)
    t2 = TestModule(2)
    t3 = TestModule(3)
    t4 = TestModule(4)
    c1 = Sequential(t1, t2)
    c2 = Sequential(t3, t4)
    s = Sequential(c1, c2)
    assert dict(s.descendants) == {
        "0": c1,
        "1": c2,
        "0.0": t1,
        "0.1": t2,
        "1.0": t3,
        "1.1": t4,
    }


def test_call() -> None:
    s = Sequential(
        Sequential(
            TestModule(1),
            TestModule(2),
        ),
        Sequential(
            TestModule(3),
            TestModule(4),
        ),
    )
    t = Tensor.constant(0)
    assert s(t).item() == 10
