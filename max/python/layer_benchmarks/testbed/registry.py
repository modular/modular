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

"""Harness registry: maps names to harness classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from testbed.harness import LayerTestHarness

HARNESS_REGISTRY: dict[str, type[LayerTestHarness[Any, Any, Any]]] = {}


def register_harness(
    name: str,
) -> Callable[
    [type[LayerTestHarness[Any, Any, Any]]],
    type[LayerTestHarness[Any, Any, Any]],
]:
    """Decorator that registers a LayerTestHarness subclass by name.

    Usage::

        @register_harness("attention_with_rope")
        class AttentionWithRopeHarness(LayerTestHarness):
            ...
    """

    def decorator(
        cls: type[LayerTestHarness[Any, Any, Any]],
    ) -> type[LayerTestHarness[Any, Any, Any]]:
        if name in HARNESS_REGISTRY:
            raise ValueError(
                f"Harness '{name}' already registered "
                f"(existing: {HARNESS_REGISTRY[name].__name__}, "
                f"new: {cls.__name__})"
            )
        HARNESS_REGISTRY[name] = cls
        return cls

    return decorator
