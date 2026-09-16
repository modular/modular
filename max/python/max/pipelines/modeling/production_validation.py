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

"""The escape hatches a pipeline marks a deliberate footgun with.

.. code-block:: python

    from max.pipelines.modeling.production_validation import prod_validator

    with prod_validator.allow_eager(reason="input batching, off hot path"):
        ...
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from max.experimental.validation import active_validator

__all__ = ["prod_validator"]


class _ProdValidatorEscapeHatches:
    """The escape hatches for :class:`ProductionValidator`."""

    @contextlib.contextmanager
    def allow_eager(self, *, reason: str) -> Iterator[None]:
        """Opts in to eager execution for the duration of the block."""
        if validator := active_validator():
            with validator.allow_eager(reason=reason):
                yield
        else:
            yield

    @contextlib.contextmanager
    def graph_break(self, *, reason: str) -> Iterator[None]:
        """Marks the block as an expected boundary between compiled graphs."""
        if validator := active_validator():
            with validator.graph_break(reason=reason):
                yield
        else:
            yield

    @contextlib.contextmanager
    def allow_device_transfer(self, *, reason: str) -> Iterator[None]:
        """Opts in to device-to-host transfers for the duration of the block."""
        if validator := active_validator():
            with validator.allow_device_transfer(reason=reason):
                yield
        else:
            yield

    def discard_output(self, output: Any, *, reason: str) -> None:
        """Marks compiled-model outputs as intentionally unread."""
        if validator := active_validator():
            validator.discard_output(output, reason=reason)


prod_validator = _ProdValidatorEscapeHatches()
"""Use to opt in to the escape hatches for :class:`ProductionValidator`."""
