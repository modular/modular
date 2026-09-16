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

"""Helper methods for working with the eager usage validator.

The eager usage validator protects against unsafe eager usages.

.. code-block:: python

    from max.pipelines.modeling.eager_validation import eager_validator

    with eager_validator.allow_eager(reason="input batching, off hot path"):
        ...
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, Literal

from max.experimental.validation import EagerUsageValidator, active_validator

from .config_enums import EagerValidatorMode

__all__ = ["Phase", "eager_usage_validator", "eager_validator"]

Phase = Literal["initialization", "execution"]
"""Which part of a pipeline's life a validation scope covers."""


def eager_usage_validator(
    mode: EagerValidatorMode, phase: Phase
) -> EagerUsageValidator:
    """Builds the validator guarding ``phase``, per the configured mode.

    Args:
        mode: The configured ``runtime.eager_usage_validator`` mode.
        phase: Which part of the pipeline's life the scope covers.

    Returns:
        A validator, disabled when ``mode`` excludes ``phase``.
    """
    enabled = mode is EagerValidatorMode.ENABLED or (
        mode is EagerValidatorMode.INIT_ONLY and phase == "initialization"
    )
    return EagerUsageValidator(enabled=enabled, label=phase)


class _EagerValidatorEscapeHatches:
    """The escape hatches for :class:`EagerUsageValidator`."""

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


eager_validator = _EagerValidatorEscapeHatches()
"""Use to opt in to the escape hatches for :class:`EagerUsageValidator`."""
