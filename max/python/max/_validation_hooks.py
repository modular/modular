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

"""Eager usage validation hook trackers.

Breaks the circular dependency between the validator and the methods that
require safeguards. Dependency-free on purpose: ``max.experimental`` builds on
``max.engine``, so a hook module inside the former is out of the latter's
reach.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from max.driver import Device
    from max.engine import Model
    from max.experimental.tensor import Tensor
    from max.experimental.validation import EagerUsageValidator

VALIDATORS: ContextVar[tuple[EagerUsageValidator, ...]] = ContextVar(
    "max_validators", default=()
)
"""Validators in scope for the calling context."""


def active_validator() -> EagerUsageValidator | None:
    """Returns the innermost validator in scope, or ``None``."""
    validators = VALIDATORS.get()
    return validators[-1] if validators else None


@contextlib.contextmanager
def register(validator: EagerUsageValidator) -> Iterator[None]:
    """Registers the validator."""
    token = VALIDATORS.set((*VALIDATORS.get(), validator))
    try:
        yield
    finally:
        VALIDATORS.reset(token)


def device_transfer(api: str, tensor: Tensor, dest: Device) -> None:
    """Reports a transfer that may synchronize an accelerator with the host."""
    if validator := active_validator():
        validator._device_transfer(api, tensor, dest)


def compiled_call(model: Model, args: Any = (), outputs: Any = ()) -> None:
    """Reports a call into a compiled graph.

    Args:
        model: The engine model that ran. Identity matters as well as the
            name: several ModuleV3 graphs compiled from a method called
            ``forward`` all carry the name ``forward``.
        args: The buffers the call consumed.
        outputs: The buffers it produced.
    """
    if validator := active_validator():
        validator._compiled_call(model, args, outputs)


def mark_used(values: Any) -> None:
    """Reports values consumed by a call the validator cannot otherwise see.

    A compiled output is only excused by evidence that something reads it, and
    the validator sees a read only where a hook reports one. Call this from a
    consumer that is not itself worth reporting, so that a compiled output
    feeding it does not look abandoned.
    """
    if validator := active_validator():
        validator._mark_used(values)
