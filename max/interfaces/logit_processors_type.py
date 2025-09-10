# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Module for defining the `LogitsProcessor` type.

A callable for processing model logits, taking a `ProcessorInputs` object as
input. The `ProcessorInputs` dataclass contains the following fields:

    - logits: The model logits, a float32 tensor with shape `(N, vocab_size)`.
    `N` is the number of logits returned by the model.
    - context: The context containing the inputs to the model.

The function can update the logits in place if needed.

Examples:

.. code-block:: python

    # Example 1:
    def logits_processor(inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        ...

    # Example 2:
    class SuppressBeginToken:
        def __init__(self, tokens_to_suppress: list[int], steps: int):
            self.tokens_to_suppress = tokens_to_suppress
            self.steps = steps
            self.step_counter = 0

        def __call__(self, inputs: ProcessorInputs) -> None:
            logits = inputs.logits
            if self.step_counter < self.steps:
                logits[-1, self.tokens_to_suppress] = -10000
                self.step_counter += 1
    processor = SuppressBeginToken(tokens_to_suppress=[0, 1], steps=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import max.driver as md
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from .context import InputContext


@dataclass
class ProcessorInputs:
    logits: md.Tensor
    context: InputContext


LogitsProcessor: TypeAlias = Callable[[ProcessorInputs], None]
