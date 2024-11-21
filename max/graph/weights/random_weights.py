# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Module for random weights implementing with the Weights protocol."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from max.dtype import DType

from ..type import ShapeLike
from ..weight import Weight


@dataclass
class RandomWeights:
    """A class that mimics a Weights implementation with a checkpoint file.

    Unlike checkpoint-backed weights, this doesn't carry a mapping from weight
    names to mmap'ed numpy arrays.
    Rather, when .allocate is called, this generates a backing NumPy array of
    the desired tensor spec on the fly and stores it.
    This is useful for generating weights from testing and using them in
    subcomponents that expect a weights implementation backed by a checkpoint.
    """

    _allocated: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    _prefix: str = ""

    def __getattr__(self, attr: str) -> RandomWeights:
        """Append to the weight's name."""
        self._prefix = f"{self._prefix}.{attr}" if self._prefix else str(attr)
        return self

    def __getitem__(self, idx: int | str) -> RandomWeights:
        return self.__getattr__(str(idx))

    def allocate(self, dtype: DType, shape: ShapeLike) -> Weight:
        """Creates a Weight that can be added to a graph."""
        if dtype == DType.bfloat16:
            # Only try to import torch if it is actually needed.
            try:
                import torch  # type: ignore
            except ImportError:
                torch = None

            if torch is None:
                msg = "must have torch to use RandomWeights with bfloat16"
                raise ValueError(msg)

            self._allocated[self._prefix] = torch.rand(*shape).to(
                torch.bfloat16
            )
        else:
            self._allocated[self._prefix] = np.array(
                np.random.randn(*shape)
            ).astype(dtype.to_numpy())

        return Weight(name=self._prefix, dtype=dtype, shape=shape)

    @property
    def allocated_weights(self) -> dict[str, np.ndarray]:
        """Gets the values of all weights that were allocated previously."""
        return self._allocated
