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
"""Configuration types for MAX speculative decoding.

Exposes :class:`SpeculativeConfig`, which controls the speculative decoding
method, the number of draft tokens per step, and the rejection sampling
strategy used to verify drafts.
"""

from __future__ import annotations

import logging
from typing import Literal

from max.config import ConfigFileModel
from pydantic import Field, field_validator

logger = logging.getLogger("max.pipelines")

SpeculativeMethod = Literal["standalone", "eagle", "mtp"]
"""The supported methods for speculative decoding."""

RejectionSamplingStrategy = Literal[
    "greedy", "residual", "typical-acceptance", "logit-comparison"
]
"""The supported strategies for verifying drafted tokens against the target.

- ``greedy``: accepts a drafted token only when it matches the target's
  argmax at that position.
- ``residual``: samples from the residual distribution after subtracting
  the draft's probability, the standard rejection-sampling rule for
  matching the target distribution.
- ``typical-acceptance``: accepts drafted tokens that fall within the
  target's typical set, trading a small distributional mismatch for higher
  acceptance rates. Default for ``eagle`` and ``mtp``.
- ``logit-comparison``: compares target and draft logits directly to decide
  acceptance.
"""


class SpeculativeConfig(ConfigFileModel):
    """Configures speculative decoding for a pipeline.

    Speculative decoding accelerates token generation by having a small
    draft step propose several candidate tokens that the larger target
    verifies in one forward pass. This class selects the method
    (:attr:`speculative_method`), how many tokens to draft per step
    (:attr:`num_speculative_tokens`), and how the target verifies them
    (:attr:`rejection_sampling_strategy`).

    The CLI surfaces these fields as ``--speculative-method``,
    ``--num-speculative-tokens``, ``--rejection-sampling-strategy``, and
    ``--synthetic-acceptance-rate``. Construct the config directly when
    configuring a pipeline programmatically:

    .. code-block:: python

        from max.pipelines import SpeculativeConfig

        spec = SpeculativeConfig(
            speculative_method="eagle",
            num_speculative_tokens=3,
        )
    """

    speculative_method: SpeculativeMethod | None = Field(
        default=None, description="The speculative decoding method to use."
    )
    """The speculative decoding method to use.

    One of ``"standalone"``, ``"eagle"``, or ``"mtp"``. When ``None``,
    speculative decoding is disabled.
    """

    num_speculative_tokens: int = Field(
        default=2, description="The number of speculative tokens."
    )
    """The number of tokens the draft proposes per verification pass.

    Defaults to ``2``. Larger values can raise the average draft
    acceptance length and peak speedup, but they may hurt acceptance
    rates at later positions and increase kernel latencies from the
    additional tokens.
    """

    rejection_sampling_strategy: RejectionSamplingStrategy | None = Field(
        default=None,
        description=(
            "Rejection sampling strategy for verifying draft tokens. "
            "Defaults to ``typical-acceptance`` for ``eagle``/``mtp`` and "
            "``residual`` for ``standalone``."
        ),
    )
    """The rejection sampling strategy used to verify drafted tokens.

    When ``None``, defaults to ``"typical-acceptance"`` for ``eagle`` and
    ``mtp`` and ``"residual"`` for ``standalone``.
    """

    synthetic_acceptance_rate: float | None = Field(
        default=None,
        description=(
            "Synthetic acceptance rate for benchmarking (``0.0`` to ``1.0``). "
            "When set, the rejection sampler bypasses the real "
            "draft/target comparison and accepts each draft position "
            "with a calibrated probability so the mean joint acceptance "
            "across ``num_speculative_tokens`` positions matches this value."
        ),
    )
    """A benchmarking-only override that accepts drafts with a calibrated
    probability, ignoring real logits.

    Must be between ``0.0`` and ``1.0``. When set, each draft position is
    accepted with a probability calibrated so that the mean joint
    acceptance across :attr:`num_speculative_tokens` positions matches this
    value. Use it to model hypothetical speedups without changing the draft
    model; leave unset for real serving.
    """

    @field_validator("synthetic_acceptance_rate")
    @classmethod
    def _validate_synthetic_acceptance_rate(
        cls, v: float | None
    ) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                "synthetic_acceptance_rate must be between 0.0 and 1.0,"
                f" got {v}"
            )
        return v

    _config_file_section_name: str = "speculative_config"

    def is_eagle(self) -> bool:
        """Returns whether the configured method is EAGLE.

        EAGLE drafts share the target's embedding and ``lm_head`` layers
        and read the target's hidden states.
        """
        return self.speculative_method == "eagle"

    def is_standalone(self) -> bool:
        """Returns whether the configured method is a standalone draft model."""
        return self.speculative_method == "standalone"

    def is_mtp(self) -> bool:
        """Returns whether the configured method is multi-token prediction (MTP)."""
        return self.speculative_method == "mtp"

    def uses_greedy_rejection(self) -> bool:
        """Returns whether the ``"greedy"`` rejection sampling strategy is selected."""
        return self.rejection_sampling_strategy == "greedy"

    def uses_typical_acceptance(self) -> bool:
        """Returns whether the ``"typical-acceptance"`` strategy is selected."""
        return self.rejection_sampling_strategy == "typical-acceptance"

    def uses_logit_comparison(self) -> bool:
        """Returns whether the ``"logit-comparison"`` strategy is selected."""
        return self.rejection_sampling_strategy == "logit-comparison"
