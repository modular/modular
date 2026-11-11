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
"""Sampling custom ops."""

from max.nn._graph_ops.sampling import (
    AcceptanceSampler,
    LogitPenalties,
    compute_synthetic_acceptance_base_rate,
    greedy_acceptance_sampler,
    stochastic_acceptance_sampler,
    synthetic_acceptance_sampler,
)

from .min_p import MinPSampler
from .rejection_sampler import RejectionSampler, RejectionSamplerWithResiduals

__all__ = [
    "AcceptanceSampler",
    "LogitPenalties",
    "MinPSampler",
    "RejectionSampler",
    "RejectionSamplerWithResiduals",
    "compute_synthetic_acceptance_base_rate",
    "greedy_acceptance_sampler",
    "stochastic_acceptance_sampler",
    "synthetic_acceptance_sampler",
]
