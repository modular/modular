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
"""The gemma4 pipeline model reports log probabilities.

``LogProbabilitiesMixin`` is what turns a ``max_logprobs`` request into
per-token log-probability outputs. The capability a request route reads is
not the mixin's presence in the MRO but the class's own
``compute_log_probabilities``: the OpenAI route
(``_computes_log_probabilities``) and ``generate`` both compare it by
identity against ``PipelineModel.compute_log_probabilities``, the base stub
that raises ``NotImplementedError``.
"""

from __future__ import annotations

from max.pipelines.architectures.diffusion_gemma.model import (
    DiffusionGemmaForBlockDiffusionModel,
)
from max.pipelines.architectures.gemma4.model import Gemma3_MultiModalModel
from max.pipelines.lib import PipelineModel
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin


def test_pipeline_model_mixes_in_log_probabilities() -> None:
    assert LogProbabilitiesMixin in Gemma3_MultiModalModel.__mro__


def test_pipeline_model_claims_log_probabilities() -> None:
    base = PipelineModel.compute_log_probabilities
    assert LogProbabilitiesMixin.compute_log_probabilities is not base
    # The claim is only real if the class actually resolves the mixin's
    # implementation: a stub override that raises, or any other class in
    # front of the mixin in the MRO, keeps the mixin in ``__mro__`` while
    # reporting no log probabilities.
    assert Gemma3_MultiModalModel.compute_log_probabilities is (
        LogProbabilitiesMixin.compute_log_probabilities
    )
    # The predicate the OpenAI route and ``generate`` evaluate: anything but
    # the base implementation means "this model computes log probabilities".
    assert Gemma3_MultiModalModel.compute_log_probabilities is not base


def test_block_diffusion_model_does_not_claim_log_probabilities() -> None:
    # The block-diffusion model subclasses the gemma4 pipeline model, so the
    # mixin does enter its MRO -- but its generation loop never computes log
    # probabilities, and a model that claimed otherwise would make the
    # OpenAI route reject an ``echo`` + ``logprobs`` request that omits
    # ``--enable-echo``, on a model that could not score it anyway.
    assert LogProbabilitiesMixin in DiffusionGemmaForBlockDiffusionModel.__mro__
    assert (
        DiffusionGemmaForBlockDiffusionModel.compute_log_probabilities
        is PipelineModel.compute_log_probabilities
    )
