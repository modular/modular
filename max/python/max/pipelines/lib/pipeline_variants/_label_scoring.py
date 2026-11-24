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

"""Per-batch label scoring shared by the text generation pipelines.

A scoring request is an ordinary one-token generation whose context carries
``label_token_ids``. The pipelines call :meth:`BatchLabelScorer.score` on the
step's logits before the sampler runs and attach the result to
:attr:`TextGenerationOutput.label_log_probabilities`.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.context import TextContext, TextGenerationContextType
from max.pipelines.modeling.types import RequestID
from max.pipelines.sampling import LabelScorer

from ..interfaces import ModelOutputs
from ..utils import CompilationTimer


class BatchLabelScorer:
    """Scores the candidate labels of the scoring requests in a batch.

    The gather graph is compiled on the first batch that holds a scoring
    request, so servers that never see one pay nothing at startup.
    """

    def __init__(
        self,
        session: InferenceSession,
        device: Device,
        in_dtype: DType,
        vocab_size: int | None,
    ) -> None:
        """Creates a scorer; compilation is deferred to the first use.

        Args:
            session: The inference session that compiles the gather graph.
            device: The model device the logits live on.
            in_dtype: The dtype of the model's next-token logits.
            vocab_size: The tokenizer's token count, or ``None`` when the
                tokenizer cannot report it (scoring is then refused).
        """
        self._session = session
        self._device = device
        self._in_dtype = in_dtype
        self._vocab_size = vocab_size
        self._scorer: LabelScorer | None = None

    def score(
        self,
        model_outputs: ModelOutputs,
        flat_batch: Sequence[TextGenerationContextType],
    ) -> dict[RequestID, list[float]]:
        """Scores the candidate labels of every scoring request in the batch.

        Must run before the sampler: its penalty and min-token processors edit
        the logits buffer in place.

        Args:
            model_outputs: Outputs of this step's forward pass.
            flat_batch: The flattened context batch, matching the logits rows.

        Returns:
            Full-vocabulary label log-probabilities keyed by request id, for
            the scoring requests only. Empty when the batch has none.

        Raises:
            ValueError: If the batch has a scoring request but the model
                returns variable logits or the vocabulary size is unknown.
        """
        # Only the step that completes the prompt has the answer position as
        # its last row. A chunked-prefill step or a decode step does not.
        label_token_ids = [
            context.label_token_ids
            if isinstance(context, TextContext)
            and not context.tokens.actively_chunked
            and context.tokens.generated_length == 0
            else None
            for context in flat_batch
        ]
        if not any(label_token_ids):
            return {}
        if model_outputs.logit_offsets is not None:
            raise ValueError("label scoring does not support variable logits")
        logits = model_outputs.next_token_logits
        if logits is None:
            logits = model_outputs.logits
        scores = self._label_scorer().score(logits, label_token_ids)
        return {
            context.request_id: row_scores
            for context, row_scores in zip(flat_batch, scores, strict=True)
            if row_scores is not None
        }

    def _label_scorer(self) -> LabelScorer:
        if self._scorer is None:
            if self._vocab_size is None:
                raise ValueError(
                    "label scoring needs the model's vocabulary size"
                )
            with CompilationTimer("label scorer") as timer:
                self._scorer = LabelScorer(
                    self._session,
                    self._device,
                    self._in_dtype,
                    self._vocab_size,
                )
                timer.mark_build_complete()
        return self._scorer
