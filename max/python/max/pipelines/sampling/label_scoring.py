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

"""Label scoring: full-vocabulary log-probabilities of fixed candidate tokens.

Decision-style serving (``/v1/decisions``) never samples. It asks, for each
request, how likely each of a few candidate label tokens is as the next token
after the prompt. The probability must be normalized over the *whole*
vocabulary (not just the candidates) so callers can read how much mass the
labels hold, and top-N log-probabilities cannot provide that for an arbitrary
candidate set. This module builds a small graph that applies ``log_softmax``
over the (unpadded) vocabulary and gathers the candidate columns on device, so
only ``batch x num_labels`` floats ever cross to the host.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.profiler import traced


def _build_label_scoring_graph(
    device: DeviceRef,
    in_dtype: DType,
    unpadded_vocab_size: int,
) -> Graph:
    """Builds the label-gather graph.

    Args:
        device: The device the graph runs on (the model device).
        in_dtype: The dtype of the model's next-token logits.
        unpadded_vocab_size: The tokenizer's token count. Logits for ids at or
            past it are masked to ``-inf`` before the normalization, so a
            padded ``lm_head`` cannot take probability mass.

    Returns:
        A graph mapping ``logits[batch, vocab]`` and
        ``label_ids[batch, num_labels]`` (int64) to float32
        ``label_log_probabilities[batch, num_labels]``.

    Raises:
        ValueError: If ``unpadded_vocab_size`` is not positive.
    """
    if unpadded_vocab_size <= 0:
        raise ValueError(
            f"unpadded_vocab_size must be positive, got {unpadded_vocab_size}"
        )
    logits_type = TensorType(in_dtype, ["batch", "vocab_size"], device=device)
    label_ids_type = TensorType(
        DType.int64, ["batch", "num_labels"], device=device
    )
    with Graph(
        "label_scoring",
        input_types=[logits_type, label_ids_type],
    ) as graph:
        logits = ops.cast(graph.inputs[0].tensor, DType.float32)
        label_ids = graph.inputs[1].tensor

        vocab_dim = logits.shape[1]
        token_ids = ops.range(
            0,
            vocab_dim,
            out_dim=vocab_dim,
            dtype=DType.int64,
            device=logits.device,
        )
        logits = ops.where(
            token_ids >= unpadded_vocab_size, float("-inf"), logits
        )
        logprobs = ops.logsoftmax(logits, axis=-1)
        gathered = ops.gather_nd(
            logprobs, ops.unsqueeze(label_ids, -1), batch_dims=1
        )
        graph.output(gathered)
    return graph


class LabelScorer:
    """Gathers candidate-token log-probabilities from next-token logits.

    Construction compiles the label-gather graph, which normalizes each logits
    row over the unpadded vocabulary and reads the log-probability at each
    candidate token id. It runs on the device the logits live on.

    .. code-block:: python

        scorer = LabelScorer(session, device, DType.bfloat16, vocab_size)
        # logits: a [2, vocab] Buffer; row 1 is not a scoring request.
        scores = scorer.score(logits, [[1234, 5678], None])
        # scores == [[logp_1234, logp_5678], None]

    Args:
        session: The inference session used to load the graph.
        device: The device the model logits live on.
        in_dtype: The dtype of the model's next-token logits.
        unpadded_vocab_size: The tokenizer's token count.

    Raises:
        ValueError: If ``unpadded_vocab_size`` is not positive.
    """

    def __init__(
        self,
        session: InferenceSession,
        device: Device,
        in_dtype: DType,
        unpadded_vocab_size: int,
    ) -> None:
        self._device = device
        self._unpadded_vocab_size = unpadded_vocab_size
        graph = _build_label_scoring_graph(
            DeviceRef.from_device(device), in_dtype, unpadded_vocab_size
        )
        self._model: Model = session.load(graph)

    @traced
    def score(
        self,
        logits: Buffer,
        label_token_ids: Sequence[Sequence[int] | None],
    ) -> list[list[float] | None]:
        """Scores candidate labels for each row of ``logits``.

        Launches the graph once for the whole batch, then copies the gathered
        scores to the host, which blocks until the launch finishes. Returns
        without launching when no row has candidates.

        Args:
            logits: ``[batch, vocab]`` next-token logits, one row per request.
            label_token_ids: Per-row candidate token ids, or ``None`` for rows
                that are not scoring requests (those rows get ``None``).

        Returns:
            Per-row full-vocabulary log-probabilities of the candidates, in
            candidate order, or ``None`` for non-scoring rows.

        Raises:
            ValueError: If ``logits`` has a different number of rows than
                ``label_token_ids``, or a candidate id is outside the
                vocabulary.
        """
        batch_size = len(label_token_ids)
        if int(logits.shape[0]) != batch_size:
            raise ValueError(
                f"logits batch {int(logits.shape[0])} != {batch_size} requests"
            )
        widest = max((len(ids) for ids in label_token_ids if ids), default=0)
        if widest == 0:
            return [None] * batch_size

        padded: npt.NDArray[np.int64] = np.zeros(
            (batch_size, widest), dtype=np.int64
        )
        for row, ids in enumerate(label_token_ids):
            if not ids:
                continue
            if not all(0 <= i < self._unpadded_vocab_size for i in ids):
                raise ValueError(
                    f"label token id out of vocabulary range in row {row}: "
                    f"{ids}"
                )
            padded[row, : len(ids)] = ids

        if logits.device != self._device:
            logits = logits.to(self._device)
        label_ids = Buffer.from_numpy(padded).to(self._device)
        (result,) = self._model(logits, label_ids)
        assert isinstance(result, Buffer)
        scores = result.to(CPU()).to_numpy()

        return [
            scores[row, : len(ids)].astype(np.float64).tolist() if ids else None
            for row, ids in enumerate(label_token_ids)
        ]
