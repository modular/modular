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
"""DeepSeek-V4.1-Flash computes the log probabilities serve asks for."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.deepseekV41.model import (
    DeepseekV41Inputs,
    DeepseekV41Model,
)
from max.pipelines.architectures.unified_dspark_deepseekV41.model import (
    UnifiedDSparkDeepseekV41Model,
)
from max.pipelines.lib import ModelOutputs
from max.pipelines.lib.log_probabilities import (
    LogProbabilitiesMixin,
    log_probabilities_ragged_graph,
)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


@pytest.mark.parametrize(
    "model_cls", [DeepseekV41Model, UnifiedDSparkDeepseekV41Model]
)
def test_overrides_compute_log_probabilities(model_cls: type) -> None:
    """Both models inherit ``LogProbabilitiesMixin``'s implementation."""
    assert issubclass(model_cls, LogProbabilitiesMixin)
    assert (
        model_cls.compute_log_probabilities
        is LogProbabilitiesMixin.compute_log_probabilities
    )


@pytest.fixture(scope="module")
def device() -> Device:
    return CPU()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def logprobs_model(device: Device, session: InferenceSession) -> Model:
    """The log-probabilities graph, compiled once for the module."""
    return session.load(
        log_probabilities_ragged_graph(DeviceRef.from_device(device), levels=3)
    )


def _pipeline_model(device: Device, logprobs_model: Model) -> DeepseekV41Model:
    """A ``DeepseekV41Model`` with only the state log probabilities read."""
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    model.return_logits = ReturnLogits.LAST_TOKEN
    model._logprobs_device = device
    model._logprobs_model = logprobs_model
    return model


def test_compute_log_probabilities_on_last_token_logits(
    device: Device, session: InferenceSession, logprobs_model: Model
) -> None:
    """The graph's last-token float32 logits feed the logprobs kernel."""
    model = _pipeline_model(device, logprobs_model)

    rng = np.random.default_rng(0)
    vocab = 64
    # A 3-token prefill row and a 1-token decode row.
    offsets = np.array([0, 3, 4], dtype=np.uint32)
    tokens = np.array([5, 9, 11, 13], dtype=np.int64)
    logits = rng.standard_normal((2, vocab)).astype(np.float32) * 4
    sampled = logits.argmax(axis=-1).astype(np.int64)

    inputs = DeepseekV41Inputs(
        tokens=Buffer.from_numpy(tokens),
        input_row_offsets=Buffer.from_numpy(offsets),
        return_n_logits=Buffer.from_numpy(np.array([1], dtype=np.int64)),
    )
    logits_buf = Buffer.from_numpy(logits)
    outputs = ModelOutputs(logits=logits_buf, next_token_logits=logits_buf)

    result = model.compute_log_probabilities(
        session,
        inputs,
        outputs,
        Buffer.from_numpy(sampled),
        batch_top_n=[5, 1],
        batch_echo=[False, False],
    )

    expected = _log_softmax(logits.astype(np.float64))
    for row, (lp, top_n) in enumerate(zip(result, [5, 1], strict=True)):
        assert lp is not None
        assert lp.sampled_token_ids == [int(sampled[row])]
        np.testing.assert_allclose(
            lp.token_log_probabilities,
            [expected[row, sampled[row]]],
            rtol=1e-5,
            atol=1e-5,
        )
        (top,) = lp.top_log_probabilities
        assert len(top) == top_n
        # Greedy: the sampled token is the top-1 and its logprob the max.
        assert max(top, key=lambda t: top[t]) == int(sampled[row])
        want = np.argsort(-expected[row])[:top_n]
        assert sorted(top) == sorted(int(t) for t in want)
        for token, value in top.items():
            np.testing.assert_allclose(
                value, expected[row, token], rtol=1e-5, atol=1e-5
            )
        assert sum(np.exp(v) for v in top.values()) <= 1.0 + 1e-6


def test_zero_top_n_skips_the_row(
    device: Device, session: InferenceSession, logprobs_model: Model
) -> None:
    model = _pipeline_model(device, logprobs_model)
    logits = np.zeros((1, 8), dtype=np.float32)
    logits_buf = Buffer.from_numpy(logits)
    result = model.compute_log_probabilities(
        session,
        DeepseekV41Inputs(
            tokens=Buffer.from_numpy(np.array([1], dtype=np.int64)),
            input_row_offsets=Buffer.from_numpy(
                np.array([0, 1], dtype=np.uint32)
            ),
            return_n_logits=Buffer.from_numpy(np.array([1], dtype=np.int64)),
        ),
        ModelOutputs(logits=logits_buf, next_token_logits=logits_buf),
        Buffer.from_numpy(np.array([0], dtype=np.int64)),
        batch_top_n=[0],
        batch_echo=[False],
    )
    assert result == [None]
