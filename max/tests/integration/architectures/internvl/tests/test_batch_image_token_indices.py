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

"""Unit tests for InternVL _batch_image_token_indices logic.

These tests verify that image token indices are correctly batched when
mixing CE (prefill) and TG (decode) requests in the same batch, as
happens with in-flight batching enabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from max.pipelines.architectures.internvl.model import InternVLModel


def _make_mock_context(
    *,
    active_length: int,
    needs_vision_encoding: bool,
    image_token_indices: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock TextAndVisionContext with the specified properties."""
    ctx = MagicMock()
    ctx.tokens.active_length = active_length
    ctx.needs_vision_encoding = needs_vision_encoding
    if image_token_indices is not None:
        ctx.extra_model_args = {"image_token_indices": image_token_indices}
    else:
        ctx.extra_model_args = {}
    return ctx


def _make_mock_model() -> InternVLModel:
    """Create a minimal mock InternVLModel with CPU device for testing."""
    from max.driver import CPU

    model = MagicMock(spec=InternVLModel)
    model.devices = [CPU()]
    model._batch_image_token_indices = (
        InternVLModel._batch_image_token_indices.__get__(model)
    )
    return model


def test_batch_indices_single_vision_request() -> None:
    """A single CE request with vision should return its indices."""
    model = _make_mock_model()
    ctx = _make_mock_context(
        active_length=10,
        needs_vision_encoding=True,
        image_token_indices=np.array([2, 3, 4, 5], dtype=np.int32),
    )
    result = model._batch_image_token_indices([ctx])
    assert result is not None
    np.testing.assert_array_equal(result[0].to_numpy(), [2, 3, 4, 5])


def test_batch_indices_two_vision_requests() -> None:
    """Two CE requests with vision should concatenate indices with offset."""
    model = _make_mock_model()
    ctx1 = _make_mock_context(
        active_length=10,
        needs_vision_encoding=True,
        image_token_indices=np.array([2, 3, 4, 5], dtype=np.int32),
    )
    ctx2 = _make_mock_context(
        active_length=8,
        needs_vision_encoding=True,
        image_token_indices=np.array([1, 2, 3], dtype=np.int32),
    )
    result = model._batch_image_token_indices([ctx1, ctx2])
    assert result is not None
    # ctx2's indices should be offset by ctx1's active_length (10)
    np.testing.assert_array_equal(
        result[0].to_numpy(), [2, 3, 4, 5, 11, 12, 13]
    )


def test_batch_indices_text_only_request() -> None:
    """A text-only request should return None."""
    model = _make_mock_model()
    ctx = _make_mock_context(
        active_length=10,
        needs_vision_encoding=False,
    )
    result = model._batch_image_token_indices([ctx])
    assert result is None


def test_batch_indices_mixed_ce_and_tg_with_stale_indices() -> None:
    """In-flight batching: CE request with vision + TG request with stale indices.

    This is the key regression test. When in-flight batching mixes a new CE
    request (needs vision encoding) with a TG request that previously had
    images (still has image_token_indices in extra_model_args but
    needs_vision_encoding=False), only the CE request's indices should be
    included.

    Before the fix, both sets of indices were included, causing a mismatch
    between vision embeddings count and indices count (e.g. 1792 vs 3584).
    """
    model = _make_mock_model()

    # CE request: new prefill with vision
    ce_ctx = _make_mock_context(
        active_length=100,
        needs_vision_encoding=True,
        image_token_indices=np.array([10, 11, 12, 13], dtype=np.int32),
    )

    # TG request: decode step, previously had vision, stale indices remain
    tg_ctx = _make_mock_context(
        active_length=1,  # single decode token
        needs_vision_encoding=False,  # images already encoded
        image_token_indices=np.array([10, 11, 12, 13], dtype=np.int32),
    )

    result = model._batch_image_token_indices([ce_ctx, tg_ctx])
    assert result is not None
    # Only CE request's indices should be present, TG's stale indices excluded
    np.testing.assert_array_equal(result[0].to_numpy(), [10, 11, 12, 13])


def test_batch_indices_tg_only_with_stale_indices() -> None:
    """All TG requests with stale indices should return None."""
    model = _make_mock_model()

    tg1 = _make_mock_context(
        active_length=1,
        needs_vision_encoding=False,
        image_token_indices=np.array([5, 6, 7], dtype=np.int32),
    )
    tg2 = _make_mock_context(
        active_length=1,
        needs_vision_encoding=False,
        image_token_indices=np.array([3, 4], dtype=np.int32),
    )

    result = model._batch_image_token_indices([tg1, tg2])
    assert result is None


def test_batch_indices_ce_vision_with_text_only_request() -> None:
    """CE vision request + text-only request (no stale indices)."""
    model = _make_mock_model()

    ce_ctx = _make_mock_context(
        active_length=50,
        needs_vision_encoding=True,
        image_token_indices=np.array([5, 6, 7, 8], dtype=np.int32),
    )
    text_ctx = _make_mock_context(
        active_length=30,
        needs_vision_encoding=False,
    )

    result = model._batch_image_token_indices([ce_ctx, text_ctx])
    assert result is not None
    np.testing.assert_array_equal(result[0].to_numpy(), [5, 6, 7, 8])
