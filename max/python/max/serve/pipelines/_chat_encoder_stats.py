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

"""Publishes a tokenizer's chat-encoder outcomes as a ``maxserve`` counter.

The tokenizer lives in ``max.pipelines``, which cannot import the serve
telemetry stack, so it only counts. This is the serve half, the same split as
:mod:`max.serve.pipelines.preprocess_cache_stats`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from max.pipelines.modeling.types import ChatEncoderOutcomesProbe
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.serve")


class ChatEncoderOutcomesRecorder:
    """Records one tokenizer's chat-encoder outcomes, per request."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer_name = type(tokenizer).__name__
        self._take: Callable[[], Mapping[str, int]] | None = None
        if isinstance(tokenizer, ChatEncoderOutcomesProbe):
            self._take = tokenizer.take_chat_encoder_outcomes

    def record(self) -> None:
        """Publishes the outcomes counted since the last call."""
        if self._take is None:
            return
        try:
            for outcome, count in self._take().items():
                if count:
                    METRICS.tokenizer_chat_encoder_requests(count, outcome)
        except Exception:
            # Called from inside the request's re-raising try block, for a
            # metric nothing downstream needs: dropping it beats a 500 on every
            # request, and logging once beats a log line on every request.
            logger.warning(
                "%s.take_chat_encoder_outcomes() failed; publishing no"
                " chat-encoder metrics.",
                self._tokenizer_name,
                exc_info=True,
            )
            self._take = None
