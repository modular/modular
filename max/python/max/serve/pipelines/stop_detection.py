# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


class StopDetector:
    """
    Utility to detect a stop sequence in a continuation
    """

    def __init__(self, stop: str | list[str] | None) -> None:
        self.continuation_tail = ""
        self.stop: list[str]

        # Clean up self.stop: List[str]
        if stop is None:
            self.stop = []
        else:
            if isinstance(stop, str):
                self.stop = [stop]
            else:
                self.stop = list(stop)

        if len(self.stop) > 0:
            self._max_stop_length = max(map(len, self.stop))

    def step(self, next_token_decoded: str) -> str | None:
        """
        Register an incremental decoded str into the continuation buffer.
        If a stop sequence is detected, return the matched sequence. Else,
        return None.
        """
        if len(self.stop) == 0:
            return None

        self.continuation_tail += next_token_decoded

        # Magic number; just don't proc this constantly
        if len(self.continuation_tail) > 8 * self._max_stop_length:
            self.continuation_tail = self.continuation_tail[
                -self._max_stop_length :
            ]

        matches = list(
            filter(
                lambda x: x[0] >= 0,
                sorted(
                    [(self.continuation_tail.find(x), x) for x in self.stop],
                    key=lambda x: x[0],
                ),
            )
        )

        if len(matches) == 0:
            return None

        return matches[0][1]
