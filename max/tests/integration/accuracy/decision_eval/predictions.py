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

"""One decision per case, and the JSONL file that stores them.

MAX, the reference implementation and the published JevBench systems all end
up as the same record, so one scorer and one parity check read them all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prediction:
    """A model's probability for every option of one case."""

    case_id: str
    family_id: str
    slice: str
    task_type: str
    perturbation: str
    option_order: list[int]
    """Position ``i`` of this case's options is original option ``option_order[i]``."""
    gold_index: int
    probabilities: list[float]
    """One per option, in the order the case lists them."""

    @property
    def predicted_index(self) -> int:
        return max(
            range(len(self.probabilities)), key=self.probabilities.__getitem__
        )


def write_predictions(path: Path, predictions: Iterable[Prediction]) -> None:
    """Writes ``predictions`` to ``path``, one JSON object per line."""
    with path.open("w") as out:
        for prediction in predictions:
            out.write(json.dumps(asdict(prediction)) + "\n")


def read_predictions(path: Path) -> list[Prediction]:
    """Reads a file written by :func:`write_predictions`."""
    with path.open() as source:
        return [
            Prediction(**json.loads(line)) for line in source if line.strip()
        ]
