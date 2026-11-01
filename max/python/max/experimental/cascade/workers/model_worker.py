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
"""Call contract shared by the cascade model workers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import MaybeAsync, ResultIter, Worker
from max.experimental.cascade.interfaces.gen_ai import TextGenOptions

Int32Array = npt.NDArray[np.int32]


class ModelWorker(Worker, ABC):
    """Base class and call contract shared by the model workers.

    Subclasses named by ``--model-worker-impl`` are constructed as
    ``cls(config)`` with the :class:`~max.pipelines.lib.PipelineConfig`.
    """

    @abstractmethod
    async def decode(
        self,
        options: MaybeAsync[TextGenOptions],
        tokens: MaybeAsync[Int32Array],
        /,
    ) -> ResultIter[Int32Array]:
        """Streams the token ids generated for a tokenized prompt, in order."""
        ...
