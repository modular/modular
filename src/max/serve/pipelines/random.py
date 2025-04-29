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

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class RandomTokenGeneratorContext:
    prompt: str
    seq_len: int


@dataclass
class RandomTokenGenerator:
    def new_context(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> RandomTokenGeneratorContext:
        if max_new_tokens is not None:
            raise NotImplementedError("max_new_tokens is not supported.")
        return RandomTokenGeneratorContext(prompt, len(prompt))

    def next_token(
        self, batch: dict[str, RandomTokenGeneratorContext]
    ) -> dict[str, str]:
        # Generate random values for each request including 0
        results = {rid: random.randint(0, 10) for rid in batch.keys()}
        # Requests which produced 0 are "completed" and not returned
        return {rid: str(rvalue) for rid, rvalue in results.items() if rvalue}

    def release(self, context: RandomTokenGeneratorContext):
        pass
