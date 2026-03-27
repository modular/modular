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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import override

from .pixel import PixelBenchmarkDataset
from .types import RequestSamples


class SyntheticPixelBenchmarkDataset(PixelBenchmarkDataset):
    @override
    def fetch(self) -> None:
        """Fetch Synthetic Pixel dataset.

        Synthetic pixel prompts are generated in-memory and do not require a
        local file.
        """
        pass

    @override
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase | None,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs: Any,
    ) -> RequestSamples:
        image_options = self._build_image_options(
            image_width=kwargs.get("image_width"),
            image_height=kwargs.get("image_height"),
            image_steps=kwargs.get("image_steps"),
            image_guidance_scale=kwargs.get("image_guidance_scale"),
            image_negative_prompt=kwargs.get("image_negative_prompt"),
            image_seed=kwargs.get("image_seed"),
        )

        requests = [
            self._build_request(
                prompt=f"Random prompt {idx} for benchmarking pixel generation pipelines",
                image_options=image_options,
            )
            for idx in range(num_requests)
        ]
        return RequestSamples(requests=requests)
