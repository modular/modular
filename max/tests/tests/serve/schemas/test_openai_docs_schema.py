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
"""Checks the hand-written API docs schema against the request models."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import yaml
from max.pipelines.request.provider_options import MAX_NUM_IMAGES


def _find(node: object, key: str) -> Iterator[object]:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                yield value
            yield from _find(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from _find(value, key)


def test_num_images_range_matches_the_request_model() -> None:
    docs = yaml.safe_load(Path(os.environ["OPENAI_DOCS_YAML"]).read_text())
    [num_images] = list(_find(docs, "num_images"))
    assert isinstance(num_images, dict)
    assert num_images["minimum"] == 1
    assert num_images["maximum"] == MAX_NUM_IMAGES
