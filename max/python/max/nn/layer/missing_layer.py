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

"""Defines :class:`MissingLayer`, the place of a layer another stage builds."""

from __future__ import annotations

from typing import Any, NoReturn

from .layer import Module


class MissingLayer(Module):
    """Holds the place of a decoder layer another pipeline stage builds.

    A pipeline stage builds only its own decoder layers. Its
    :class:`~max.nn.layer.LayerList` still keeps one entry per decoder layer,
    with a :class:`MissingLayer` at each position it does not build, so every
    layer it does build keeps its checkpoint name. A :class:`MissingLayer`
    declares no weights.

    Args:
        layer_id: The decoder layer this entry stands for.
    """

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id

    def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            f"Decoder layer {self.layer_id} belongs to another pipeline stage."
        )
