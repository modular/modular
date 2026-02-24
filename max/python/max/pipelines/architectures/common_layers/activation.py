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

"""Activation function utilities."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from max.experimental import functional as F
from max.experimental.tensor import Tensor


def activation_function_from_name(name: str) -> Callable[[Tensor], Tensor]:
    activations: dict[str, Callable[[Tensor], Tensor]] = {
        "silu": F.silu,
        "swish": F.silu,
        "gelu": F.gelu,
        "gelu_tanh": partial(F.gelu, approximate="tanh"),
        "gelu_quick": partial(F.gelu, approximate="quick"),
        "quick_gelu": partial(F.gelu, approximate="quick"),
        "relu": F.relu,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid,
    }
    activation_function = activations.get(name)
    if activation_function is None:
        raise ValueError(f"Unrecognized activation function name ({name})")
    return activation_function
