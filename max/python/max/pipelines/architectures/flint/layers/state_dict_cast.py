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

"""Casts floating-point weights to the serving dtype, leaving other dtypes alone.

The ModuleV3 load path does not apply the dtype cast a default GPU serve does.
"""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph.weights import WeightData

FLOAT_DTYPES = frozenset({DType.float32, DType.float16, DType.bfloat16})


def cast_state_dict(
    state_dict: dict[str, Any], target: DType
) -> dict[str, Any]:
    """Returns the state dict with floating-point weights cast to ``target``."""
    if target not in FLOAT_DTYPES:
        raise ValueError(f"cast target must be a float dtype, got {target}")
    return {
        key: (
            value.astype(target)
            if isinstance(value, WeightData)
            and value.dtype in FLOAT_DTYPES
            and value.dtype != target
            else value
        )
        for key, value in state_dict.items()
    }
