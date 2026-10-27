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

"""Engram token map for the ModuleV3 stack: a thin adapter over main.

The compressed-id map builder lives in main's
:mod:`~...deepseekV41.layers.engram_token_map` and is reused as-is; this
module adds the recorded digest of the real model's map (checked by the
loader and the tests, never just the count, since a ``tokenizers``
Unicode bump can keep the size and change entries) and the loader-facing
entry point that also returns the compressed vocab size every hash
multiplier derives from.
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.deepseekV41.layers.engram_token_map import (
    BackendTokenizerLike,
    TokenizerLike,
    build_compressed_token_map,
    compressed_token_map_array,
)

__all__ = [
    "EXPECTED_COMPRESSED_VOCAB_SIZE",
    "EXPECTED_MAP_SHA256",
    "BackendTokenizerLike",
    "TokenizerLike",
    "build_compressed_token_map",
    "check_map_digest",
    "compressed_token_map_array",
    "compressed_token_map_with_size",
]

# The compressed map's recorded digest (sha256 of the int32 little-endian
# bytes). The loader and tests check this, never just the count.
EXPECTED_MAP_SHA256 = (
    "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"
)

# The compressed vocab size every hash multiplier derives from.
EXPECTED_COMPRESSED_VOCAB_SIZE = 99092


def compressed_token_map_with_size(
    tokenizer: TokenizerLike,
) -> tuple[npt.NDArray[np.int32], int]:
    """The int32 map plus the compressed vocab size it defines.

    The V3 Engram hash constants derive from both: the map is the ABI
    operand the hash reads at trace time, and the size bounds every
    layer's multiplier draws.

    Args:
        tokenizer: A tokenizer exposing ``__len__`` and a
            ``backend_tokenizer`` with ``.decode()``/``.id_to_token()``,
            the one the model serves with.

    Returns:
        ``(map, compressed_vocab_size)`` where ``map`` is ``[vocab_size]``
        int32 in tokenizer id order.
    """
    lookup, size = build_compressed_token_map(tokenizer)
    return np.asarray(lookup, dtype=np.int32), size


def check_map_digest(token_map: npt.ArrayLike) -> str:
    """Checks a real-model token map against its recorded digest.

    Args:
        token_map: The ``[vocab]`` int32 compressed-id map.

    Returns:
        The hex digest, equal to :data:`EXPECTED_MAP_SHA256`.

    Raises:
        ValueError: If the sha256 of the int32 little-endian bytes is not
            the recorded digest.
    """
    array = np.asarray(token_map, dtype=np.int32)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    if digest != EXPECTED_MAP_SHA256:
        raise ValueError(
            f"engram token map sha256 {digest} != recorded "
            f"{EXPECTED_MAP_SHA256} (the tokenizer does not reproduce the "
            "recorded Engram map?)"
        )
    return digest
