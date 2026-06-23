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
import hashlib

import numpy as np
import pytest
from max._core_mojo import block_hasher_sha256


def _py_reference(
    tokens: np.ndarray, block_size: int, parent_hash: bytes
) -> list[bytes]:
    """Pure-Python reference implementation of the chained SHA-256 block hasher."""
    assert tokens.dtype == np.int32
    n = tokens.size // block_size
    prev = parent_hash
    out: list[bytes] = []
    for i in range(n):
        block_bytes = tokens[i * block_size : (i + 1) * block_size].tobytes()
        local = hashlib.sha256(block_bytes).digest()
        seq = hashlib.sha256(local + prev).digest()
        out.append(seq)
        prev = seq
    return out


@pytest.mark.parametrize("block_size", [1, 4, 64, 128])
@pytest.mark.parametrize("num_blocks", [1, 2, 32])
def test_matches_python_reference(block_size: int, num_blocks: int) -> None:
    tokens = np.arange(block_size * num_blocks, dtype=np.int32)
    parent = b"\x00" * 32
    got = block_hasher_sha256(tokens, block_size, parent)
    expected = _py_reference(tokens, block_size, parent)
    assert got == expected


def test_chaining_matches_split() -> None:
    """Hashing N+M blocks in one shot equals hashing N then M with chained parent."""
    block_size = 16
    tokens = np.arange(block_size * 5, dtype=np.int32)
    parent = b"\xab" * 32

    one_shot = block_hasher_sha256(tokens, block_size, parent)
    first_two = block_hasher_sha256(
        tokens[: 2 * block_size], block_size, parent
    )
    last_three = block_hasher_sha256(
        tokens[2 * block_size :], block_size, first_two[-1]
    )
    assert one_shot == first_two + last_three


def test_parent_hash_isolation() -> None:
    """Different parent_hash with the same tokens produces different seq hashes."""
    tokens = np.arange(128, dtype=np.int32)
    a = block_hasher_sha256(tokens, 32, b"\x00" * 32)
    b = block_hasher_sha256(tokens, 32, b"\x01" * 32)
    assert a != b
    assert all(x != y for x, y in zip(a, b, strict=False))


def test_each_hash_is_32_bytes() -> None:
    tokens = np.arange(640, dtype=np.int32)
    out = block_hasher_sha256(tokens, 128, b"\x00" * 32)
    assert all(isinstance(x, bytes) and len(x) == 32 for x in out)


def test_invalid_parent_hash_length_raises() -> None:
    tokens = np.arange(128, dtype=np.int32)
    with pytest.raises(ValueError):
        block_hasher_sha256(tokens, 32, b"\x00" * 16)


def test_dtype_coerced_to_int32() -> None:
    """Non-int32 inputs should be cast (matches existing block_hasher behavior)."""
    tokens_int64 = np.arange(128, dtype=np.int64)
    tokens_int32 = tokens_int64.astype(np.int32)
    a = block_hasher_sha256(tokens_int64, 32)
    b = block_hasher_sha256(tokens_int32, 32)
    assert a == b
