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
# RUN: %bare-mojo build %S/mojo_module.mojo --emit shared-lib -o mojo_module.so
# RUN: python3 %s

import sys
import time

import numpy as np

# Put the current directory (containing module.so) on the Python module lookup
# path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module


def s2us(s: float) -> float:
    return s * 1000 * 1000


def naive_python_hashing(tokens: np.ndarray, block_size: int) -> list[int]:
    num_elts = tokens.size
    num_hashes = num_elts // block_size

    results = []
    for i in range(num_hashes):
        block = tokens[i * block_size : (i + 1) * block_size]
        hash_val = hash(tuple(block))
        results.append(hash_val)

    return results


if __name__ == "__main__":
    print("-" * 80)
    print("Hello from Block Hasher Example!")
    print("-" * 80)

    block_size = 128
    # Use int32 for tokens
    tokens = np.arange(512, dtype=np.int32)

    elapsed = []

    for i in range(10):
        enable_dbg_prints = int(i == 0)

        t0 = time.time()
        hashes_mojo_return_list = mojo_module.mojo_block_hasher_return_list(
            tokens, block_size, enable_dbg_prints
        )
        t1 = time.time()
        dt_mojo_return_list = s2us(t1 - t0)

        t0 = time.time()
        # Use uint64 for hashes
        hashes_mojo_np_array_inout = np.empty(
            len(tokens) // block_size, dtype=np.uint64
        )
        mojo_module.mojo_block_hasher_inplace(
            tokens, hashes_mojo_np_array_inout, block_size, enable_dbg_prints
        )
        t1 = time.time()
        dt_mojo_np_array_inout = s2us(t1 - t0)

        t0 = time.time()
        hashes_naive = naive_python_hashing(tokens, block_size)
        t1 = time.time()
        dt_naive = s2us(t1 - t0)

        if i == 0:
            print(f"Mojo hashes_mojo_return_list: {hashes_mojo_return_list}")
            print(
                f"Mojo hashes_mojo_np_array_inout: {hashes_mojo_np_array_inout}"
            )
            print(f"Python hashes: {hashes_naive}")

        print(
            f"Time taken: "
            f"{dt_mojo_return_list:.2f} us (Mojo Return List) vs "
            f"{dt_mojo_np_array_inout:.2f} us (Mojo Inplace) vs "
            f"{dt_naive:.2f} us (Python)"
        )
