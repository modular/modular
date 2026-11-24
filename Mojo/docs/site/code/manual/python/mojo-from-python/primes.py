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
# The page's `primes.py`, verbatim.


def count_primes(limit):  # noqa: ANN001, ANN201
    # Count 2 separately because the loop skips even numbers.
    count = 1 if limit > 2 else 0
    for n in range(3, limit, 2):
        is_prime = True
        for d in range(3, n, 2):
            if d * d > n:
                break
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count
