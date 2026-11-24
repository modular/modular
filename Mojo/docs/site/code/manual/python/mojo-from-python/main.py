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
# The page's `main.py`, verbatim.

import time

import mojo.importer  # noqa: F401

# isort: split

import fast_primes  # type: ignore[import-not-found]
import primes

for module in (primes, fast_primes):
    start = time.perf_counter()
    count = module.count_primes(2_000_000)
    elapsed = time.perf_counter() - start
    print(f"{module.__name__}: {count} primes in {elapsed:.2f} s")
