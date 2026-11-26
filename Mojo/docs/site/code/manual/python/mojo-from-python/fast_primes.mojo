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
# The page's two `fast_primes.mojo` blocks, merged as step 2 of "Run both
# versions" says. The second block's imports sit at the top of the file, where
# its first comment puts them.

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


def count_primes(limit: Int) -> Int:
    # Count 2 separately because the loop skips even numbers.
    var count = 1 if limit > 2 else 0
    for n in range(3, limit, 2):
        var is_prime = True
        for d in range(3, n, 2):
            if d * d > n:
                break
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count


# Python-facing wrapper to call `count_primes()`
def py_count_primes(limit: PythonObject) raises -> PythonObject:
    return count_primes(Int(py=limit))


# Python-facing module
@export
def PyInit_fast_primes() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("fast_primes")
        module.def_function[py_count_primes]("count_primes")
        return module.finalize()
    except e:
        abort(t"error creating fast_primes: {e}")
