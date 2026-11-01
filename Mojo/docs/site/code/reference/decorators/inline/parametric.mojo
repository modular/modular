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

from std.testing import *


@inline(inlineLevel)
def scaled[inlineLevel: InlineLevel = .automatic](x: Int) -> Int:
    return x * 3


def main() raises:
    print(scaled(2))  # Inlined heuristically (default)
    print(scaled[.never](2))  # Don't inline
    print(scaled[.always](2))  # Always inline
    assert_equal(scaled[.never](2), 6)
    # All three instantiations agree on the result; only the inlining differs.
    assert_equal(scaled[.always](2), scaled[.never](2))
    assert_equal(scaled(2), scaled[.never](2))
