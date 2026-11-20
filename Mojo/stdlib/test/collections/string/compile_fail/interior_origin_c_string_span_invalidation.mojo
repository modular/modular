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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# A `CStringSpan` vended by a `String` carries an interior origin, so
# reassigning the string invalidates it at compile time.


def main():
    var s = String("hello world, long enough to live on the heap")
    var c = s.as_c_string_span()
    s = String("a different string, long enough to live on the heap")
    # CHECK: use of invalidated interior reference 's["bytes"]'
    print(len(c))
