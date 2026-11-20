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

# CHECK: error: invalid call to 'needs_even': lacking evidence to prove correctness
# CHECK: note: constraint declared here needs evidence for 'is_even(x)'


def is_even(x: Int) -> Bool:
    return x % 2 == 0


def needs_even[x: Int]() where is_even(x):
    pass


# start-opaque-calls-bad
def forward_even_bad[x: Int]() where x % 2 == 0:
    # ERROR: Needs evidence for `is_even(x)`.
    needs_even[x]()
    # end-opaque-calls-bad


def main():
    pass
