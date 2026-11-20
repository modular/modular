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

# Pins the canonicalization equivalences the manual lists. Each caller carries
# one spelling of a proposition and calls a callee that requires the other, so
# a regression in canonicalization breaks the build rather than the prose.


def needs_ge_one[n: Int]() where n >= 1:
    pass


def has_gt_zero[n: Int]() where n > 0:
    needs_ge_one[n]()


def needs_gt_one[n: Int]() where n > 1:
    pass


def has_ge_two[n: Int]() where n >= 2:
    needs_gt_one[n]()


def needs_two_times[n: Int]() where 2 * n > 4:
    pass


def has_n_plus_n[n: Int]() where n + n > 4:
    needs_two_times[n]()


# Context-free folding: `1 + 1` is folded to `2` before the comparison.
def needs_ge_one_plus_one[n: Int]() where n >= 1 + 1:
    pass


def has_ge_two_folded[n: Int]() where n >= 2:
    needs_ge_one_plus_one[n]()


def main():
    has_gt_zero[3]()
    has_ge_two[3]()
    has_n_plus_n[3]()
    has_ge_two_folded[3]()
    print("All canonicalization equivalences hold.")
