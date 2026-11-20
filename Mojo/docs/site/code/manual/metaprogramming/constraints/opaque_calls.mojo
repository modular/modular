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


# start-opaque-calls-setup
def is_even(x: Int) -> Bool:
    return x % 2 == 0


def needs_even[x: Int]() where is_even(x):
    pass
    # end-opaque-calls-setup


# start-opaque-calls-good
def forward_even_good[x: Int]() where is_even(x):
    # SUCCESS: the constraint is the identical call, so it matches.
    needs_even[x]()
    # end-opaque-calls-good


def main():
    comptime assert is_even(4)
    forward_even_good[4]()
    print("Forwarded an even parameter.")
