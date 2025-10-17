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


def main():
    # start-owning-variable
    owning_variable = "Owned value"
    # end-owning-variable

    _ = owning_variable

    # start-move-value
    first = [1, 2, 3]
    second = first^
    # end-move-value

    _ = second

    # start-explicit-copy
    first = [1, 2, 3]
    second = first.copy()
    # end-explicit-copy

    _ = second

    # start-implicit-copy
    one_value = 15
    another_value = one_value  # implicit copy
    # end-implicit-copy

    _ = another_value
