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

from algorithm import map


def main():
    # Create list with initial values to act on
    var list: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Function applied to the value at each index
    @parameter
    fn exponent_2(idx: Int):
        list[idx] = 2.0 ** list[idx]

    # Apply the mapped function across the index range
    map[exponent_2](len(list))

    # Show results
    for idx in range(len(list)):
        print(list[idx])
