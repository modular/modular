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


# start-reference-parameter-on-type
fn on_type():
    print(SIMD[DType.float32, 2].size)  # prints 2
    # end-reference-parameter-on-type


# start-reference-parameter-on-instance
fn on_instance():
    var x = SIMD[DType.int32, 2](4, 8)
    print(x.dtype)  # prints int32
    # end-reference-parameter-on-instance
    _ = x


def main():
    on_type()
    on_instance()
