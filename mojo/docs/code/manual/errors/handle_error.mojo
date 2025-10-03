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
def incr(n: Int) -> Int:
    if n == Int.MAX:
        raise "inc: integer overflow"
    else:
        return n + 1


def main():
    for value in [0, 1, Int.MAX]:
        try:
            print()
            print("try     =>", value)
            if value == 1:
                continue
            result = "{} incremented is {}".format(value, incr(value))
        except e:
            print("except  =>", e)
        else:
            print("else    =>", result)
        finally:
            print("finally => ====================")
