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

from logger import Level, Logger


# CHECK-LABEL: Test logging at info level
def test_log_info():
    print("=== Test logging at info level")
    var log = Logger[Level.INFO]()

    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK: INFO::: hello
    log.info("hello")


# CHECK-LABEL: Test no logging by default
fn test_log_noset():
    print("=== Test no logging by default")
    var log = Logger()

    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK-NOT: INFO::: hello
    log.info("hello")


def main():
    test_log_info()
    test_log_noset()
