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
from testing import TestSuite


def test_log_critical():
    var log = Logger[Level.CRITICAL]()

    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK-NOT: DEBUG::: hello world
    log.info("hello", "world")

    # CHECK: CRITICAL::: hello
    log.critical("hello")


fn main() raises:
    var suite = TestSuite()

    suite.test[test_log_critical]()

    suite^.run()
