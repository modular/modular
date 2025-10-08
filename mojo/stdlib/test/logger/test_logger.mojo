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

from builtin._location import _SourceLocation
from logger import Level, Logger
from testing import TestSuite


# CHECK-LABEL: Test logging at trace level
def test_log_trace():
    print("=== Test logging at trace level")
    var log = Logger[Level.TRACE]()

    # CHECK: TRACE::: hello
    log.trace("hello")

    var log2 = Logger[Level.DEBUG]()
    # CHECK-NOT: TRACE::: hello
    log2.trace("hello")


# CHECK-LABEL: Test logging at info level
def test_log_info():
    print("=== Test logging at info level")
    var log = Logger[Level.INFO]()

    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK: INFO::: hello
    log.info("hello")


# CHECK-LABEL: Test no logging by default
def test_log_noset():
    print("=== Test no logging by default")
    var log = Logger()

    # CHECK-NOT: DEBUG::: hello world
    log.debug("hello", "world")

    # CHECK-NOT: INFO::: hello
    log.info("hello")


# CHECK-LABEL: Test logging with prefix
def test_log_with_prefix():
    print("=== Test logging with prefix")

    var log = Logger[Level.TRACE](prefix="[XYZ] ")

    # CHECK: [XYZ] hello
    log.trace("hello")


# CHECK-LABEL: Test logging with location
def test_log_with_location():
    print("=== Test logging with location")

    alias log = Logger[Level.TRACE](prefix="", source_location=True)

    # CHECK: test_logger.mojo:73:14] hello
    log.trace("hello")


# CHECK-LABEL: Test logging with custom location
def test_log_with_custom_location():
    print("=== Test logging with custom location")

    alias log = Logger[Level.TRACE](prefix="", source_location=True)

    # CHECK: somefile.mojo:42:999] hello
    log.trace("hello", location=_SourceLocation(42, 999, "somefile.mojo"))


# CHECK-LABEL: Test logging with sep/end
def test_log_with_sep_end():
    print("=== Test logging with sep/end")

    var log = Logger[Level.TRACE]()

    # CHECK: hello mojo world!!!
    log.trace("hello", "world", sep=" mojo ", end="!!!\n")


def main():
    var suite = TestSuite()

    suite.test[test_log_trace]()
    suite.test[test_log_info]()
    suite.test[test_log_noset]()
    suite.test[test_log_with_prefix]()
    suite.test[test_log_with_location]()
    suite.test[test_log_with_custom_location]()
    suite.test[test_log_with_sep_end]()

    suite^.run()
