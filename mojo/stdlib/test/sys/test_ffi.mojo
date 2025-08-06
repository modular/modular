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

from sys.ffi import get_errno, set_errno, ErrNo
from os.path import realpath
from testing import assert_raises, assert_equal


def test_errno():
    # test it raises the correct libc error
    with assert_raises(contains=String(ErrNo.ENOENT)):
        _ = realpath("does/not/exist")

    # Test that it sets errno correctly
    assert_equal(get_errno(), ErrNo.ENOENT)
    assert_equal(String(ErrNo.ENOENT), "No such file or directory")

    # test that errno can be reset to success
    set_errno(ErrNo.SUCCESS)
    assert_equal(get_errno(), ErrNo.SUCCESS)

    # Make sure we can set errno to a different value
    set_errno(ErrNo.EPERM)
    if get_errno() != ErrNo.EPERM:
        raise Error("Failed to set errno to EPERM")


def main():
    test_errno()
