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

# REQUIRES: system-linux, intel_amx
# RUN: %mojo-no-debug %s

from sys import has_intel_amx, os_is_linux

from linalg.intel_amx_intrinsics import init_intel_amx
from testing import assert_false, assert_true


fn test_has_intel_amx() raises:
    assert_true(os_is_linux())
    assert_true(has_intel_amx())
    assert_true(init_intel_amx())


fn main() raises:
    test_has_intel_amx()
