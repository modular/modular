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
# RUN: mojo %s 2>&1

from pathlib import Path
from sys import DLHandle
from sys.ffi import _find_dylib, _try_find_dylib


def check_dlhandle():
    _ = DLHandle("libm.so.6")


def check_find_dylib():
    _ = _find_dylib("libm.so.6")


def check_find_dylib_multiple():
    _ = _find_dylib("invalid", "libm.so.6")


def check_try_find_dylib():
    try:
        # CHECK: Failed to load my_library from invalid
        _ = _try_find_dylib["my_library"]("invalid")
    except e:
        print(e)


def main():
    check_dlhandle()
    check_find_dylib()
    check_find_dylib_multiple()
    check_try_find_dylib()
