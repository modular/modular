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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that float literals do not implicitly convert to integral SIMD types.
# Explicit construction (`UInt(400.)`) is supported, but implicit narrowing
# from float to int stays rejected. See
# https://github.com/modular/modular/issues/5290.


def takes_uint(x: UInt):
    pass


# CHECK: constraint failed: the SIMD type must be floating point
def main():
    takes_uint(400.0)
