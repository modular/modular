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

# COM: Metal kernels must build on any host, with or without the Metal
# toolchain, because the metallib is packaged in-tree.
# RUN: env MODULAR_METAL_INTREE_METALLIB=1 MACOSX_DEPLOYMENT_TARGET=15.0 not %mojo-build --target-accelerator=apple-m5-metal4 %s -o %t.old 2>&1 | FileCheck %s --check-prefix=TOO-OLD
# RUN: env MODULAR_METAL_INTREE_METALLIB=1 MACOSX_DEPLOYMENT_TARGET=abc not %mojo-build --target-accelerator=apple-m5-metal4 %s -o %t.bad 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: env MODULAR_METAL_INTREE_METALLIB=maybe not %mojo-build --target-accelerator=apple-m5-metal4 %s -o %t.env 2>&1 | FileCheck %s --check-prefix=BAD-ENV
# RUN: env MODULAR_METAL_INTREE_METALLIB=1 %mojo-build --target-accelerator=apple-m5-metal4 %s -o %t
# RUN: grep -a -c MTLB %t | FileCheck %s

# TOO-OLD: MACOSX_DEPLOYMENT_TARGET=15.0 is older than macOS 26.0
# INVALID: invalid version number in 'MACOSX_DEPLOYMENT_TARGET=abc'
# BAD-ENV: MODULAR_METAL_INTREE_METALLIB='maybe' is not a boolean

from max.gpu.host import DeviceContext


def hello():
    pass


# CHECK: {{^[1-9]}}
def main() raises:
    with DeviceContext() as ctx:
        _ = ctx.compile_function[hello]()
