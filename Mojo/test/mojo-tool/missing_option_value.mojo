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

# Reject an option that ends the command line without its value.
# RUN: not mojo build %s -o 2>&1 | FileCheck %s
# RUN: not mojo run -I 2>&1 | FileCheck %s --check-prefix=INCLUDE
# RUN: not mojo doc %s -I 2>&1 | FileCheck %s --check-prefix=INCLUDE
# RUN: not mojo precompile %s -o 2>&1 | FileCheck %s
# CHECK: mojo{{.*}}: error: missing value for '-o'
# INCLUDE: mojo{{.*}}: error: missing value for '-I'

# RUN: not mojo build --diagnostic-format json %s -o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=JSON
# JSON: {"kind":"error","message":"missing value for '-o'"}


def main():
    pass
