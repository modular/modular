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

# 'imm' and 'read' are the same convention: both lower to an imm-origin
# read_mem reference. (The 'read' spelling additionally warns; warnings go to
# stderr and are not FileChecked here.)

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: lit.fn @"imm_spelling(::String)"[imm *"x`"](%x: !lit.ref<!String, imm *"x`"> read_mem)
def imm_spelling(imm x: String) -> Int:
    return 1


# CHECK: lit.fn @"read_spelling(::String)"[imm *"x`"](%x: !lit.ref<!String, imm *"x`"> read_mem)
def read_spelling(read x: String) -> Int:
    return 1
