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

# Diagnostic String descriptors and formatting buffers must stay out of the
# caller, including at -O0. Only the failure block calls the no-inline helper.
#
# RUN: mkdir -p %t
# RUN: %mojo-build-no-debug-no-assert %s -D ASSERT=all --emit llvm -o %t/all.ll
# RUN: FileCheck %s --check-prefixes=CHECK,ALL --input-file=%t/all.ll
# RUN: %mojo-build-no-debug-no-assert %s -D ASSERT=safe --emit llvm -o %t/safe.ll
# RUN: FileCheck %s --check-prefixes=CHECK,SAFE --input-file=%t/safe.ll
# RUN: %mojo-build-no-debug-no-assert %s -D ASSERT=none --emit llvm -o %t/none.ll
# RUN: FileCheck %s --check-prefix=NONE --input-file=%t/none.ll

from std.collections import check_bounds


# CHECK-LABEL: define {{.*}}void @bounds_probe(
# CHECK-NOT: alloca { ptr, i64, i64 }
# CHECK: icmp uge
# CHECK: br i1
# CHECK-NOT: alloca { ptr, i64, i64 }
# CHECK: call void @{{.*}}_check_bounds_fail
# CHECK-NOT: _debug_assert_fail_format
# CHECK: ret void
# NONE-LABEL: define {{.*}}void @bounds_probe(
# NONE-NOT: _check_bounds_fail
# NONE-NOT: icmp
# NONE: ret void
@export
def bounds_probe(idx: Int, size: Int) abi("C"):
    check_bounds(idx, size)


# ALL-LABEL: define {{.*}}void @bounds_probe_optout(
# ALL: icmp uge
# ALL: call void @{{.*}}_check_bounds_fail
# ALL: ret void
# SAFE-LABEL: define {{.*}}void @bounds_probe_optout(
# SAFE-NOT: _check_bounds_fail
# SAFE-NOT: icmp
# SAFE: ret void
@export
def bounds_probe_optout(idx: Int, size: Int) abi("C"):
    check_bounds[cpu_default=False](idx, size)


def main():
    bounds_probe(0, 1)


# CHECK-LABEL: define internal void @{{.*}}_check_bounds_fail
# CHECK-SAME: #[[ATTR:[0-9]+]]
# CHECK: call void @{{.*}}_debug_assert_fail_format
# CHECK: attributes #[[ATTR]] = { noinline
