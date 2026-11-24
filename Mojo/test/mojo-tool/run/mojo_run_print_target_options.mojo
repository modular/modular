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

# Test that `mojo run` handles the --print-* target queries.

# RUN: mojo run --print-supported-targets 2>&1 | FileCheck %s --check-prefix=TARGETS
# RUN: mojo run --print-supported-accelerators 2>&1 | FileCheck %s --check-prefix=ACCELERATORS
# RUN: mojo run --print-supported-cpus --target-triple x86_64-unknown-linux-gnu 2>&1 | FileCheck %s --check-prefix=CPUS
# RUN: not mojo run --print-supported-cpus 2>&1 | FileCheck %s --check-prefix=CPUS_NO_TARGET
# RUN: mojo run --print-effective-target --target-abi lp64d 2>&1 | FileCheck %s --check-prefix=EFFECTIVE
# RUN: not mojo run --print-effective-target --print-supported-targets 2>&1 | FileCheck %s --check-prefix=MULTI

# A query before the input exits without running the program.
# RUN: mojo run --print-effective-target %s 2>&1 | FileCheck %s --check-prefixes=EFFECTIVE_ONLY

# A query after the input belongs to the program.
# RUN: mojo run %s --print-supported-targets 2>&1 | FileCheck %s --check-prefix=PROGRAM

# TARGETS: Registered Targets:
# TARGETS: x86-64

# ACCELERATORS: Supported Accelerator Architectures:

# CPUS: Available CPUs for target x86_64-unknown-linux-gnu:
# CPUS: haswell

# CPUS_NO_TARGET: error: --print-supported-cpus requires --target-triple to be specified

# EFFECTIVE: Effective target configuration:
# EFFECTIVE: --target-abi lp64d

# MULTI: error: only one --print-* option can be specified at a time

# EFFECTIVE_ONLY: Effective target configuration:
# EFFECTIVE_ONLY-NOT: ran main

# PROGRAM-NOT: Registered Targets:
# PROGRAM: ran main


def main():
    print("ran main")
