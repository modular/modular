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

# Test how `mojo run` validates compilation option values.

# A repeated option takes its last value.
# RUN: mojo run --num-threads=1 --num-threads=2 %s | FileCheck %s --check-prefix=RAN
# RUN: mojo run --elaboration-error-limit=1 --elaboration-error-limit=5 %s | FileCheck %s --check-prefix=RAN
# RAN: ran

# RUN: not mojo run -O7 %s 2>&1 | FileCheck %s --check-prefix=OPT
# OPT: error: invalid optimization level '7', expected number from 0-3 inclusive

# RUN: not mojo run --debug-level=some %s 2>&1 | FileCheck %s --check-prefix=DEBUG
# DEBUG: error: invalid debug level 'some', expected one of: `none` (the default value), `line-tables`, or `full`

# RUN: not mojo run --num-threads=many %s 2>&1 | FileCheck %s --check-prefix=THREADS
# THREADS: error: Invalid number of threads: many

# RUN: not mojo run --debug-info-language=Rust %s 2>&1 | FileCheck %s --check-prefix=LANGUAGE
# LANGUAGE: error: invalid debug info language 'Rust', expected one of: `C` or `Mojo`

# RUN: not mojo run --elaboration-error-verbose=loud %s 2>&1 | FileCheck %s --check-prefix=VERBOSE
# VERBOSE: error: invalid elaboration-error-verbose 'loud', expected one of: `no-params` (the default value), `simple-params`, or `all-params`

# RUN: not mojo run --shared-libasan %s 2>&1 | FileCheck %s --check-prefix=SHARED
# SHARED: error: cannot enable --shared-libasan without enabling --sanitize=address


def main():
    print("ran")
