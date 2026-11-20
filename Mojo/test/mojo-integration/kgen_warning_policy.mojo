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

# Test -Werror, -Wno-error and --disable-warnings in kgen.

# RUN: kgen -elaborate %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: not kgen -elaborate -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: kgen -elaborate -Werror -Wno-error %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: not kgen -elaborate -Wno-error -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: kgen -elaborate --disable-warnings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty

# WARNING: warning: assignment to 'foo' was never used
# ERROR: error: assignment to 'foo' was never used
# DISABLED-NOT: foo


def main():
    var foo = 1
