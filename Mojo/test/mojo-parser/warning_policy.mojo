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

# Test -Werror, -Wno-error and --disable-warnings in kgen-translate.

# RUN: %parse-mojo-isolated %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: not %parse-mojo-isolated -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: %parse-mojo-isolated -Werror -Wno-error %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: not %parse-mojo-isolated -Wno-error -Werror %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: %parse-mojo-isolated --disable-warnings %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty

# WARNING: warning: old_fn is deprecated
# ERROR: error: old_fn is deprecated
# DISABLED-NOT: old_fn


@deprecated("old_fn is deprecated")
def old_fn():
    pass


def caller():
    old_fn()
