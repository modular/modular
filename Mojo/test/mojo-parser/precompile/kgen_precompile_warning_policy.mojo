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

# Test -Werror, -Wno-error and --disable-warnings in kgen -precompile.

# RUN: %precompile-mojo-isolated %S/inputs/warning_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: not %precompile-mojo-isolated -Werror %S/inputs/warning_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: %precompile-mojo-isolated -Werror -Wno-error %S/inputs/warning_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: not %precompile-mojo-isolated -Wno-error -Werror %S/inputs/warning_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=ERROR
# RUN: %precompile-mojo-isolated --disable-warnings %S/inputs/warning_package -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty

# WARNING: warning: assignment to 'foo' was never used
# ERROR: error: assignment to 'foo' was never used
# DISABLED-NOT: foo
