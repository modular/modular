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

# `mojo doc` accepts and honours the shared compilation diagnostic options.

# RUN: mojo doc %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARNINGS
# RUN: mojo doc --disable-warnings %s -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-WARNINGS
# RUN: mojo doc --ignore-deprecated=deprecatedFn %s -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-WARNINGS
# RUN: mojo doc --ignore-deprecated deprecatedFn %s -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-WARNINGS
# WARNINGS: warning: deprecatedFn is deprecated
# NO-WARNINGS-NOT: warning:

# RUN: mojo doc -I %S/../package/inputs/unstable_apis %S/../package/inputs/unstable_apis/consumer -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=STABLE
# RUN: mojo doc --warn-on-unstable-apis -I %S/../package/inputs/unstable_apis %S/../package/inputs/unstable_apis/consumer -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNSTABLE
# STABLE-NOT: warning: use of unstable API
# UNSTABLE: warning: use of unstable API 'unstable_fn'


@deprecated("deprecatedFn is deprecated")
def deprecatedFn():
    pass


def caller():
    deprecatedFn()
