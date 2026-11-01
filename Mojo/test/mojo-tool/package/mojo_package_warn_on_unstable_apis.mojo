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

# RUN: mojo precompile -I %S/inputs/unstable_apis %S/inputs/unstable_apis/consumer -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=DEFAULT --allow-empty
# RUN: mojo precompile --warn-on-unstable-apis -I %S/inputs/unstable_apis %S/inputs/unstable_apis/consumer -o %t.mojoc 2>&1 | FileCheck %s --check-prefix=WARN

# DEFAULT-NOT: warning: use of unstable API

# WARN: consumer{{/|\\}}__init__.mojo:20:5: warning: use of unstable API 'unstable_fn'
