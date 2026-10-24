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

# Test the command-line interface to kgen-translate with -emit-mojo-doc.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/empty && touch %t.dir/notes.txt

# RUN: not %mojo-docs-isolated -I %S %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN-FLAG
# UNKNOWN-FLAG: error: unsupported argument '-I' with -emit-mojo-doc

# RUN: not %mojo-docs-isolated %s %s 2>&1 | FileCheck %s --check-prefix=TWO-INPUTS
# TWO-INPUTS: error: -emit-mojo-doc takes a single input, got '{{.*}}mojo_doc_input_errors.mojo' and '{{.*}}mojo_doc_input_errors.mojo'

# RUN: not %mojo-docs-isolated %t.dir/notes.txt 2>&1 | FileCheck %s --check-prefix=NOT-MOJO
# RUN: not %mojo-docs-isolated %t.dir/empty 2>&1 | FileCheck %s --check-prefix=NOT-MOJO
# NOT-MOJO: error: cannot open '{{.*}}', since it does not appear to be a Mojo file

# RUN: not %mojo-docs-isolated %t.dir/missing.mojo 2>&1 | FileCheck %s --check-prefix=MISSING
# MISSING: error: could not generate documentation
