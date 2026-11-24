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

# Precompilation validates doc strings and then strips their locations.
# Documenting the precompiled package must not validate them again.

# RUN: mkdir -p %t
# RUN: mojo precompile %S/inputs/bad_doc_package -o %t/bad_doc_package.mojoc \
# RUN:   2>&1 | FileCheck %s --check-prefix=PRECOMPILE
# PRECOMPILE: warning: unknown argument 'y' in doc string

# RUN: mojo doc %t/bad_doc_package.mojoc -o %t/out.json 2>&1 \
# RUN:   | FileCheck %s --allow-empty --check-prefix=DOC
# DOC-NOT: {{warning|error}}

# RUN: FileCheck %s --check-prefix=JSON < %t/out.json
# JSON: "name": "foo"
