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

# Every directory beneath a search path (`-mojo-search-paths`, or the config's
# `import_path`) that isn't itself a source package was also searched for
# top-level names. This was a bug. `-I` directories are not searched this way
# so did not exhibit this behaviour.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics \
# RUN:   -mojo-search-paths=%S/inputs/subdir_search_root %s
# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics \
# RUN:   -I=%S/inputs/subdir_search_root %s 2>&1

# A module in a plain directory nested below a search path should not resolve
# as a top-level name.
# expected-error @+1 {{unable to locate module 'deep_mod'}}
import deep_mod


def main():
    deep_mod.deep_fn()


# // -----

# This includes plain directories inside a source package: the module should
# not be reachable at the top level without the package's name.
# expected-error @+1 {{unable to locate module 'leaked_mod'}}
import leaked_mod


def main():
    leaked_mod.leaked_fn()
