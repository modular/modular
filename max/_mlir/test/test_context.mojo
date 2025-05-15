# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug -D MLIRC_DYLIB=.graph_lib %s


import _mlir
from testing import assert_equal


def main():
    with _mlir.Context() as ctx:
        var module = _mlir.Module(_mlir.Location.unknown(ctx))
        assert_equal("module {\n}\n", String(module))

        # Right now the lifetime of `module` is poorly defined.
        # This `destroy()` is just a temp. workaround so
        # ASAN does not complain (and can therefore catch realer bugs)
        module.destroy()
