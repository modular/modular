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
"""CPU producer: compiles one of the tiny MiMo-V2 DFlash model's graphs to a
MEF with no GPU, for the GPU tests to initialize instead of compiling.

Run as a build action by the ``precompiled_mefs`` macro; see the package's
``BUILD.bazel``.
"""

from __future__ import annotations

from mimo_dflash_harness import PRECOMPILED_DEVICES, named_graph
from test_common.mef_precompile import precompile_entrypoint

if __name__ == "__main__":
    precompile_entrypoint(named_graph, device_count=PRECOMPILED_DEVICES)
