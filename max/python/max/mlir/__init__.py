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

from .._mlir._mlir_libs._mlir import *
from .._mlir._mlir_libs._mlir.ir import *


def site_initialize() -> None:
    from max._core import graph  # type: ignore

    from .._mlir._mlir_libs import get_dialect_registry  # type: ignore

    graph.load_modular_dialects(get_dialect_registry())


site_initialize()
