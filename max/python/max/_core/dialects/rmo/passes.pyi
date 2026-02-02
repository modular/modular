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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Callable

import max._core

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

DiagnosticHandler = Callable

def LegalizeRMOOps(skip_rmo_rebind: bool = False) -> max._core.Pass:
    """
    There are broadly two categories of RMO operators. The first are those
    analogous to an existing MO operator. These are lowered by properly
    rebinding each input and result to the types determined in the `MOAnalogue`
    interface. These operators have a lowering automatically generated as long
    as the `MOAnalogue` operator is implemented.

    The next are RMO operators which lower to other RMO operators.
    """
