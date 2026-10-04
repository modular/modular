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

from linalg.matmul.gpu.sm90.dispatch import (
    llama_8b_fp8_table,
    llama_405b_fp8_table,
)
from linalg.matmul.gpu.sm90.tuning_configs import TuningGroup


def main() raises:
    comptime assert llama_8b_fp8_table.check()
    comptime assert llama_405b_fp8_table.check()
    _test_enumlike()


def _is_core(group: TuningGroup) -> Bool:
    """Returns True when the runtime `__match` selects `TuningGroup.CORE`.

    Every case is named so the match fails to compile if `_enum_case_names`
    drifts from the comptime case constants.
    """
    __match group:
        case .CORE:
            return True
        case .MISCELLANEOUS:
            return False
        case .INTERNVL:
            return False
        case .LLAMA_3_3_70B:
            return False
        case .GEMMA_3_27B:
            return False


def _test_enumlike() raises:
    var group = TuningGroup.CORE
    assert _is_core(group), "expected TuningGroup.CORE to match case .CORE"

    var last_group = TuningGroup.GEMMA_3_27B
    assert not _is_core(
        last_group
    ), "expected TuningGroup.GEMMA_3_27B not to match case .CORE"

    assert (
        String(TuningGroup.CORE) == "TuningGroup.CORE"
    ), "default Writable should format as TypeName.case"
