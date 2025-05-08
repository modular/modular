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

from sys.info import _accelerator_arch

from testing import *


def main():
    var accelerator_arch = _accelerator_arch()

    assert_true(
        accelerator_arch == "nvidia:80"
        or accelerator_arch == "nvidia:84"
        or accelerator_arch == "nvidia:86"
        or accelerator_arch == "nvidia:89"
        or accelerator_arch == "nvidia:90"
        or accelerator_arch == "nvidia:90a"
        or accelerator_arch == "amdgpu:94",
        "Expected specific accelerator_archs, got: " + accelerator_arch,
    )
