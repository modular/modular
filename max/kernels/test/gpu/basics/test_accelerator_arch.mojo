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
        # NVIDIA
        accelerator_arch == "nvidia:80"
        or accelerator_arch == "nvidia:84"
        or accelerator_arch == "nvidia:86"
        or accelerator_arch == "nvidia:89"
        or accelerator_arch == "nvidia:90"
        or accelerator_arch == "nvidia:90a"
        or accelerator_arch == "nvidia:100"
        or accelerator_arch == "nvidia:100a"
        # AMD
        or accelerator_arch == "amdgpu:gfx942"
        or accelerator_arch == "amdgpu:gfx1101"
        or accelerator_arch == "amdgpu:gfx1102"
        or accelerator_arch == "amdgpu:gfx1103"
        or accelerator_arch == "amdgpu:gfx1200"
        or accelerator_arch == "amdgpu:gfx1201",
        "Expected specific accelerator_archs, got: " + accelerator_arch,
    )
