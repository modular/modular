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
# REQUIRES: NVIDIA-GPU
# TODO: MSTDL-1156
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s | FileCheck %s


from nvml import Device
from nvml.nvml import _get_nvml_library_path


fn has_nvml_library() -> Bool:
    try:
        _ = _get_nvml_library_path()
        return True
    except:
        return False


def main():
    if not has_nvml_library():
        return

    var dev = Device(0)
    for clock in dev.mem_clocks():
        # CHECK: Clock =
        print("Clock =", clock[])
    var driver_version = dev.get_driver_version()
    # CHECK: Driver version =
    print("Driver version =", String(driver_version))
    # CHECK: Driver major version =
    print("Driver major version =", driver_version.major())
