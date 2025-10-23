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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import os

import max._core.driver

class SafeTensor:
    """A parser for the HuggingFace SafeTensors format."""

    def keys(self) -> list[str]:
        """Returns the list of tensor keys present."""

    def get_tensor(self, name: str) -> max._core.driver.Tensor:
        """Returns a tensor with a given key"""

    def __enter__(self) -> SafeTensor: ...
    def __exit__(
        self,
        exc_type: object | None = None,
        exc_value: object | None = None,
        traceback: object | None = None,
    ) -> None: ...

def safe_open(
    filepath: str | os.PathLike, device: max._core.driver.Device | None = None
) -> SafeTensor:
    """
    Loads and parses a SafeTensor file from the given path onto the given
    device. Defaults to loading on the CPU.

    NOTE: Currently only implemented for CPUs.
    """
