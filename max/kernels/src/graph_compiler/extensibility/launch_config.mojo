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
"""Defines the launch configuration a declarative kernel returns."""

from max.gpu.host import Dim


trait GridBlockLaunchConfig:
    """Requires the grid and block dims of a kernel launch.

    A declarative kernel's `execute_configuration` returns a value of this
    trait, and the graph compiler launches the kernel's `execute_device` with
    these dims.
    """

    def get_grid_dim(self) -> Dim:
        """Returns the dims of the launch's grid of blocks.

        Returns:
            The grid dims.
        """
        ...

    def get_block_dim(self) -> Dim:
        """Returns the dims of each block of the launch.

        Returns:
            The block dims.
        """
        ...
