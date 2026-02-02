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

"""Extension for max.dtype to support additional attributes."""

from numpy import finfo as np_finfo

from .dtype import DType


class finfo:
    """A numerical properties of a floating point max.dtype.DType.

    This class mimics torch.finfo behavior without torch dependency,
    including support for bfloat16.

    NOTE: Currently, it's applied through patching.
    This extension is better to be implemented in dtype library itself.
    """

    def __init__(self, dtype: DType):
        """Initialize finfo for a given max.dtype.DType.

        Args:
            dtype: The data type to get limits for.
        """
        if dtype == DType.bfloat16:
            self.min = -3.38953e38
            self.max = 3.38953e38
            self.bits = 16
            self.eps = 0.0078125
            self.resolution = 0.01
            self.tiny = 1.17549e-38
            self.dtype = "bfloat16"
        else:
            np_finfo_obj = np_finfo(dtype.to_numpy())
            self.min = float(np_finfo_obj.min)
            self.max = float(np_finfo_obj.max)
            self.bits = np_finfo_obj.bits
            self.eps = float(np_finfo_obj.eps)
            self.resolution = float(np_finfo_obj.resolution)
            self.tiny = float(np_finfo_obj.tiny)
            self.dtype = str(np_finfo_obj.dtype)


DType.finfo = finfo  # type: ignore[attr-defined]
