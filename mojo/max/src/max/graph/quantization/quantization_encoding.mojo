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

"""Defines the interface for quantized types."""

from max.tensor import Tensor


trait QuantizationEncoding:
    """Describes the encoding for a data type that can be quantized.

    Any type that conforms to this trait implicitly knows the relationship
    between the buffer and tensor layout for its quantization encoding.
    In particular, the `quantize()` function takes in a tensor with a logical
    shape in terms of its elements. Then it returns a uint8 tensor with a
    different shape that instead describes the shape of the bytes storage
    buffer after applying the quantization encoding.
    """

    @staticmethod
    def quantize(tensor: Tensor[DType.float32]) -> Tensor[DType.uint8]:
        """Quantizes the full-precision tensor to the quantized
        type associated with this `QuantizationEncoding` instance.

        Args:
            tensor: Full-precision tensor to quantize.

        Returns:
            A `Tensor` quantized to the quantized storage format of this
            `QuantizationEncoding` instance. The tensor datatype is `uint8`
            because this is simply a bytes buffer. The actual data structure
            in that buffer depends on the encoding.
        """
        ...

    @staticmethod
    fn id() -> String:
        """Returns a unique string identifier for this quantization encoding."""
        ...
