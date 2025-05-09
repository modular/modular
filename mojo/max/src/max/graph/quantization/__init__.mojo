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

"""APIs to quantize graph tensors.

This package includes a generic quantization encoding interface and some
quantization encodings that conform to it, such as bfloat16 and Q4_0 encodings.

The main interface for defining a new quantized type is
`QuantizationEncoding.quantize()`. This takes a full-precision tensor
represented as float32 and quantizes it according to the encoding. The
resulting quantized tensor is represented as a bytes tensor. For that reason,
the `QuantizationEncoding` must know how to translate between the tensor shape
and its corresponding quantized buffer shape.

For example, this code quantizes a tensor with the Q4_0 encoding:

```mojo
from max.tensor import Tensor
from max.graph.quantization import Q4_0Encoding

var tensor: Tensor[DType.float32]
# Initialize `tensor`.

# Quantize using the `Q4_0` quantization encoding.
var quantized: Tensor[DType.uint8] = Q4_0Encoding.quantize(tensor)

# Now `quantized` is packed according to the `Q4_0` encoding and can be
# used to create graph constants and serialized to disk.
```

Specific ops in the MAX Graph API that use quantization can be found in the
[`ops.quantized_ops`](/max/api/mojo/graph/ops/quantized_ops) module. You
can also add a quantized node in your graph with
[`Graph.quantize()`](/max/api/mojo/graph/graph/Graph#quantize).

To save the quantized tensors to disk, use
[`graph.checkpoint.save()`](/max/api/mojo/graph/checkpoint/save_load/save).

The Graph API does not support model
training, so you must import your model weights, load them as
[`Tensor`](/mojo/stdlib/tensor/tensor/Tensor) values, and then quantize them.
"""

from .encodings import (
    BFloat16Encoding,
    Float32Encoding,
    Q4_0Encoding,
    Q4_KEncoding,
    Q5_KEncoding,
    Q6_KEncoding,
)
from .encodings import _BlockQ4K as BlockQ4K
from .encodings import _BlockQ5K as BlockQ5K
from .encodings import _BlockQ6K as BlockQ6K
from .encodings import _BlockQ40 as BlockQ40
from .quantization_encoding import QuantizationEncoding
