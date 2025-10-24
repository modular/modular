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
"""Arccosine (inverse cosine) operation."""

from ..graph import DeviceRef, TensorValue


def acos(x: TensorValue) -> TensorValue:
    """Computes the arccosine (inverse cosine) of the input tensor.

    Returns values in the range [0, π] for inputs in [-1, 1].
    Uses the optimized math.acos() function from the Mojo standard library,
    which employs a Remez approximation with domain splitting for improved accuracy.

    Creates a new op node to compute the elementwise arccosine of a
    symbolic tensor and adds it to the graph, returning the symbolic result.

    .. code-block:: python

        def acos_graph():
            input_type = TensorType(dtype=DType.float32, shape=(3,), device=DeviceRef.CPU())

            with Graph("acos_graph", input_types=(input_type,)) as graph:
                x = graph.inputs[0]
                out = ops.acos(x)
                graph.output(out)

    Args:
        x: Input tensor with values in [-1, 1]. If values are outside this
           domain, they will be clamped to the valid range.

    Returns:
        Arccosine of the input in radians [0, π]. The result will have:
        - the same dtype as the input
        - the same shape as the input

    Raises:
        Error: If the symbol doesn't represent a tensor value.
        Error: If the input is not a floating-point dtype.
    """
    from .. import dtype_promotion  # Avoid circular import

    x = dtype_promotion._restrict_to_strong_dtypes(x)
    device = x.device
    return ops.custom(
        name="mo.acos",
        device=device,
        values=[x],
        out_types=TensorType(
            dtype=x.dtype,
            shape=x.tensor.shape,
            device=DeviceRef.from_device(device),
        ),
    )[0].tensor
