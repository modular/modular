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
"""Identity layer that passes through input unchanged."""

from __future__ import annotations

from max.graph import TensorValue

from .layer import Module


class Identity(Module):
    """Identity layer that passes through input unchanged.

    This layer is useful for skipping certain operations (like normalization)
    in specific architectures such as EAGLE speculative decoding, where the
    draft model receives already-normalized hidden states from the target model.

    Example:
        >>> from max.nn import Identity
        >>> identity = Identity()
        >>> # In graph construction:
        >>> output = identity(input_tensor)  # output == input_tensor
    """

    def __call__(self, x: TensorValue) -> TensorValue:
        """Pass through the input unchanged.

        Args:
            x: Input tensor value.

        Returns:
            The same tensor value as input.
        """
        return x
