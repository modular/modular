# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import math

import numpy as np

"""
This file contains the op implementations for numpy.
"""


class CommonImpls:
    def add(self, x, y):  # noqa: ANN001, ANN201
        return x + y

    def sub(self, x, y):  # noqa: ANN001, ANN201
        return x - y

    def mul(self, x, y):  # noqa: ANN001, ANN201
        return x * y

    def lt(self, x, y):  # noqa: ANN001, ANN201
        return x < y

    def lte(self, x, y):  # noqa: ANN001, ANN201
        return x <= y

    def gt(self, x, y):  # noqa: ANN001, ANN201
        return x > y

    def gte(self, x, y):  # noqa: ANN001, ANN201
        return x >= y

    def eq(self, x, y):  # noqa: ANN001, ANN201
        return x == y

    def neq(self, x, y):  # noqa: ANN001, ANN201
        return x != y


# Numpy implementation of each primitive.
class NumpyImpl(CommonImpls):
    def constant(self, values, dtype=None):  # noqa: ANN001, ANN201
        if dtype:
            return np.array(values, dtype=dtype)

        return np.array(values)

    def ceil(self, x):  # noqa: ANN001, ANN201
        return np.ceil(x)

    def floor(self, x):  # noqa: ANN001, ANN201
        return np.floor(x)

    def transpose(self, input, permute):  # noqa: ANN001, ANN201
        return np.transpose(input, permute)

    def bmm(self, x, y):  # noqa: ANN001, ANN201
        return np.matmul(x, y)

    def range(self, start, stop, step):  # noqa: ANN001, ANN201
        return np.arange(start, stop, step)

    def reshape(self, x, shape):  # noqa: ANN001, ANN201
        return x.reshape(shape)

    def split(self, x, split_sizes, axis):  # noqa: ANN001, ANN201
        # To match the TF api exactly we need to manually slice up the tensor.
        to_return = []

        # Create an identity slice along every axis, basically [:, :, :]
        # or [...]. We update this inplace to insert the actual dimension
        # we split along later.
        base_slice = [slice(0, 123141231, 1)] * x.ndim

        # Iterate through the splits and add a slice along the dimension we
        # are splitting over.
        split_point = 0
        for size in split_sizes:
            base_slice[axis] = slice(split_point, split_point + size, 1)
            split_point = split_point + size

            # The slice will now be [:, :, slice(...), :] where axis = 2 and
            # the slice is the specific slice of this output of the split.
            to_return.append(x[tuple(base_slice)])

        return to_return

    def sum(self, x, axis):  # noqa: ANN001, ANN201
        return np.sum(x, axis=axis, keepdims=True)

    def max(self, x, y):  # noqa: ANN001, ANN201
        return np.maximum(x, y)

    def gather(self, x, indices, axis):  # noqa: ANN001, ANN201
        return np.take(x, indices, axis=axis)

    def strided_slice(self, x, start, stop, step):  # noqa: ANN001, ANN201
        slice_to_apply = []
        for i in range(len(start)):
            slice_to_apply.append(slice(start[i], stop[i], step[i]))
        return x[tuple(slice_to_apply)]

    def abs(self, x):  # noqa: ANN001, ANN201
        return np.abs(x)

    def erf(self, x):  # noqa: ANN001, ANN201
        apply = lambda t: math.erf(t)
        return np.vectorize(apply)(x)

    def exp(self, x):  # noqa: ANN001, ANN201
        return np.exp(x)

    def log(self, x):  # noqa: ANN001, ANN201
        return np.log(x)

    def log1p(self, x):  # noqa: ANN001, ANN201
        return np.log1p(x)

    def min(self, x, limit):  # noqa: ANN001, ANN201
        return np.minimum(x, limit)

    def select(self, cond, x, y):  # noqa: ANN001, ANN201
        return np.where(cond, x, y)

    def sin(self, x):  # noqa: ANN001, ANN201
        return np.sin(x)

    def cos(self, x):  # noqa: ANN001, ANN201
        return np.cos(x)

    def tanh(self, x):  # noqa: ANN001, ANN201
        return np.tanh(x)

    def is_nan(self, x):  # noqa: ANN001, ANN201
        return np.isnan(x)

    def is_inf(self, x):  # noqa: ANN001, ANN201
        return np.isinf(x)

    def logical_and(self, x, y):  # noqa: ANN001, ANN201
        return np.logical_and(x, y)

    def logical_or(self, x, y):  # noqa: ANN001, ANN201
        return np.logical_or(x, y)

    def logical_xor(self, x, y):  # noqa: ANN001, ANN201
        return np.logical_xor(x, y)

    def rsqrt(self, x):  # noqa: ANN001, ANN201
        return 1.0 / np.sqrt(x)
