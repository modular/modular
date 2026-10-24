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

import inspect
import random
import sys

import numpy as np

"""
This file contains the test implementations. Each test class added to this file
can then be picked up by the tools just using its name. For instance:

python export_mo_graph.py -n BasicTranspose

Will pick up the class named `BasicTranspose` in this file.
"""


# Inherit from this to get some sensible defaults.
class TestBase:
    @staticmethod
    def compare_results(ours, theirs):  # noqa: ANN001, ANN205
        return np.allclose(ours, theirs, rtol=1e-05, atol=1e-08, equal_nan=True)

    @staticmethod
    def num_runs() -> int:
        return 25


# Basic transpose test case.
class BasicTranspose(TestBase):
    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [([4, 2, 1, 6], [4])]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        # Create a test of rank 3 with and a permute vector of rank 1.
        return [
            ((None, None, None, None), np.float32),
            ((4,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        permute = np.arange(len(shapes[0])).astype(np.int32)
        permute = np.random.permutation(permute)
        return [
            np.random.uniform(-2.0, 2.0, shapes[0]).astype(np.float32),
            permute,
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        return impl.transpose(inputs[0], inputs[1])


# A normal batch matmul.
class BatchMatmul(TestBase):
    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [([5, 11, 6], [1, 6, 23])]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None, None), np.float32),
            ((None, None, None), np.float32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        return [
            np.random.uniform(-2.0, 2.0, shapes[0]).astype(np.float32),
            np.random.uniform(-2.0, 2.0, shapes[1]).astype(np.float32),
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        return impl.bmm(inputs[0], inputs[1])


# A normal batch matmul.
class BatchMatmulTrans(BatchMatmul):
    @staticmethod
    def compare_results(ours, theirs):  # noqa: ANN001, ANN205
        return np.allclose(ours, theirs, rtol=1e-05, atol=1e-06, equal_nan=True)

    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [([5, 11, 6], [1, 23, 6])]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        permute = impl.constant((0, 2, 1))
        trans = impl.transpose(inputs[1], permute)
        return impl.bmm(inputs[0], trans)


class BatchMatmulRelu(BatchMatmul):
    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        x = impl.bmm(inputs[0], inputs[1])
        return impl.max(x, 0)


# Test for range
class Range(TestBase):
    @staticmethod
    def compare_results(ours, theirs):  # noqa: ANN001, ANN205
        return np.array_equal(ours, theirs)

    @staticmethod
    def num_runs() -> int:
        return 100

    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [(3,)]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [((3,), np.int32)]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        start = np.random.randint(-64, 64)
        stop = np.random.randint(-64, 64)
        diff = stop - start
        if diff == 0:
            start, stop, step = (-3, 3, 1)
        elif diff > 0:
            step = np.random.randint(1, diff) if diff != 1 else 1
        else:
            step = np.random.randint(diff, -1) if diff != -1 else -1

        return [np.array([start, stop, step], dtype=np.int32)]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        # Until `mt` supports the input type we have to slice them out #18187.
        start, stop, step = impl.split(inputs[0], [1, 1, 1], 0)
        start = impl.reshape(start, shape=())
        stop = impl.reshape(stop, shape=())
        step = impl.reshape(step, shape=())
        return impl.range(start, stop, step)


class Split(TestBase):
    @staticmethod
    def compare_results(ours, theirs):  # noqa: ANN001, ANN205
        return np.array_equal(ours, theirs)

    @staticmethod
    def num_runs() -> int:
        return 300

    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [([20, 12, 12], [6], [1])]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None, None), np.int32),
            ((6,), np.int32),
            ((1,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        start = np.random.randint(-128, 128, size=shapes[0]).astype(np.int32)
        # The split axis is now a compile-time attribute on mo.split (the graph
        # splits along a fixed axis), so pin the axis input to match the
        # attribute baked into split.mlir (axis = 0).
        axis = np.array([0], dtype=np.int32)

        dim_splitting_along = start.shape[axis[0]]
        split_sizes = []
        splits_left_to_make = shapes[1][0]

        # Create the vector representing the size along each dimension to split
        # ensuring we are splitting to at least 1 at each point.
        for i in range(shapes[1][0] - 1):  # noqa: B007
            assert dim_splitting_along > 0
            assert dim_splitting_along > splits_left_to_make
            assert splits_left_to_make > 0

            split_size = np.random.randint(
                1, dim_splitting_along - splits_left_to_make
            )
            split_sizes.append(split_size)

            dim_splitting_along -= split_size
            splits_left_to_make -= 1

        # Last dim is however many are left
        split_sizes.append(dim_splitting_along)

        # If the first number is big we might end up with all ones so shuffle
        # a bit so it's not always the first one.
        np.random.shuffle(split_sizes)

        split_sizes = np.array(split_sizes).astype(np.int32)  # type: ignore

        return [start, split_sizes, axis]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        axis = impl.reshape(inputs[2], shape=())
        return impl.split(inputs[0], inputs[1], axis)


def get_test_case(target_name):  # noqa: ANN001, ANN201
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if name == target_name:
                return obj
    raise Exception(f"No object of name {name} found")


class TransposeReduce(TestBase):
    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [
            ([65, 67, 3, 17], [1], [4]),
        ]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None, None, None), np.float32),
            ((1,), np.int32),
            ((4,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        permute = np.arange(len(shapes[0])).astype(np.int32)
        permute = np.random.permutation(permute)
        # use randint for input to avoid fp tolerance issues
        input = np.random.randint(-5, 5, size=shapes[0]).astype(np.float32)
        # The reduction axis is now a compile-time attribute on mo.reduce.add
        # (the graph reduces along a fixed axis), so pin the axis input to
        # match the attribute baked into transpose_reduce.mlir (axis = 3).
        return [
            input,
            np.array([3], dtype=np.int32),
            permute,
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        reduce_axis = impl.reshape(inputs[1], shape=())
        trans = impl.transpose(inputs[0], inputs[2])
        return impl.sum(trans, reduce_axis)


class GatherReduceConstAxis(TestBase):
    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [
            ([30, 20], [10, 20]),
        ]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None), np.float32),
            ((None, None), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        return [
            # Use randint for input to avoid fp tolerance issues
            np.random.randint(-5, 5, size=shapes[0]).astype(np.float32),
            np.random.randint(0, shapes[0][0], size=shapes[1]).astype(np.int32),
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        reduce_axis = impl.reshape(impl.constant((1,)), shape=())
        gathered = impl.gather(inputs[0], inputs[1], 0)
        return impl.sum(gathered, reduce_axis)


class GatherOfGathers(TestBase):
    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        # 1st array to gather from, 2nd first indices to gather, then indices to gather from that.. and so on. Last input is the axis
        return [
            ([300, 100], [50], [25], [10], [1], [1]),
        ]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None), np.float32),
            ((50,), np.int32),
            ((25,), np.int32),
            ((10,), np.int32),
            ((1,), np.int32),
            ((1,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        axis = random.randint(0, len(shapes[0]) - 1)

        return [
            np.random.randint(-1000, 1000, size=shapes[0]).astype(np.float32),
            np.random.randint(0, shapes[0][axis], size=shapes[1]).astype(
                np.int32
            ),
            np.random.randint(0, shapes[1][0], size=shapes[2]).astype(np.int32),
            np.random.randint(0, shapes[2][0], size=shapes[3]).astype(np.int32),
            np.random.randint(0, shapes[3][0], size=shapes[4]).astype(np.int32),
            axis,
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        axis = impl.reshape(inputs[-1], shape=())
        g_1 = impl.gather(inputs[0], inputs[1], axis)
        g_2 = impl.gather(g_1, inputs[2], axis)
        g_3 = impl.gather(g_2, inputs[3], axis)
        return impl.gather(g_3, inputs[4], axis)


class MultiViewFusionBase(TestBase):
    @staticmethod
    def num_runs() -> int:
        return 100

    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        # Input, transpose permutation, slice indices
        return [
            ([5, 10, 5], [3], [3], [3], [3]),
        ]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None, None), np.float32),
            ((3,), np.int32),
            ((3,), np.int32),
            ((3,), np.int32),
            ((3,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        num_dims = len(shapes[0])

        # Create a transpose permutation.
        permute = np.arange(num_dims).astype(np.int32)
        permute = np.random.permutation(permute)

        # Walk each axis forwards or backwards, and express the bounds as
        # Python-style negative indices half the time. numpy is the oracle
        # here, so every encoding below has to mean the same to it as to MO.
        start, stop, step = [], [], []
        for dim in shapes[0]:
            dim = int(dim)
            magnitude = int(np.random.randint(1, 3))
            backwards = bool(np.random.randint(0, 2))
            first = int(np.random.randint(0, dim))

            if backwards:
                count = int(np.random.randint(1, first // magnitude + 2))
                last = first - (count - 1) * magnitude
                stop_idx = last - magnitude
            else:
                max_count = (dim - 1 - first) // magnitude + 1
                count = int(np.random.randint(1, max_count + 1))
                last = first + (count - 1) * magnitude
                stop_idx = last + magnitude

            # A backwards walk ending at 0 has no non-negative stop: -1 would
            # mean the last element, so use the out-of-range value that
            # normalizes to -1 on both sides.
            if stop_idx < 0:
                stop_val = -(dim + 1)
            elif 0 < stop_idx < dim and np.random.randint(0, 2):
                stop_val = stop_idx - dim
            else:
                stop_val = stop_idx

            start_val = first - dim if np.random.randint(0, 2) else first

            step.append(-magnitude if backwards else magnitude)
            start.append(start_val)
            stop.append(stop_val)

        return [
            np.random.uniform(-2.0, 2.0, shapes[0]).astype(np.float32),
            np.array(start, dtype=np.int32),
            np.array(stop, dtype=np.int32),
            np.array(step, dtype=np.int32),
            permute,
        ]


class MultiViewFusion(MultiViewFusionBase):
    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        # Slice input
        sliced = impl.strided_slice(inputs[0], inputs[1], inputs[2], inputs[3])
        transposed = impl.transpose(sliced, inputs[-1])
        return transposed + transposed


class MultiViewFusionNoAdd(MultiViewFusionBase):
    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        # Slice input
        sliced = impl.strided_slice(inputs[0], inputs[1], inputs[2], inputs[3])
        return impl.transpose(sliced, inputs[-1])


class Elementwise(TestBase):
    @staticmethod
    def num_runs() -> int:
        return 100

    @staticmethod
    def compare_results(ours, theirs):  # noqa: ANN001, ANN205
        return np.allclose(ours, theirs, rtol=1e-04, atol=1e-07, equal_nan=True)

    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [([20, 5, 1, 6],)]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None, None, None), np.float32),
            ((None, None, None, None), np.float32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        return [
            np.random.uniform(-20.0, 20.0, shapes[0]).astype(np.float32),
            np.random.uniform(-20.0, 20.0, shapes[0]).astype(np.float32),
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        x = inputs[0]
        tanh = impl.tanh(inputs[1])
        sub = impl.sub(x, tanh)
        mul = impl.mul(sub, x)
        erf = impl.erf(mul)
        abs_ = impl.abs(x)
        exp = impl.exp(abs_)

        max = impl.max(x, impl.constant(0.5, np.float32))
        o = impl.add(exp, max)

        min = impl.min(x, impl.constant(-0.5, np.float32))
        o = impl.add(o, min)

        cnd = impl.gt(x, exp)
        sel1 = impl.select(cnd, x, exp)

        # Doesn't work?
        # cnd = impl.gte(o, exp)
        # sel2 = impl.select(cnd, x, exp)
        # add = impl.add(sel1, sel2)

        add = impl.add(sel1, o)

        log = impl.log(add)
        ceil = impl.ceil(add)
        floor = impl.floor(add)
        log1p = impl.log1p(add)
        rsqrt = impl.rsqrt(add)
        add = impl.add(add, log)
        add = impl.add(add, rsqrt)
        add = impl.add(add, log1p)
        add = impl.add(add, ceil)
        add = impl.add(add, floor)

        # TODO: Lower in TF.
        # sin = impl.sin(add)
        # cos = impl.cos(add)
        # mul = impl.mul(sin, cos)

        return impl.add(erf, add)


# Test the case of static shape as this changes the broadcast & fusion logic.
class ElementwiseStatic(Elementwise):
    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [((20, 5, 1, 6), np.float32), ((20, 5, 1, 6), np.float32)]


class ElementwiseBoolOps(TestBase):
    @staticmethod
    def num_runs() -> int:
        return 100

    @staticmethod
    def input_sizes_to_test():  # noqa: ANN205
        return [([20, 5, 1, 6],)]

    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [
            ((None, None, None, None), bool),
            ((None, None, None, None), np.float32),
        ]

    @staticmethod
    def generate_inputs(shapes):  # noqa: ANN001, ANN205
        float_vals = [0.0, 1.0, np.inf, np.nan, -np.inf]
        return [
            np.random.rand(*shapes[0]) > 0.5,
            np.random.choice(float_vals, size=shapes[0]).astype(np.float32),
        ]

    @staticmethod
    def run(inputs, impl):  # noqa: ANN001, ANN205
        bool_input = inputs[0]
        float_input = inputs[1]
        isnan = impl.is_nan(float_input)
        o = impl.logical_and(isnan, bool_input)
        isinf = impl.is_inf(float_input)
        o = impl.eq(o, isinf)
        return impl.neq(o, bool_input)


class ElementwiseBoolOpsStatic(ElementwiseBoolOps):
    @staticmethod
    def get_input_specs():  # noqa: ANN205
        return [((20, 5, 1, 6), bool), ((20, 5, 1, 6), np.float32)]
