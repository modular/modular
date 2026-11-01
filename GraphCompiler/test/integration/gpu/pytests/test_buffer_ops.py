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
import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.engine.api import InferenceSession


def load_test_file(file_name):  # noqa: ANN001, ANN201
    testdata_dir = Path(
        os.environ.get("TESTDATA_DIRECTORY", Path(__file__).parent / "testdata")
    )
    return testdata_dir / file_name


@pytest.fixture
def slice_ir():  # noqa: ANN201
    return load_test_file("slice.mlir")


@pytest.fixture
def add_relu_ir():  # noqa: ANN201
    return load_test_file("add-relu.mlir")


@pytest.fixture
def buffer_create_return_ir():  # noqa: ANN201
    return load_test_file("buffer-create-return.mlir")


def test_slice(slice_ir) -> None:  # noqa: ANN001
    cuda = Accelerator()
    cpu = CPU()

    session = InferenceSession(devices=[cpu, cuda])
    model = session.load(slice_ir)

    buffer = Buffer.from_numpy(np.ones((5, 5), dtype=np.float32)).to(cuda)
    slice = Buffer.from_numpy(np.ones((3, 3), dtype=np.float32)).to(cuda)

    start = Buffer.from_numpy(np.array([0, 0], dtype=np.int64))
    stop = Buffer.from_numpy(np.array([5, 5], dtype=np.int64))
    step = Buffer.from_numpy(np.array([2, 2], dtype=np.int64))

    model.execute(buffer, slice, start, stop, step)

    buf_final = np.from_dlpack(buffer.to(cpu))

    expected = np.ones((5, 5), dtype=np.float32)
    expected[::2, ::2] = -2

    np.testing.assert_equal(expected, buf_final)


def test_add_relu(add_relu_ir) -> None:  # noqa: ANN001
    cuda = Accelerator()
    cpu = CPU()

    session = InferenceSession(devices=[cpu, cuda])
    model = session.load(add_relu_ir)

    buffer0 = Buffer.from_numpy(np.ones((2, 2), dtype=np.float32)).to(cuda)
    tensor0 = Buffer.from_numpy(
        np.array([[-1.1, 3.0], [-5.0, -0.5]], dtype=np.float32)
    ).to(cpu)

    model.execute(buffer0, tensor0)

    actual = np.from_dlpack(buffer0.to(cpu))

    expected = np.array([[0, 4.0], [0, 0.5]], dtype=np.float32)

    np.testing.assert_equal(expected, actual)


def test_buffer_create_return(buffer_create_return_ir) -> None:  # noqa: ANN001
    cuda = Accelerator()
    cpu = CPU()

    session = InferenceSession(devices=[cpu, cuda])
    model = session.load(buffer_create_return_ir)

    raw_buffer = np.ones([2, 2], dtype=np.float32)
    raw_tensor = np.ones([2, 2], dtype=np.float32) * 3

    buffer = Buffer.from_numpy(raw_buffer)
    tensor = Buffer.from_numpy(raw_tensor)

    results = model.execute(buffer, tensor)

    dev_result = np.from_dlpack(results[0].to(cpu))
    expected_dev_result = np.ones([2, 2], dtype=np.float32) * 4

    np.testing.assert_equal(expected_dev_result, dev_result)

    expected_cpu_buffer_result = np.ones([2, 2], dtype=np.float32) * 7

    np.testing.assert_equal(expected_cpu_buffer_result, raw_buffer)
