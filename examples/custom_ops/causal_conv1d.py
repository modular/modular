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

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def create_graph(
    name: str,
    custom_op_name: str,
    dtype: DType,
    in_shape: list,
    w_shape: list,
    b_shape: list,
    mojo_kernels: Path,
    device: DeviceRef,
    threads: int,
    elements: int,
    width: int,
) -> Graph:
    output2d = [in_shape[0] * in_shape[1], in_shape[2]]

    with Graph(
        name,
        input_types=[
            TensorType(DType.float32, shape=in_shape, device=device),
            TensorType(DType.float32, shape=w_shape, device=device),
            TensorType(DType.float32, shape=b_shape, device=device),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph_xxx:
        input, weights, bias = graph_xxx.inputs
        if dtype == DType.bfloat16:
            input = input.tensor.cast(DType.bfloat16)
            weights = weights.tensor.cast(DType.bfloat16)
            bias = bias.tensor.cast(DType.bfloat16)
        results = ops.custom(
            name=custom_op_name,
            parameters={
                "threads": threads,
                "elements": elements,
                "width": width,
            },
            values=[input, weights, bias],
            out_types=[
                TensorType(
                    dtype,
                    shape=in_shape,
                    device=device,
                ),
                TensorType(
                    dtype,
                    shape=output2d,
                    device=device,
                ),
            ],
        )
        if dtype == DType.bfloat16:
            results[0] = results[0].tensor.cast(DType.float32)
        graph_xxx.output(*results)
        return graph_xxx


def main():
    mojo_kernels = Path(__file__).parent / "kernels"

    nBatches = 8
    nChannels = 4
    sequenceLength = 1024
    width = 4

    device = CPU()

    np.random.seed(123)
    I_n = np.random.randn(nBatches, nChannels, sequenceLength).astype("f")
    W_n = np.random.randn(nChannels, width).astype("f")
    B_n = np.zeros(nChannels).astype("f")
    I = Tensor.from_numpy(I_n).to(device)
    W = Tensor.from_numpy(W_n).to(device)
    B = Tensor.from_numpy(B_n).to(device)

    threads = 1
    elements = 4
    dtype = DType.float32

    in_shape = [nBatches, nChannels, sequenceLength]
    w_shape = [nChannels, width]
    b_shape = [nChannels]
    device_cpu = CPU()

    # Set up an inference session for running the graph.
    device = device_cpu
    session = InferenceSession(devices=[device])
    graph_cpu = create_graph(
        "causl_conv1d_cpu",
        "causal_conv1d_cpu",
        dtype,
        in_shape,
        w_shape,
        b_shape,
        mojo_kernels,
        DeviceRef.from_device(device_cpu),
        threads,
        elements,
        width,
    )
    # Compile the graph.
    model = session.load(graph_cpu)
    output_cpu = model.execute(I, W, B)[0]
    assert isinstance(output_cpu, Tensor)
    output_cpu_np = output_cpu.to_numpy()

    if accelerator_count() != 0:
        threads = 32
        elements = 8
        width = 4
        dtype = DType.bfloat16
        in_shape = [nBatches, nChannels, sequenceLength]
        w_shape = [nChannels, width]
        b_shape = [nChannels]

        device_gpu = Accelerator()

        graph_gpu = create_graph(
            "causl_conv1d_gpu",
            "causal_conv1d_v1",
            dtype,
            in_shape,
            w_shape,
            b_shape,
            mojo_kernels,
            DeviceRef.from_device(device_gpu),
            threads,
            elements,
            width,
        )
        session = InferenceSession(devices=[device_gpu])

        # Compile the graph.
        model = session.load(graph_gpu)
        I_gpu = I.to(device_gpu)
        W_gpu = W.to(device_gpu)
        B_gpu = B.to(device_gpu)
        output_gpu = model.execute(I_gpu, W_gpu, B_gpu)[0]
        assert isinstance(output_gpu, Tensor)
        output_gpu_np = output_gpu.to_numpy().astype(np.float32)

        if (
            np.allclose(
                output_gpu_np,
                output_cpu_np,
                rtol=1e-05,
                atol=1e-01,
                equal_nan=False,
            )
            == True
        ):
            print("Sucess!")
        else:
            print("Failed!")
            print("GPU results: ", output_gpu_np)
            print("CPU results: ", output_cpu_np)
            print("Input: ", I_n)
            print("Weights: ", W_n)
            print("Differences:", np.isclose(output_cpu_np, output_gpu_np))


if __name__ == "__main__":
    main()
