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

from imageio.v3 import imread, imwrite
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)


def main():
    if accelerator_count() == 0:
        raise RuntimeError(
            "The image_pipeline example currently only runs on GPU"
        )

    device = Accelerator()

    img = imread("dogs.jpg")

    color_tensor = TensorType(
        DType.uint8,
        shape=img.shape,
        device=DeviceRef.from_device(device),
    )

    gray_tensor = TensorType(
        DType.uint8,
        shape=[img.shape[0], img.shape[1]],
        device=DeviceRef.from_device(device),
    )

    graph = Graph(
        "image_pipeline",
        input_types=[color_tensor],
        custom_extensions=[Path(__file__).parent / "kernels"],
    )

    def grayscale(x: TensorValue) -> TensorValue:
        return ops.custom(
            name="grayscale",
            values=[x],
            out_types=[gray_tensor],
        )[0].tensor

    def brightness(x: TensorValue, brightness: float) -> TensorValue:
        return ops.custom(
            name="brightness",
            values=[x, ops.constant(brightness, DType.float32)],
            out_types=[x.type],
        )[0].tensor

    def blur(x: TensorValue, blur_size: int) -> TensorValue:
        return ops.custom(
            name="blur",
            values=[x, ops.constant(blur_size, DType.int64)],
            out_types=[x.type],
        )[0].tensor

    with graph:
        grayed = grayscale(graph.inputs[0])
        brightened = brightness(grayed, brightness=1.5)
        blurred = blur(brightened, blur_size=8)
        graph.output(blurred)

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    img_dev = Tensor.from_numpy(img).to(device)

    result = model.execute(img_dev)[0]
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    imwrite("dogs_out.jpg", result.to_numpy())


if __name__ == "__main__":
    main()
