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

from max import engine
from max.graph import Graph, TensorType, ops
from max.tensor import Tensor, TensorShape, TensorSpec
from memory import UnsafePointer, memcpy
from python import Python as py
from python import PythonObject


def load_model_weights() -> PythonObject:
    np = py.import_module("numpy")
    fin = py.evaluate('open("model_weights.npy", mode="rb")')
    model_weights = np.load(
        fin, allow_pickle=True
    ).item()  # note this is of type PythonObject
    fin.close()
    print("python type of model_weights:", py.type(model_weights))
    for item in model_weights.items():
        print(item[0], item[1].shape, py.type(item[1]))

    return model_weights


@always_inline
fn numpy_data_pointer[
    dtype: DType
](numpy_array: PythonObject) raises -> UnsafePointer[Scalar[dtype]]:
    return numpy_array.__array_interface__["data"][0].unsafe_get_as_pointer[
        dtype
    ]()


@always_inline
fn memcpy_from_numpy(array: PythonObject, tensor: Tensor) raises:
    src = numpy_data_pointer[tensor.type](array)
    dst = tensor._ptr
    length = tensor.num_elements()
    memcpy(dst, src, length)


@always_inline
fn numpy_to_tensor[dtype: DType](array: PythonObject) raises -> Tensor[dtype]:
    shape = List[Int]()
    array_shape = array.shape
    for dim in array_shape:
        shape.append(dim)

    out = Tensor[dtype](shape)
    memcpy_from_numpy(array, out)
    return out^


def build_mnist_graph(
    fc1w: Tensor[DType.float32],
    fc1b: Tensor[DType.float32],
    fc2w: Tensor[DType.float32],
    fc2b: Tensor[DType.float32],
) -> Graph:
    # Note: "batch" is a symbolic dim which is known ahead of time vs dynamic dim
    graph = Graph(TensorType(DType.float32, "batch", 28 * 28))
    # PyTorch linear is defined as: x W^T + b so we need to transpose the weights
    fc1 = (
        graph[0] @ ops.transpose(graph.constant(fc1w), 1, 0)
    ) + graph.constant(fc1b)

    relu = ops.relu(fc1)

    fc2 = (relu @ ops.transpose(graph.constant(fc2w), 1, 0)) + graph.constant(
        fc2b
    )
    out = ops.softmax(fc2)  # adding explicit softmax for inference prob
    graph.output(out)
    graph.verify()
    return graph


def load_mnist_test_data() -> PythonObject:
    torchvision = py.import_module("torchvision")
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=None, download=False
    )
    return test_dataset


def preprocess(image: PythonObject) -> PythonObject:
    transforms = py.import_module("torchvision.transforms")
    image_tensor = transforms.ToTensor()(image)
    image_tensor_normalized = transforms.Normalize(
        Python.tuple(0.5), Python.tuple(0.5)
    )(image_tensor)
    reshaped_image = image_tensor_normalized.reshape(1, 28 * 28).numpy()
    return reshaped_image


def argmax(t: Tensor) -> List[Int]:
    var res = List[Int](capacity=t.dim(0))
    for i in range(t.dim(0)):
        var max_val = Scalar[t.type].MIN
        var max_idx = 0
        for j in range(t.dim(1)):
            if t[i, j] > max_val:
                max_val = t[i, j]
                max_idx = j
        res.append(max_idx)
    return res


def main():
    weights_dict = load_model_weights()
    fc1w = numpy_to_tensor[DType.float32](weights_dict["fc1.weight"])
    fc1b = numpy_to_tensor[DType.float32](weights_dict["fc1.bias"])
    fc2w = numpy_to_tensor[DType.float32](weights_dict["fc2.weight"])
    fc2b = numpy_to_tensor[DType.float32](weights_dict["fc2.bias"])

    mnist_graph = build_mnist_graph(fc1w^, fc1b^, fc2w^, fc2b^)
    session = engine.InferenceSession()
    model = session.load(mnist_graph)

    for name in model.get_model_input_names():
        print("input:", name[])

    for name in model.get_model_output_names():
        print("output:", name[])

    correct = 0
    total = 0
    # use batch size of 1 in this example
    test_dataset = load_mnist_test_data()
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        image = item[0]
        label = item[1]

        preprocessed_image = preprocess(image)

        output = model.execute("input0", preprocessed_image)
        probs = output.get[DType.float32]("output0")

        predicted = argmax(probs)[0]

        correct += Int(predicted == label)
        total += 1

    print(
        "Accuracy of the network on the 10000 test images:",
        100 * correct / total,
        "%",
    )
