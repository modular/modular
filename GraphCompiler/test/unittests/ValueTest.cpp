//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Some unit tests for List, Dict, Tuple and friends.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Support/Value.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "GraphCompiler/Support/Dict.h"
#include "GraphCompiler/Support/FormatYAML.h"
#include "GraphCompiler/Support/List.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "llvm/Support/YAMLTraits.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace GML;

namespace {

struct ValueTest : ::testing::Test {};

static AsyncRT::CPUDeviceRef getCPUDevice() {
  return AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
                                       AsyncRT::CPUDeviceOptions().forDebug());
}

static Tensor makeTensor(AsyncRT::CPUDevice &cpuDevice, size_t size) {
  M::TensorSpec spec({size}, DType::f32);
  Tensor result =
      Tensor::createOwned(std::move(spec), cpuDevice, /*alignment=*/{});
  return result;
}

static List makeTensorList(AsyncRT::CPUDevice &cpuDevice, size_t size) {
  List list;
  for (size_t i = 0; i < size; ++i) {
    TensorSpec spec({i}, M::DType::f32);

    Tensor tensor = makeTensor(cpuDevice, i);
    list.getElements().push_back(AsyncRT::AnyAsyncValueRef::createReady<Tensor>(
        cpuDevice, std::move(tensor)));
  }

  return list;
}

template <typename T>
static std::string toYamlString(T t) {
  std::string yamlString;
  llvm::raw_string_ostream yamlStream(yamlString);
  llvm::yaml::Output yamlOutput(yamlStream);
  yamlOutput << t;
  yamlStream.flush();
  return yamlString;
}

static void assertTensorBufferEqual(const Tensor &lhs, const Tensor &rhs) {
  ASSERT_EQ(lhs.getBuffer(), rhs.getBuffer());
  ASSERT_EQ(lhs.getSizeInBytes(), rhs.getSizeInBytes());
}

TEST_F(ValueTest, ListTest) {
  auto cpuDevice = getCPUDevice();
  List list1 = makeTensorList(*cpuDevice, 3);

  List list2 = makeTensorList(*cpuDevice, 3);

  // Lists contain distinct tensors, and are not the same.
  ASSERT_NE(list1, list2);

  List list3;

  for (Variant &elem : list1.getElements()) {
    list3.getElements().push_back(
        AsyncRT::AnyAsyncValueRef::copy(elem.getValue().getPointer()));
  }

  // Lists contain the same tensors, and are the same.
  ASSERT_EQ(list1, list3);
}

TEST_F(ValueTest, TupleOfNoneTest) {
  auto cpuDevice = getCPUDevice();
  Tuple tuple1;
  Tuple tuple2;
  Tuple tuple3;

  tuple1.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice));
  tuple2.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice));
  tuple3.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice));

  tuple1.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice));
  tuple2.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice));

  ASSERT_EQ(tuple1, tuple2);
  ASSERT_NE(tuple1, tuple3);
  ASSERT_NE(tuple2, tuple3);

  ASSERT_EQ(std::to_string(tuple1), "Tuple(<none>, <none>)");
  ASSERT_EQ(std::to_string(tuple3), "Tuple(<none>)");
}

TEST_F(ValueTest, DictTest) {
  auto cpuDevice = getCPUDevice();
  Dict dict1;
  Dict dict2;
  Dict dict3;

  dict1.getElements()[AsyncRT::AnyAsyncValueRef::createReady<std::string>(
      *cpuDevice, "a")] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  dict2.getElements()[AsyncRT::AnyAsyncValueRef::createReady<std::string>(
      *cpuDevice, "a")] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  dict3.getElements()[AsyncRT::AnyAsyncValueRef::createReady<std::string>(
      *cpuDevice, "b")] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  Tensor tensor = makeTensor(*cpuDevice, 1);
  dict3.getElements()[AsyncRT::AnyAsyncValueRef::createReady<std::string>(
      *cpuDevice, "b")] =
      AsyncRT::AnyAsyncValueRef::createReady<Tensor>(*cpuDevice,
                                                     std::move(tensor));

  ASSERT_EQ(std::to_string(dict1), "Dict{\"a\": <none>}");
}

TEST_F(ValueTest, VariantDictKeys) {
  auto cpuDevice = getCPUDevice();
  List list;
  list.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<std::string>(*cpuDevice, "a"));

  list.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int64_t>(*cpuDevice, 2));

  Dict dict1;
  Dict dict2;

  dict1.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<List>(*cpuDevice, list))] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  dict2.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<List>(*cpuDevice, list))] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  ASSERT_EQ(dict1, dict2);
  ASSERT_EQ(std::to_string(dict1), "Dict{List[\"a\", 2]: <none>}");
}

TEST_F(ValueTest, YamlTest0) {
  auto cpuDevice = getCPUDevice();

  Dict dict1;

  dict1.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<std::string>(*cpuDevice, "a"))] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  dict1.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<double>(*cpuDevice, 1.0))] =
      AsyncRT::AnyAsyncValueRef::createReady<int64_t>(*cpuDevice, 5);

  std::string ystring = toYamlString(dict1);

  // Test YAML string.
  ASSERT_EQ(ystring, "---\n"
                     "dict:\n"
                     "  - key:\n"
                     "      string:          a\n"
                     "    value:\n"
                     "      none:            {}\n"
                     "  - key:\n"
                     "      double:          1\n"
                     "    value:\n"
                     "      long:            5\n"
                     "...\n");

  llvm::yaml::Input yin(ystring);

  Dict dict2;
  yin >> dict2;

  // Test round tripping through YAML.
  ASSERT_EQ(dict1, dict2);
}

TEST_F(ValueTest, YamlTest1) {
  auto cpuDevice = getCPUDevice();

  List list;
  list.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<std::string>(*cpuDevice, "a"));

  list.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int64_t>(*cpuDevice, 2));

  Dict dict1;

  dict1.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<List>(*cpuDevice, list))] =
      AsyncRT::AnyAsyncValueRef::createReady<None>(*cpuDevice);

  // Test YAML string.
  std::string ystring = toYamlString(dict1);

  ASSERT_EQ(ystring, "---\n"
                     "dict:\n"
                     "  - key:\n"
                     "      list:\n"
                     "        list:\n"
                     "          - string:          a\n"
                     "          - long:            2\n"
                     "    value:\n"
                     "      none:            {}\n"
                     "...\n");

  llvm::yaml::Input yin(ystring);

  Dict dict2;
  yin >> dict2;

  ASSERT_EQ(dict1, dict2);
}

TEST_F(ValueTest, YamlTest2) {
  auto cpuDevice = getCPUDevice();

  Tuple tuple;
  tuple.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<double>(*cpuDevice, 1.5f));

  tuple.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<bool>(*cpuDevice, true));

  Dict dict1;

  dict1.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<Tuple>(*cpuDevice, tuple))] =
      AsyncRT::AnyAsyncValueRef::createReady<double>(*cpuDevice, 3.5f);

  // Test YAML string.
  std::string ystring = toYamlString(dict1);

  ASSERT_EQ(ystring, "---\n"
                     "dict:\n"
                     "  - key:\n"
                     "      tuple:\n"
                     "        tuple:\n"
                     "          - double:          1.5\n"
                     "          - bool:            true\n"
                     "    value:\n"
                     "      double:          3.5\n"
                     "...\n");

  llvm::yaml::Input yin(ystring);

  Dict dict2;
  yin >> dict2;

  ASSERT_EQ(dict1, dict2);
}

// Tests serializing `VariantResultInfo` containing composed types to YAML.
TEST_F(ValueTest, ResultInfoTest) {
  auto cpuDevice = getCPUDevice();

  Tuple tuple;
  tuple.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<double>(*cpuDevice, 1.5f));

  tuple.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<bool>(*cpuDevice, true));

  Dict dict;

  dict.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<Tuple>(*cpuDevice, tuple))] =
      AsyncRT::AnyAsyncValueRef::createReady<TensorResultInfo>(
          *cpuDevice, "tensor1", "2x4xf32");

  std::shared_ptr<ResultInfo> resultInfo = std::make_shared<VariantResultInfo>(
      "outputName",
      AsyncRT::AnyAsyncValueRef::createReady<Dict>(*cpuDevice, dict));

  // Test YAML string.
  std::string ystring = toYamlString(resultInfo);

  ASSERT_EQ(ystring, "---\n"
                     "name:            outputName\n"
                     "variant:\n"
                     "  dict:\n"
                     "    dict:\n"
                     "      - key:\n"
                     "          tuple:\n"
                     "            tuple:\n"
                     "              - double:          1.5\n"
                     "              - bool:            true\n"
                     "        value:\n"
                     "          tensor-result-info:\n"
                     "            name:            tensor1\n"
                     "            shape:           2x4xf32\n"
                     "...\n");
}

// Tests converting composed types containing `Tensor` to composed types
// containing `TensorResultInfo` for tensor leaf nodes.
TEST_F(ValueTest, ResultConversionTest) {
  auto cpuDevice = getCPUDevice();

  // Create 4 different tensors.
  Tensor tensor = makeTensor(*cpuDevice, 8);
  AsyncRT::AnyAsyncValueRef wrappedTensor =
      AsyncRT::AnyAsyncValueRef::createReady<Tensor>(*cpuDevice,
                                                     std::move(tensor));

  List list = makeTensorList(*cpuDevice, 3);

  Dict dict;

  // Reuse some of the tensors occurring in the list in the dictionary as well.
  dict.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<std::string>(*cpuDevice, "a"))] =
      list.getElements()[0];

  dict.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<std::string>(*cpuDevice, "b"))] =
      wrappedTensor;

  dict.getElements()[Variant(
      AsyncRT::AnyAsyncValueRef::createReady<std::string>(*cpuDevice, "c"))] =
      list.getElements()[2];

  Tuple tuple;
  tuple.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<List>(*cpuDevice, list));
  tuple.getElements().push_back(
      AsyncRT::AnyAsyncValueRef::createReady<Dict>(*cpuDevice, dict));

  // Conversion for YAML and binary serialization.
  AsyncRT::AnyAsyncValueRef in =
      AsyncRT::AnyAsyncValueRef::createReady<Tuple>(*cpuDevice, tuple);
  ProcessedResults processedResults;
  AsyncRT::AnyAsyncValueRef out =
      M::getResultForBinaryOutput(in, processedResults).takeValue();

  std::shared_ptr<ResultInfo> resultInfo =
      std::make_shared<VariantResultInfo>("outputName", out);

  // Check that each tensor is only captured once.
  ASSERT_EQ(processedResults.getTensors().size(), 4ULL);

  // Check all tensors are collected in depth-first preorder, and named
  // correctly.
  assertTensorBufferEqual(*processedResults.getTensors().at("__tensor_0__"),
                          list.getElements()[0].getValue().get<Tensor>());
  assertTensorBufferEqual(*processedResults.getTensors().at("__tensor_1__"),
                          list.getElements()[1].getValue().get<Tensor>());
  assertTensorBufferEqual(*processedResults.getTensors().at("__tensor_2__"),
                          list.getElements()[2].getValue().get<Tensor>());
  assertTensorBufferEqual(*processedResults.getTensors().at("__tensor_3__"),
                          wrappedTensor.get<Tensor>());

  // Test YAML string.
  std::string ystring = toYamlString(resultInfo);

  ASSERT_EQ(ystring, "---\n"
                     "name:            outputName\n"
                     "variant:\n"
                     "  tuple:\n"
                     "    tuple:\n"
                     "      - list:\n"
                     "          list:\n"
                     "            - tensor-result-info:\n"
                     "                name:            __tensor_0__\n"
                     "                shape:           '0xf32'\n"
                     "            - tensor-result-info:\n"
                     "                name:            __tensor_1__\n"
                     "                shape:           1xf32\n"
                     "            - tensor-result-info:\n"
                     "                name:            __tensor_2__\n"
                     "                shape:           2xf32\n"
                     "      - dict:\n"
                     "          dict:\n"
                     "            - key:\n"
                     "                string:          a\n"
                     "              value:\n"
                     "                tensor-result-info:\n"
                     "                  name:            __tensor_0__\n"
                     "                  shape:           '0xf32'\n"
                     "            - key:\n"
                     "                string:          b\n"
                     "              value:\n"
                     "                tensor-result-info:\n"
                     "                  name:            __tensor_3__\n"
                     "                  shape:           8xf32\n"
                     "            - key:\n"
                     "                string:          c\n"
                     "              value:\n"
                     "                tensor-result-info:\n"
                     "                  name:            __tensor_2__\n"
                     "                  shape:           2xf32\n"
                     "...\n");
}

} // namespace
