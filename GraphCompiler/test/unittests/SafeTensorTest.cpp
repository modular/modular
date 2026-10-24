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

#include "GraphCompiler/Support/SafeTensor/SafeTensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

using namespace M;

template <size_t N>
M::Driver::DeviceBufferRef
getDeserializedBuffer(const char (&data)[N],
                      const M::Driver::DeviceContextRef &ctx) {
  // There's got to be an easier way to do this
  auto buffer = ctx->enqueueCreateBuffer(N);
  assert(!buffer.isError());
  auto copyResult = (*buffer)->copyFrom(data);
  assert(!copyResult.isError());
  auto syncResult = ctx->synchronize();
  assert(!syncResult.isError());
  return *buffer;
}

static std::string toString(const llvm::json::Value &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  value.print(os);
  return str;
}

static std::string toString(const SafeTensor::TensorInfo &info) {
  return toString(info.toJSON());
}

static M::Driver::DeviceContextRef makeCPU() {
  auto ctx = M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
  assert(!ctx.isError());
  return *ctx;
}

TEST(SafeTensor, CanReadMetadata) {
  auto device = makeCPU();
  auto buffer = getDeserializedBuffer(
      "\x59\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":["
      "2,2],\"data_offsets\":[0,16]},\"__metadata__\":{\"foo\":\"bar\"}}"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      device);

  auto safeTensorOr = SafeTensor::get(device, buffer);
  ASSERT_FALSE(safeTensorOr.isError()) << safeTensorOr.getError();

  ASSERT_EQ(toString(safeTensorOr->getMetadata()), "{\"foo\":\"bar\"}");
}

TEST(SafeTensor, CanReadTensorInfo) {
  auto device = makeCPU();
  auto buffer = getDeserializedBuffer(
      "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,"
      "2],\"data_offsets\":[0,16]}}"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      device);

  auto safeTensorOr = SafeTensor::get(device, buffer);
  ASSERT_FALSE(safeTensorOr.isError()) << safeTensorOr.getError();

  // llvm::outs() << "metadata = " << safeTensor.getMetadata() << "\n";
  auto tensorInfo = safeTensorOr->getTensorInfo("test");
  ASSERT_FALSE(tensorInfo.isError()) << tensorInfo.getError();
  ASSERT_EQ(toString(std::move(*tensorInfo)),
            "{\"data_offsets\":[0,16],\"dtype\":\"I32\",\"shape\":[2,2]}");

  ASSERT_EQ(tensorInfo->dtype, DType::si32);
  ASSERT_EQ(tensorInfo->shape, SmallVector<size_t>({2, 2}));
  ASSERT_EQ(tensorInfo->offsets.first, 0);
  ASSERT_EQ(tensorInfo->offsets.second, 16);

  ASSERT_EQ(tensorInfo->getSizeInBytes(), 16);
}

TEST(SafeTensor, HeaderTooSmall) {
  auto device = makeCPU();
  auto buffer = getDeserializedBuffer("", device);

  auto safeTensorOr = SafeTensor::get(device, buffer);
  ASSERT_TRUE(safeTensorOr.isError()) << safeTensorOr.getError();
  ASSERT_STREQ(safeTensorOr.getError(), "file is too small to be a safetensor");
}

TEST(SafeTensor, HeaderTooLarge) {
  auto device = makeCPU();
  auto buffer = getDeserializedBuffer(
      "<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,"
      "2],\"data_offsets\":[0,16]}}"
      "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
      device);

  auto safeTensorOr = SafeTensor::get(device, buffer);
  ASSERT_TRUE(safeTensorOr.isError()) << safeTensorOr.getError();
  ASSERT_STREQ(safeTensorOr.getError(),
               "header length is greater than the max header size");
}
