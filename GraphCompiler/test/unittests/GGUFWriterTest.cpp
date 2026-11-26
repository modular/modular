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

#include "GraphCompiler/Support/GGUF/GGUFWriter.h"
#include "GraphCompiler/Support/GGUF/GGUF.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

using namespace M;

namespace {

M::Driver::DeviceBufferRef
getDeserializedBuffer(const std::string &data,
                      const M::Driver::DeviceContextRef &ctx) {
  auto buffer = ctx->enqueueCreateBuffer(data.size());
  assert(!buffer.isError());
  auto copyResult = (*buffer)->copyFrom(data.data());
  assert(!copyResult.isError());
  auto syncResult = ctx->synchronize();
  assert(!syncResult.isError());
  return *buffer;
}

M::Driver::DeviceContextRef makeCPU() {
  auto ctx = M::Driver::DeviceContext::create(M::Driver::Device::cpuAPI);
  assert(!ctx.isError());
  return *ctx;
}

/// Re-reads a serialized file through the reader.
ErrorOr<GGUF> readBack(const std::string &serialized,
                       const M::Driver::DeviceContextRef &device) {
  auto buffer = getDeserializedBuffer(serialized, device);
  return GGUF::get(device, buffer);
}

} // namespace

TEST(GGUFWriter, RoundTripsMetadataAndTensors) {
  auto device = makeCPU();

  std::string floats(8 * sizeof(float), '\0');
  for (size_t i = 0; i < 8; ++i)
    reinterpret_cast<float *>(floats.data())[i] = static_cast<float>(i);
  // One Q4_K block of zeros: 144 bytes.
  std::string blocks(2 * 144, '\0');

  GGUFWriter writer;
  writer.setMetadataString("general.architecture", "example");
  writer.setMetadataUInt("example.block_count", 4);
  writer.setMetadataInt("example.head_count", -1);
  writer.setMetadataFloat("example.eps", 1e-5);
  writer.setMetadataBool("example.has_bias", true);
  ASSERT_FALSE(writer.addTensor("a", GGUF::F32, {2, 4}, floats).isError());
  ASSERT_FALSE(
      writer.addTensor("quantized", GGUF::Q4_K, {2, 256}, blocks).isError());

  auto serializedOr = writer.serialize();
  ASSERT_FALSE(serializedOr.isError()) << serializedOr.getError();

  auto ggufOr = readBack(*serializedOr, device);
  ASSERT_FALSE(ggufOr.isError()) << ggufOr.getError();
  const GGUF &gguf = *ggufOr;
  EXPECT_EQ(gguf.getVersion(), 3u);
  EXPECT_EQ(gguf.getAlignment(), 32u);

  auto architecture = gguf.getMetadata().get("general.architecture");
  ASSERT_NE(architecture, nullptr);
  EXPECT_EQ(*architecture->getAsString(), "example");
  auto blockCount = gguf.getMetadata().get("example.block_count");
  ASSERT_NE(blockCount, nullptr);
  EXPECT_EQ(*blockCount->getAsInteger(), 4);
  auto defaultEps = gguf.getMetadata().get("example.eps");
  ASSERT_NE(defaultEps, nullptr);
  EXPECT_DOUBLE_EQ(*defaultEps->getAsNumber(), 1e-5);
  auto hasBias = gguf.getMetadata().get("example.has_bias");
  ASSERT_NE(hasBias, nullptr);
  EXPECT_EQ(*hasBias->getAsBoolean(), true);

  auto aOr = gguf.getTensorInfo("a");
  ASSERT_FALSE(aOr.isError()) << aOr.getError();
  EXPECT_EQ(aOr->shape, SmallVector<size_t>({2, 4}));
  EXPECT_EQ(aOr->getSizeInBytes(), 32u);
  EXPECT_EQ(gguf.getTensorMemory(*aOr)->byteSize(), 32u);
  const auto *values =
      reinterpret_cast<const float *>(gguf.getTensorMemory(*aOr)->ptr());
  for (size_t i = 0; i < 8; ++i)
    EXPECT_EQ(values[i], static_cast<float>(i));

  auto quantizedOr = gguf.getTensorInfo("quantized");
  ASSERT_FALSE(quantizedOr.isError()) << quantizedOr.getError();
  EXPECT_TRUE(quantizedOr->isQuantized());
  EXPECT_EQ(quantizedOr->shape, SmallVector<size_t>({2, 256}));
  EXPECT_EQ(quantizedOr->getByteShape().get(), SmallVector<size_t>({2, 144}));
  EXPECT_EQ(quantizedOr->getSizeInBytes(), 288u);
}

TEST(GGUFWriter, WritesNonDefaultAlignment) {
  auto device = makeCPU();

  GGUFWriter writer(64);
  ASSERT_FALSE(
      writer.addTensor("a", GGUF::I32, {2}, std::string(8, '\0')).isError());

  auto serializedOr = writer.serialize();
  ASSERT_FALSE(serializedOr.isError()) << serializedOr.getError();

  auto ggufOr = readBack(*serializedOr, device);
  ASSERT_FALSE(ggufOr.isError()) << ggufOr.getError();
  EXPECT_EQ(ggufOr->getAlignment(), 64u);
}

TEST(GGUFWriter, RejectsBadTensorData) {
  GGUFWriter writer;
  // A float32 tensor of 4 elements is 16 bytes, not 15.
  EXPECT_TRUE(
      writer.addTensor("a", GGUF::F32, {4}, std::string(15, '\0')).isError());
  // The last dimension of a block-quantized tensor must be whole blocks.
  EXPECT_TRUE(writer.addTensor("b", GGUF::Q4_K, {100}, std::string(144, '\0'))
                  .isError());
  // Unknown tensor types are rejected rather than written unreadable.
  EXPECT_TRUE(writer.addTensor("c", 999, {1}, std::string(4, '\0')).isError());
  // Duplicate names would make the file ambiguous.
  ASSERT_FALSE(
      writer.addTensor("d", GGUF::F32, {1}, std::string(4, '\0')).isError());
  EXPECT_TRUE(
      writer.addTensor("d", GGUF::F32, {1}, std::string(4, '\0')).isError());
}

TEST(GGUFWriter, RejectsBadAlignment) {
  GGUFWriter writer(48);
  EXPECT_TRUE(writer.serialize().isError());
}
