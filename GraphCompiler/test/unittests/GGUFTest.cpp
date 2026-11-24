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

#include "GraphCompiler/Support/GGUF/GGUF.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

using namespace M;

namespace {

void appendUInt(std::string &out, uint64_t value, size_t numBytes) {
  for (size_t i = 0; i < numBytes; ++i)
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
}

void appendString(std::string &out, StringRef value) {
  appendUInt(out, value.size(), 8);
  out.append(value.data(), value.size());
}

/// The fixed prologue: magic, version, tensor count and metadata count.
void appendPrologue(std::string &out, uint32_t version, uint64_t tensorCount,
                    uint64_t metadataCount) {
  appendUInt(out, 0x46554747, 4);
  appendUInt(out, version, 4);
  appendUInt(out, tensorCount, 8);
  appendUInt(out, metadataCount, 8);
}

/// Appends one string-valued metadata entry.
void appendStringEntry(std::string &out, StringRef key, StringRef value) {
  appendString(out, key);
  appendUInt(out, 8 /* kString */, 4);
  appendString(out, value);
}

/// Appends one uint32-valued metadata entry.
void appendUInt32Entry(std::string &out, StringRef key, uint32_t value) {
  appendString(out, key);
  appendUInt(out, 4 /* kUInt32 */, 4);
  appendUInt(out, value, 4);
}

/// Appends one tensor descriptor. `shape` is the logical shape; the file
/// records it fastest-varying-first.
void appendTensorInfo(std::string &out, StringRef name, ArrayRef<size_t> shape,
                      uint32_t tensorType, uint64_t offset) {
  appendString(out, name);
  appendUInt(out, shape.size(), 4);
  for (size_t i = shape.size(); i > 0; --i)
    appendUInt(out, shape[i - 1], 8);
  appendUInt(out, tensorType, 4);
  appendUInt(out, offset, 8);
}

/// Pads `out` to the next multiple of `alignment`.
void padTo(std::string &out, size_t alignment) {
  while (out.size() % alignment != 0)
    out.push_back('\0');
}

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

/// A file with one I32 tensor of logical shape [5, 2] and two metadata
/// entries, one of them a uint32.
std::string makePlainFile() {
  std::string file;
  appendPrologue(file, 3, 1, 2);
  appendStringEntry(file, "general.architecture", "example");
  appendUInt32Entry(file, "general.alignment", 64);
  appendTensorInfo(file, "a", {5, 2}, GGUF::I32, 0);
  padTo(file, 64);
  for (int i = 0; i < 10; ++i)
    appendUInt(file, i, 4);
  return file;
}

/// A file with one Q4_K tensor of logical shape [2, 256]: two rows of 144
/// bytes each.
std::string makeQuantizedFile() {
  std::string file;
  appendPrologue(file, 3, 1, 0);
  appendTensorInfo(file, "quantized", {2, 256}, GGUF::Q4_K, 0);
  padTo(file, 32);
  file.append(2 * 144, '\x5a');
  return file;
}

} // namespace

TEST(GGUF, CanReadMetadata) {
  auto device = makeCPU();
  auto buffer = getDeserializedBuffer(makePlainFile(), device);

  auto ggufOr = GGUF::get(device, buffer);
  ASSERT_FALSE(ggufOr.isError()) << ggufOr.getError();
  const GGUF &gguf = *ggufOr;

  EXPECT_EQ(gguf.getVersion(), 3u);
  EXPECT_EQ(gguf.getAlignment(), 64u);
  EXPECT_EQ(gguf.getTensorInfo().size(), 1u);

  auto architecture = gguf.getMetadata().get("general.architecture");
  ASSERT_NE(architecture, nullptr);
  EXPECT_EQ(*architecture->getAsString(), "example");

  auto infoOr = gguf.getTensorInfo("a");
  ASSERT_FALSE(infoOr.isError()) << infoOr.getError();
  EXPECT_EQ(infoOr->shape, SmallVector<size_t>({5, 2}));
  EXPECT_EQ(infoOr->tensorType, GGUF::I32);
  EXPECT_EQ(infoOr->getSizeInBytes(), 40u);
  auto dtypeOr = infoOr->getDType();
  ASSERT_FALSE(dtypeOr.isError()) << dtypeOr.getError();
  EXPECT_EQ(*dtypeOr, DType::si32);
  EXPECT_FALSE(infoOr->isQuantized());

  // The tensor's data is addressable through the parent buffer.
  auto memory = gguf.getTensorMemory(*infoOr);
  EXPECT_EQ(memory->byteSize(), 40u);
  const auto *values = reinterpret_cast<const int32_t *>(memory->ptr());
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(values[i], i);
}

TEST(GGUF, CanReadQuantizedTensor) {
  auto device = makeCPU();
  auto buffer = getDeserializedBuffer(makeQuantizedFile(), device);

  auto ggufOr = GGUF::get(device, buffer);
  ASSERT_FALSE(ggufOr.isError()) << ggufOr.getError();
  const GGUF &gguf = *ggufOr;
  EXPECT_EQ(gguf.getAlignment(), 32u);

  auto infoOr = gguf.getTensorInfo("quantized");
  ASSERT_FALSE(infoOr.isError()) << infoOr.getError();
  EXPECT_EQ(infoOr->shape, SmallVector<size_t>({2, 256}));
  EXPECT_TRUE(infoOr->isQuantized());
  EXPECT_EQ(infoOr->getQuantizationBlockSize(), 256u);
  EXPECT_EQ(infoOr->getSizeInBytes(), 288u);

  // Block-quantized data is exposed as bytes with the last dimension scaled
  // from elements to blocks.
  auto byteShapeOr = infoOr->getByteShape();
  ASSERT_FALSE(byteShapeOr.isError()) << byteShapeOr.getError();
  EXPECT_EQ(*byteShapeOr, SmallVector<size_t>({2, 144}));
  auto dtypeOr = infoOr->getDType();
  ASSERT_FALSE(dtypeOr.isError()) << dtypeOr.getError();
  EXPECT_EQ(*dtypeOr, DType::ui8);
}

TEST(GGUF, RejectsBadMagic) {
  auto device = makeCPU();
  std::string file = makePlainFile();
  file[0] = '\x00';
  auto buffer = getDeserializedBuffer(file, device);

  auto ggufOr = GGUF::get(device, buffer);
  ASSERT_TRUE(ggufOr.isError());
  EXPECT_STREQ(ggufOr.getError(), "file does not start with the GGUF magic");
}

TEST(GGUF, RejectsUnknownVersion) {
  auto device = makeCPU();
  std::string file;
  appendPrologue(file, 1, 0, 0);
  auto buffer = getDeserializedBuffer(file, device);

  auto ggufOr = GGUF::get(device, buffer);
  ASSERT_TRUE(ggufOr.isError());
  EXPECT_STREQ(ggufOr.getError(),
               "unsupported GGUF version 1, expected 2 or 3");
}

TEST(GGUF, RejectsTruncatedFile) {
  auto device = makeCPU();
  std::string file = makePlainFile();
  file.resize(file.size() - 7);
  auto buffer = getDeserializedBuffer(file, device);

  auto ggufOr = GGUF::get(device, buffer);
  ASSERT_TRUE(ggufOr.isError());
}
