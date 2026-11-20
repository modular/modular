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

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/DeviceContext/LaunchTraceRegistry.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

using namespace M::Driver;

namespace {

// Helper function to convert JSON value to string
static std::string jsonToString(const llvm::json::Value &json) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << json;
  return os.str();
}

// Test fixture for trace metadata formatting tests
class TraceMetadataTest : public ::testing::Test {};

// Test: KernelLaunchMetadata for kernel launch operation
TEST_F(TraceMetadataTest, KernelLaunchMetadataBasic) {
  KernelLaunchMetadata metadata("vec_add", /*hashValue=*/0x1234567890abcdefULL,
                                /*gridX=*/4, /*gridY=*/1, /*gridZ=*/1,
                                /*blockX=*/256, /*blockY=*/1, /*blockZ=*/1,
                                /*sharedMemBytes=*/0, /*numAttrs=*/0,
                                /*streamIndex=*/0);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::KernelLaunch);
  EXPECT_EQ(metadata.kernelName, "vec_add");
  EXPECT_EQ(metadata.semanticHash, 0x1234567890abcdefULL);
  EXPECT_EQ(metadata.gridX, 4u);
  EXPECT_EQ(metadata.gridY, 1u);
  EXPECT_EQ(metadata.gridZ, 1u);
  EXPECT_EQ(metadata.blockX, 256u);
  EXPECT_EQ(metadata.blockY, 1u);
  EXPECT_EQ(metadata.blockZ, 1u);
  EXPECT_EQ(metadata.sharedMemBytes, 0u);
  EXPECT_EQ(metadata.numAttrs, 0u);
  EXPECT_EQ(metadata.streamIndex, 0u);
}

// Test: KernelLaunchMetadata with shared memory and attributes
TEST_F(TraceMetadataTest, KernelLaunchWithSharedMemAndAttrs) {
  KernelLaunchMetadata metadata(
      "matmul_kernel", /*hashValue=*/0xabcdef1234567890ULL,
      /*gridX=*/8, /*gridY=*/8, /*gridZ=*/1,
      /*blockX=*/16, /*blockY=*/16, /*blockZ=*/1,
      /*sharedMemBytes=*/49152, /*numAttrs=*/2, /*streamIndex=*/0);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::KernelLaunch);
  EXPECT_EQ(metadata.kernelName, "matmul_kernel");
  EXPECT_EQ(metadata.sharedMemBytes, 49152u);
  EXPECT_EQ(metadata.numAttrs, 2u);
}

// Test: KernelLaunchMetadata for memcpy Host-to-Device
TEST_F(TraceMetadataTest, MemcpyHtoD) {
  KernelLaunchMetadata metadata(
      "cuMemcpyHtoD", /*hashValue=*/0x1111111111111111ULL,
      KernelLaunchMetadata::MemcpyKind::HtoD, /*byteSize=*/4096);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::Memcpy);
  EXPECT_EQ(metadata.kernelName, "cuMemcpyHtoD");
  EXPECT_EQ(metadata.semanticHash, 0x1111111111111111ULL);
  EXPECT_EQ(metadata.memcpyKind, KernelLaunchMetadata::MemcpyKind::HtoD);
  EXPECT_EQ(metadata.memcpyByteSize, 4096u);
}

// Test: KernelLaunchMetadata for memcpy Device-to-Host
TEST_F(TraceMetadataTest, MemcpyDtoH) {
  KernelLaunchMetadata metadata(
      "cuMemcpyDtoH", /*hashValue=*/0x2222222222222222ULL,
      KernelLaunchMetadata::MemcpyKind::DtoH, /*byteSize=*/8192);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::Memcpy);
  EXPECT_EQ(metadata.kernelName, "cuMemcpyDtoH");
  EXPECT_EQ(metadata.memcpyKind, KernelLaunchMetadata::MemcpyKind::DtoH);
  EXPECT_EQ(metadata.memcpyByteSize, 8192u);
}

// Test: KernelLaunchMetadata for memcpy Device-to-Device
TEST_F(TraceMetadataTest, MemcpyDtoD) {
  KernelLaunchMetadata metadata(
      "cuMemcpyDtoD", /*hashValue=*/0x3333333333333333ULL,
      KernelLaunchMetadata::MemcpyKind::DtoD, /*byteSize=*/16384);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::Memcpy);
  EXPECT_EQ(metadata.kernelName, "cuMemcpyDtoD");
  EXPECT_EQ(metadata.memcpyKind, KernelLaunchMetadata::MemcpyKind::DtoD);
  EXPECT_EQ(metadata.memcpyByteSize, 16384u);
}

// Test: KernelLaunchMetadata for memset
TEST_F(TraceMetadataTest, Memset) {
  KernelLaunchMetadata metadata("cuMemset",
                                /*hashValue=*/0x4444444444444444ULL,
                                /*byteSize=*/8192,
                                /*value=*/0x00000000ULL, /*valSize=*/4);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::Memset);
  EXPECT_EQ(metadata.kernelName, "cuMemset");
  EXPECT_EQ(metadata.semanticHash, 0x4444444444444444ULL);
  EXPECT_EQ(metadata.memsetByteSize, 8192u);
  EXPECT_EQ(metadata.memsetValue, 0x00000000ULL);
  EXPECT_EQ(metadata.memsetValueSize, 4u);
}

// Test: KernelLaunchMetadata for memset with non-zero value
TEST_F(TraceMetadataTest, MemsetNonZero) {
  KernelLaunchMetadata metadata("cuMemset",
                                /*hashValue=*/0x5555555555555555ULL,
                                /*byteSize=*/4096,
                                /*value=*/0xdeadbeefULL, /*valSize=*/4);

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::Memset);
  EXPECT_EQ(metadata.memsetByteSize, 4096u);
  EXPECT_EQ(metadata.memsetValue, 0xdeadbeefULL);
  EXPECT_EQ(metadata.memsetValueSize, 4u);
}

// Test: Comparison operators - equality
TEST_F(TraceMetadataTest, ComparisonEquality) {
  KernelLaunchMetadata meta1("kernel1", 0x1111111111111111ULL);
  KernelLaunchMetadata meta2("kernel1", 0x1111111111111111ULL);
  KernelLaunchMetadata meta3("kernel2", 0x1111111111111111ULL);
  KernelLaunchMetadata meta4("kernel1", 0x2222222222222222ULL);

  EXPECT_EQ(meta1, meta2);
  EXPECT_NE(meta1, meta3);
  EXPECT_NE(meta1, meta4);
}

// Test: Comparison operators - less than (for sorting)
TEST_F(TraceMetadataTest, ComparisonLessThan) {
  KernelLaunchMetadata meta1("aaa", 0x1111111111111111ULL);
  KernelLaunchMetadata meta2("bbb", 0x1111111111111111ULL);
  KernelLaunchMetadata meta3("aaa", 0x2222222222222222ULL);

  // Sort by name first
  EXPECT_LT(meta1, meta2);
  EXPECT_FALSE(meta2 < meta1);

  // Then by hash
  EXPECT_LT(meta1, meta3);
  EXPECT_FALSE(meta3 < meta1);
}

// Test: Default constructor
TEST_F(TraceMetadataTest, DefaultConstructor) {
  KernelLaunchMetadata metadata;

  EXPECT_EQ(metadata.kind, KernelLaunchMetadata::OperationKind::KernelLaunch);
  EXPECT_EQ(metadata.kernelName, "");
  EXPECT_EQ(metadata.semanticHash, 0u);
  EXPECT_EQ(metadata.gridX, 0u);
  EXPECT_EQ(metadata.blockX, 0u);
  EXPECT_EQ(metadata.memcpyKind, KernelLaunchMetadata::MemcpyKind::None);
  EXPECT_EQ(metadata.memcpyByteSize, 0u);
  EXPECT_EQ(metadata.memsetByteSize, 0u);
}

// ===== Formatting Tests =====

// Test: Format kernel launch operation
TEST_F(TraceMetadataTest, FormatKernelLaunch) {
  KernelLaunchMetadata metadata("vec_add", 0x1234567890abcdefULL,
                                /*gridX=*/4, /*gridY=*/1, /*gridZ=*/1,
                                /*blockX=*/256, /*blockY=*/1, /*blockZ=*/1,
                                /*sharedMemBytes=*/0, /*numAttrs=*/0,
                                /*streamIndex=*/0);

  std::string formatted = metadata.format(1);

  // Debug: print the actual output
  // Debug: std::cout <<"Formatted output:\n" << formatted << std::endl;

  // Check that the formatted string contains expected components
  EXPECT_NE(formatted.find("vec_add"), std::string::npos);
  EXPECT_NE(formatted.find("Grid: (4, 1, 1)"), std::string::npos);
  EXPECT_NE(formatted.find("Block: (256, 1, 1)"), std::string::npos);
  EXPECT_NE(formatted.find("[count: 1]"), std::string::npos);
}

// Test: Format kernel launch with shared memory and attributes
TEST_F(TraceMetadataTest, FormatKernelLaunchWithSharedMemAndAttrs) {
  KernelLaunchMetadata metadata("matmul_kernel", 0xabcdef1234567890ULL,
                                /*gridX=*/8, /*gridY=*/8, /*gridZ=*/1,
                                /*blockX=*/16, /*blockY=*/16, /*blockZ=*/1,
                                /*sharedMemBytes=*/49152, /*numAttrs=*/2,
                                /*streamIndex=*/0);

  std::string formatted = metadata.format(1);

  EXPECT_NE(formatted.find("matmul_kernel"), std::string::npos);
  EXPECT_NE(formatted.find("Grid: (8, 8, 1)"), std::string::npos);
  EXPECT_NE(formatted.find("Block: (16, 16, 1)"), std::string::npos);
  EXPECT_NE(formatted.find("Shared: 49152 bytes"), std::string::npos);
  EXPECT_NE(formatted.find("Attrs: 2"), std::string::npos);
}

// Test: Format memcpy Host-to-Device
TEST_F(TraceMetadataTest, FormatMemcpyHtoD) {
  KernelLaunchMetadata metadata("cuMemcpyHtoD", 0x1111111111111111ULL,
                                KernelLaunchMetadata::MemcpyKind::HtoD,
                                /*byteSize=*/4096);

  std::string formatted = metadata.format(1);

  EXPECT_NE(formatted.find("cuMemcpyHtoD"), std::string::npos);
  EXPECT_NE(formatted.find("Kind: Host-to-Device"), std::string::npos);
  EXPECT_NE(formatted.find("Size: 4096 bytes"), std::string::npos);
  EXPECT_NE(formatted.find("[count: 1]"), std::string::npos);
}

// Test: Format memcpy Device-to-Host
TEST_F(TraceMetadataTest, FormatMemcpyDtoH) {
  KernelLaunchMetadata metadata("cuMemcpyDtoH", 0x2222222222222222ULL,
                                KernelLaunchMetadata::MemcpyKind::DtoH,
                                /*byteSize=*/8192);

  std::string formatted = metadata.format(2);

  EXPECT_NE(formatted.find("cuMemcpyDtoH"), std::string::npos);
  EXPECT_NE(formatted.find("Kind: Device-to-Host"), std::string::npos);
  EXPECT_NE(formatted.find("Size: 8192 bytes"), std::string::npos);
  EXPECT_NE(formatted.find("[count: 2]"), std::string::npos);
}

// Test: Format memcpy Device-to-Device
TEST_F(TraceMetadataTest, FormatMemcpyDtoD) {
  KernelLaunchMetadata metadata("cuMemcpyDtoD", 0x3333333333333333ULL,
                                KernelLaunchMetadata::MemcpyKind::DtoD,
                                /*byteSize=*/16384);

  std::string formatted = metadata.format(3);

  EXPECT_NE(formatted.find("cuMemcpyDtoD"), std::string::npos);
  EXPECT_NE(formatted.find("Kind: Device-to-Device"), std::string::npos);
  EXPECT_NE(formatted.find("Size: 16384 bytes"), std::string::npos);
  EXPECT_NE(formatted.find("[count: 3]"), std::string::npos);
}

// Test: Format memset with zero value
TEST_F(TraceMetadataTest, FormatMemsetZero) {
  KernelLaunchMetadata metadata("cuMemset", 0x4444444444444444ULL,
                                /*byteSize=*/8192,
                                /*value=*/0x00000000ULL, /*valSize=*/4);

  std::string formatted = metadata.format(1);
  // Debug: std::cout <<"FormatMemsetZero output:\n" << formatted << std::endl;

  EXPECT_NE(formatted.find("cuMemset"), std::string::npos);
  EXPECT_NE(formatted.find("Size: 8192 bytes"), std::string::npos);
  EXPECT_NE(formatted.find("valSize: 4"), std::string::npos);
}

// Test: Format memset with non-zero value
TEST_F(TraceMetadataTest, FormatMemsetNonZero) {
  KernelLaunchMetadata metadata("cuMemset", 0x5555555555555555ULL,
                                /*byteSize=*/4096,
                                /*value=*/0xdeadbeefULL, /*valSize=*/4);

  std::string formatted = metadata.format(5);
  // Debug: std::cout <<"FormatMemsetNonZero output:\n" << formatted <<
  // std::endl;

  EXPECT_NE(formatted.find("cuMemset"), std::string::npos);
  EXPECT_NE(formatted.find("Size: 4096 bytes"), std::string::npos);
  EXPECT_NE(formatted.find("valSize: 4"), std::string::npos);
  EXPECT_NE(formatted.find("[count: 5]"), std::string::npos);
}

// Test: Format with different count values
TEST_F(TraceMetadataTest, FormatWithDifferentCounts) {
  KernelLaunchMetadata metadata("kernel", 0x1234ULL, /*gridX=*/1, /*gridY=*/1,
                                /*gridZ=*/1, /*blockX=*/1, /*blockY=*/1,
                                /*blockZ=*/1, /*sharedMemBytes=*/0,
                                /*numAttrs=*/0, /*streamIndex=*/0);

  std::string formatted10 = metadata.format(10);
  std::string formatted100 = metadata.format(100);

  EXPECT_NE(formatted10.find("[count: 10]"), std::string::npos);
  EXPECT_NE(formatted100.find("[count: 100]"), std::string::npos);
}

// ===== Documentation of Expected Error Message Formats =====
//
// These tests document the expected format for error messages when trace
// mismatches are detected. The actual formatting is done by
// LaunchTraceDiff::formatOperation() in DeviceContext.cpp.
//
// Expected format for KERNEL LAUNCH:
//     - vec_add (hash: 1311768467463790320)
//         Grid: (4, 1, 1), Block: (256, 1, 1), Shared: 0 bytes, Attrs: 0
//         [count: 1]
//
// With shared memory and attributes:
//     - matmul_kernel (hash: 12379813738877118864)
//         Grid: (8, 8, 1), Block: (16, 16, 1), Shared: 49152 bytes, Attrs: 2
//         [count: 1]
//
// Expected format for MEMCPY Host-to-Device:
//     - cuMemcpyHtoD (hash: 1229782938247303441)
//         Kind: Host-to-Device, Size: 4096 bytes [count: 1]
//
// Expected format for MEMCPY Device-to-Host:
//     - cuMemcpyDtoH (hash: 2459565876494606882)
//         Kind: Device-to-Host, Size: 8192 bytes [count: 1]
//
// Expected format for MEMCPY Device-to-Device:
//     - cuMemcpyDtoD (hash: 3689348814741910323)
//         Kind: Device-to-Device, Size: 16384 bytes [count: 1]
//
// Expected format for MEMSET with zero value:
//     - cuMemset (hash: 4919131752989213764)
//         Size: 8192 bytes, Value: 0x00000000 (valSize: 4) [count: 1]
//
// Expected format for MEMSET with non-zero value:
//     - cuMemset (hash: 6148914691236517205)
//         Size: 4096 bytes, Value: 0xdeadbeef (valSize: 4) [count: 1]
//
// Complete error message format:
// Graph 0 launch trace mismatch:
//   Missing operations (in baseline but not in current):
//     <formatted operation entries>
//   Unexpected operations (in current but not in baseline):
//     <formatted operation entries>

// ===== JSON Formatting Tests =====

// Test: JSON format for kernel launch
TEST_F(TraceMetadataTest, FormatJSONKernelLaunch) {
  KernelLaunchMetadata metadata("vec_add", /*hashValue=*/0x1234567890abcdefULL,
                                /*gridX=*/4, /*gridY=*/1, /*gridZ=*/1,
                                /*blockX=*/256, /*blockY=*/1, /*blockZ=*/1,
                                /*sharedMemBytes=*/1024, /*numAttrs=*/2,
                                /*streamIndex=*/0);

  std::string json = jsonToString(metadata.formatJSON());

  // Verify key fields are present in JSON
  EXPECT_NE(json.find("\"kernelName\":\"vec_add\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"KernelLaunch\""), std::string::npos);
  EXPECT_NE(json.find("\"semanticHash\""), std::string::npos);
  EXPECT_NE(json.find("\"grid\""), std::string::npos);
  EXPECT_NE(json.find("\"block\""), std::string::npos);
  EXPECT_NE(json.find("\"sharedMemBytes\":1024"), std::string::npos);
  EXPECT_NE(json.find("\"numAttrs\":2"), std::string::npos);
}

// Test: JSON format for memcpy HtoD
TEST_F(TraceMetadataTest, FormatJSONMemcpyHtoD) {
  KernelLaunchMetadata metadata(
      "cuMemcpyHtoD", /*hashValue=*/0x1111111111111111ULL,
      KernelLaunchMetadata::MemcpyKind::HtoD, /*byteSize=*/4096);

  std::string json = jsonToString(metadata.formatJSON());

  EXPECT_NE(json.find("\"kernelName\":\"cuMemcpyHtoD\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memcpy\""), std::string::npos);
  EXPECT_NE(json.find("\"memcpyKind\":\"HtoD\""), std::string::npos);
  EXPECT_NE(json.find("\"byteSize\":4096"), std::string::npos);
}

// Test: JSON format for memcpy DtoH
TEST_F(TraceMetadataTest, FormatJSONMemcpyDtoH) {
  KernelLaunchMetadata metadata(
      "cuMemcpyDtoH", /*hashValue=*/0x2222222222222222ULL,
      KernelLaunchMetadata::MemcpyKind::DtoH, /*byteSize=*/8192);

  std::string json = jsonToString(metadata.formatJSON());

  EXPECT_NE(json.find("\"kernelName\":\"cuMemcpyDtoH\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memcpy\""), std::string::npos);
  EXPECT_NE(json.find("\"memcpyKind\":\"DtoH\""), std::string::npos);
  EXPECT_NE(json.find("\"byteSize\":8192"), std::string::npos);
}

// Test: JSON format for memcpy DtoD
TEST_F(TraceMetadataTest, FormatJSONMemcpyDtoD) {
  KernelLaunchMetadata metadata(
      "cuMemcpyDtoD", /*hashValue=*/0x3333333333333333ULL,
      KernelLaunchMetadata::MemcpyKind::DtoD, /*byteSize=*/16384);

  std::string json = jsonToString(metadata.formatJSON());

  EXPECT_NE(json.find("\"kernelName\":\"cuMemcpyDtoD\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memcpy\""), std::string::npos);
  EXPECT_NE(json.find("\"memcpyKind\":\"DtoD\""), std::string::npos);
  EXPECT_NE(json.find("\"byteSize\":16384"), std::string::npos);
}

// Test: JSON format for memset with zero value
TEST_F(TraceMetadataTest, FormatJSONMemsetZero) {
  KernelLaunchMetadata metadata("cuMemset",
                                /*hashValue=*/0x4444444444444444ULL,
                                /*byteSize=*/8192,
                                /*value=*/0x00000000ULL, /*valSize=*/4);

  std::string json = jsonToString(metadata.formatJSON());

  EXPECT_NE(json.find("\"kernelName\":\"cuMemset\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memset\""), std::string::npos);
  EXPECT_NE(json.find("\"byteSize\":8192"), std::string::npos);
  EXPECT_NE(json.find("\"value\":\"0x"),
            std::string::npos); // Check for hex format
  EXPECT_NE(json.find("\"valueSize\":4"), std::string::npos);
}

// Test: JSON format for memset with non-zero value
TEST_F(TraceMetadataTest, FormatJSONMemsetNonZero) {
  KernelLaunchMetadata metadata("cuMemset",
                                /*hashValue=*/0x5555555555555555ULL,
                                /*byteSize=*/4096,
                                /*value=*/0xdeadbeefULL, /*valSize=*/4);

  std::string json = jsonToString(metadata.formatJSON());

  EXPECT_NE(json.find("\"kernelName\":\"cuMemset\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memset\""), std::string::npos);
  EXPECT_NE(json.find("\"byteSize\":4096"), std::string::npos);
  EXPECT_NE(json.find("\"value\":\"0xdeadbeef\""), std::string::npos);
  EXPECT_NE(json.find("\"valueSize\":4"), std::string::npos);
}

// ===== LaunchTraceDiff JSON Formatting Tests =====

// Test: Empty LaunchTraceDiff (no mismatch)
TEST_F(TraceMetadataTest, LaunchTraceDiffJSONEmpty) {
  LaunchTraceDiff diff;

  EXPECT_TRUE(diff.empty());

  std::string json = jsonToString(diff.formatJSON(0));

  EXPECT_NE(json.find("\"graphIdx\":0"), std::string::npos);
  EXPECT_NE(json.find("\"hasMismatch\":false"), std::string::npos);
  EXPECT_NE(json.find("\"missingOps\":[]"), std::string::npos);
  EXPECT_NE(json.find("\"unexpectedOps\":[]"), std::string::npos);
}

// Test: LaunchTraceDiff with missing operations only
TEST_F(TraceMetadataTest, LaunchTraceDiffJSONMissingOps) {
  LaunchTraceDiff diff;

  // Add missing operations
  KernelLaunchMetadata op1("vec_add", 0x1111111111111111ULL, 4, 1, 1, 256, 1, 1,
                           0, 0, 0);
  KernelLaunchMetadata op2("cuMemcpyHtoD", 0x2222222222222222ULL,
                           KernelLaunchMetadata::MemcpyKind::HtoD, 4096);

  diff.missingOps[op1] = 2;
  diff.missingOps[op2] = 1;

  EXPECT_FALSE(diff.empty());

  std::string json = jsonToString(diff.formatJSON(1));

  EXPECT_NE(json.find("\"graphIdx\":1"), std::string::npos);
  EXPECT_NE(json.find("\"hasMismatch\":true"), std::string::npos);
  EXPECT_NE(json.find("\"missingOps\""), std::string::npos);
  EXPECT_NE(json.find("\"vec_add\""), std::string::npos);
  EXPECT_NE(json.find("\"cuMemcpyHtoD\""), std::string::npos);
  EXPECT_NE(json.find("\"count\":2"), std::string::npos);
  EXPECT_NE(json.find("\"count\":1"), std::string::npos);
}

// Test: LaunchTraceDiff with unexpected operations only
TEST_F(TraceMetadataTest, LaunchTraceDiffJSONUnexpectedOps) {
  LaunchTraceDiff diff;

  // Add unexpected operations
  KernelLaunchMetadata op1("matmul_kernel", 0x3333333333333333ULL, 8, 8, 1, 16,
                           16, 1, 49152, 2, 0);
  KernelLaunchMetadata op2("cuMemset", 0x4444444444444444ULL, 8192,
                           0xdeadbeefULL, 4);

  diff.unexpectedOps[op1] = 1;
  diff.unexpectedOps[op2] = 3;

  EXPECT_FALSE(diff.empty());

  std::string json = jsonToString(diff.formatJSON(2));

  EXPECT_NE(json.find("\"graphIdx\":2"), std::string::npos);
  EXPECT_NE(json.find("\"hasMismatch\":true"), std::string::npos);
  EXPECT_NE(json.find("\"unexpectedOps\""), std::string::npos);
  EXPECT_NE(json.find("\"matmul_kernel\""), std::string::npos);
  EXPECT_NE(json.find("\"cuMemset\""), std::string::npos);
  EXPECT_NE(json.find("\"count\":1"), std::string::npos);
  EXPECT_NE(json.find("\"count\":3"), std::string::npos);
}

// Test: LaunchTraceDiff with both missing and unexpected operations
TEST_F(TraceMetadataTest, LaunchTraceDiffJSONBoth) {
  LaunchTraceDiff diff;

  // Add missing operations
  KernelLaunchMetadata missing("vec_add", 0x1111111111111111ULL, 4, 1, 1, 256,
                               1, 1, 0, 0, 0);
  diff.missingOps[missing] = 1;

  // Add unexpected operations
  KernelLaunchMetadata unexpected("vec_mul", 0x2222222222222222ULL, 8, 1, 1,
                                  128, 1, 1, 0, 0, 0);
  diff.unexpectedOps[unexpected] = 2;

  EXPECT_FALSE(diff.empty());

  std::string json = jsonToString(diff.formatJSON(3));

  EXPECT_NE(json.find("\"graphIdx\":3"), std::string::npos);
  EXPECT_NE(json.find("\"hasMismatch\":true"), std::string::npos);
  EXPECT_NE(json.find("\"missingOps\""), std::string::npos);
  EXPECT_NE(json.find("\"unexpectedOps\""), std::string::npos);
  EXPECT_NE(json.find("\"vec_add\""), std::string::npos);
  EXPECT_NE(json.find("\"vec_mul\""), std::string::npos);
}

// Test: LaunchTraceDiff with multiple operation types
TEST_F(TraceMetadataTest, LaunchTraceDiffJSONMultipleTypes) {
  LaunchTraceDiff diff;

  // Add different operation types
  KernelLaunchMetadata kernel("kernel", 0x1111ULL, 1, 1, 1, 1, 1, 1, 0, 0, 0);
  KernelLaunchMetadata memcpy("cuMemcpy", 0x2222ULL,
                              KernelLaunchMetadata::MemcpyKind::DtoH, 1024);
  KernelLaunchMetadata memset("cuMemset", 0x3333ULL, 512, 0xFFULL, 1);

  diff.missingOps[kernel] = 1;
  diff.unexpectedOps[memcpy] = 1;
  diff.unexpectedOps[memset] = 1;

  std::string json = jsonToString(diff.formatJSON(0));

  // Check all operation kinds are represented
  EXPECT_NE(json.find("\"kind\":\"KernelLaunch\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memcpy\""), std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"Memset\""), std::string::npos);
}

// ===== Global launch-trace registry tests =====
//
// These exercise the process-global trace directly, using fake stream handles
// so no GPU is required. They pin the two properties the per-stream API could
// not provide: cross-stream enqueue ordering in one list, and a stable
// per-stream index.

class LaunchTraceRegistryTest : public ::testing::Test {};

// A global trace records operations from multiple streams into one list in
// enqueue order, tagging each with a first-seen stream index.
TEST_F(LaunchTraceRegistryTest, GlobalTraceOrdersAcrossStreams) {
  auto *streamA = reinterpret_cast<void *>(0x1);
  auto *streamB = reinterpret_cast<void *>(0x2);
  auto &registry = launchTraceRegistry();

  registry.beginGlobalTrace();
  registry.appendLaunchTraceEntry(streamA, KernelLaunchMetadata("a0", 1));
  registry.appendLaunchTraceEntry(streamB, KernelLaunchMetadata("b0", 2));
  registry.appendLaunchTraceEntry(streamA, KernelLaunchMetadata("a1", 3));
  std::vector<KernelLaunchMetadata> trace = registry.takeGlobalTrace();

  ASSERT_EQ(trace.size(), 3u);
  // Enqueue order is preserved across streams.
  EXPECT_EQ(trace[0].kernelName, "a0");
  EXPECT_EQ(trace[1].kernelName, "b0");
  EXPECT_EQ(trace[2].kernelName, "a1");
  // Stream index is assigned in first-seen order and is stable per stream.
  EXPECT_EQ(trace[0].streamIndex, 0u);
  EXPECT_EQ(trace[1].streamIndex, 1u);
  EXPECT_EQ(trace[2].streamIndex, 0u);
}

// Taking the trace stops recording; a stream is enabled only while the global
// trace is active.
TEST_F(LaunchTraceRegistryTest, GlobalTraceStopsAndClears) {
  auto *stream = reinterpret_cast<void *>(0x10);
  auto &registry = launchTraceRegistry();

  EXPECT_FALSE(registry.isLaunchTraceEnabled(stream));
  registry.beginGlobalTrace();
  EXPECT_TRUE(registry.isLaunchTraceEnabled(stream));

  registry.appendLaunchTraceEntry(stream, KernelLaunchMetadata("k", 1));
  EXPECT_EQ(registry.takeGlobalTrace().size(), 1u);

  // Recording has stopped: further work is dropped and nothing is retained.
  EXPECT_FALSE(registry.isLaunchTraceEnabled(stream));
  registry.appendLaunchTraceEntry(stream, KernelLaunchMetadata("k", 1));
  EXPECT_TRUE(registry.takeGlobalTrace().empty());
}

// Re-beginning without an intervening take discards the previous trace and
// restarts first-seen stream indexing from zero.
TEST_F(LaunchTraceRegistryTest, BeginGlobalTraceClearsPrevious) {
  auto *streamA = reinterpret_cast<void *>(0x20);
  auto *streamB = reinterpret_cast<void *>(0x21);
  auto &registry = launchTraceRegistry();

  registry.beginGlobalTrace();
  registry.appendLaunchTraceEntry(streamA, KernelLaunchMetadata("stale", 1));
  registry.beginGlobalTrace();
  registry.appendLaunchTraceEntry(streamB, KernelLaunchMetadata("fresh", 2));
  std::vector<KernelLaunchMetadata> trace = registry.takeGlobalTrace();

  // Only the second trace's entry survives, and streamB is index 0 despite
  // streamA having been seen first in the discarded trace.
  ASSERT_EQ(trace.size(), 1u);
  EXPECT_EQ(trace[0].kernelName, "fresh");
  EXPECT_EQ(trace[0].streamIndex, 0u);
}

} // namespace
