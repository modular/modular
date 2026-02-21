//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/MojoPackage.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(MojoPackageTest, testRoundtrip) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));

  std::string s;
  llvm::raw_string_ostream out(s);

  M::KGEN::MojoPackageVersion expectedMojoVer{1, 2, 3};
  expectedMojoVer.label = "-dev";
  M::KGEN::MojoPackageVersion expectedModularVer{10, 0, 0};
  const char *expectedMlirChecksum = "deadbeef";

  constexpr unsigned expectedHeaderSize = 32;

  auto writeRes = M::KGEN::writeBinaryPackage(
      *module, expectedMojoVer, expectedModularVer, expectedMlirChecksum, out);
  ASSERT_FALSE(writeRes.failed());

  llvm::StringRef str = out.str();
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  ASSERT_TRUE(buffer);

  // This should be a Mojo Package
  EXPECT_TRUE(M::KGEN::isMojoPackage(*buffer));

  auto mlirBufferAndHeaderOrErr =
      M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
  ASSERT_FALSE(mlirBufferAndHeaderOrErr.isError());

  auto [header, mlirBuffer] = *mlirBufferAndHeaderOrErr;

  // We only have one version of the Mojo package format
  EXPECT_EQ(header.version, 1);

  auto mojoVer = header.mojoVersion;
  auto modularVer = header.modularVersion;
  auto mlirChecksum = header.mlirChecksum;

  EXPECT_EQ(mojoVer, expectedMojoVer);
  EXPECT_EQ(mojoVer.label, expectedMojoVer.label);
  EXPECT_EQ(modularVer, expectedModularVer);
  EXPECT_EQ(modularVer.label, expectedModularVer.label);
  EXPECT_EQ(mlirChecksum, expectedMlirChecksum);

  EXPECT_EQ(header.getSizeInBytes(), expectedHeaderSize);

  // Check that the MLIR buffer has the right magic bytes at the beginning.
  EXPECT_EQ(mlirBuffer.getBuffer().substr(0, 4), "ML\xEFR");
}

// Regression test: when a version label causes the pre-NUL byte count to land
// on an 8-byte boundary, the header size was underestimated by 8 bytes because
// the checksum NUL terminator wasn't consumed by the reader.
TEST(MojoPackageTest, testRoundtripWithLabelAlignment) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));

  std::string s;
  llvm::raw_string_ostream out(s);

  M::KGEN::MojoPackageVersion expectedMojoVer{0, 0, 0};
  M::KGEN::MojoPackageVersion expectedModularVer{26, 2, 0};
  expectedModularVer.label = ".dev2026022105";
  // 64-char checksum to match real package files.
  const char *expectedMlirChecksum =
      "0e1898dc55c6be46748497d48424c757a293ec517b47eb634acff6e0fd8ef079";

  // 4(magic) + 1(ver) + 3(reserved) + 5(mojoVer) + 19(modularVer) +
  // 64(checksum) + 1(NUL) = 97 -> alignTo(97, 8) = 104
  constexpr unsigned expectedHeaderSize = 104;

  auto writeRes = M::KGEN::writeBinaryPackage(
      *module, expectedMojoVer, expectedModularVer, expectedMlirChecksum, out);
  ASSERT_FALSE(writeRes.failed());

  llvm::StringRef str = out.str();
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  ASSERT_TRUE(buffer);

  EXPECT_TRUE(M::KGEN::isMojoPackage(*buffer));

  auto mlirBufferAndHeaderOrErr =
      M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
  ASSERT_FALSE(mlirBufferAndHeaderOrErr.isError());

  auto [header, mlirBuffer] = *mlirBufferAndHeaderOrErr;

  EXPECT_EQ(header.version, 1);
  EXPECT_EQ(header.mojoVersion, expectedMojoVer);
  EXPECT_EQ(header.modularVersion, expectedModularVer);
  EXPECT_EQ(header.modularVersion.label, expectedModularVer.label);
  EXPECT_EQ(header.mlirChecksum, expectedMlirChecksum);

  EXPECT_EQ(header.getSizeInBytes(), expectedHeaderSize);

  // Check that the MLIR buffer has the right magic bytes at the beginning.
  EXPECT_EQ(mlirBuffer.getBuffer().substr(0, 4), "ML\xEFR");
}
