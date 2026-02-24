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

TEST(MojoPackageTest, testReadErrors) {
  auto getTestBuffer =
      [](llvm::StringRef str) -> std::unique_ptr<llvm::MemoryBuffer> {
    auto buffer =
        llvm::MemoryBuffer::getMemBuffer(str, /*BufferName=*/"test.mojopkg");
    EXPECT_TRUE(buffer);
    return buffer;
  };

  { // Invalid header, no magic bytes
    auto buffer = getTestBuffer("M00G");

    EXPECT_FALSE(M::KGEN::isMojoPackage(*buffer));
    auto Err = M::KGEN::readBinaryPackageHeader(*buffer);
    EXPECT_TRUE(Err.isError());
    EXPECT_STREQ(Err.getError(), "invalid magic bytes");

    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_TRUE(BuffAndHeaderOrErr.isError());
    EXPECT_STREQ(BuffAndHeaderOrErr.getError(),
                 "invalid Mojo package 'test.mojopkg': invalid magic bytes");
  }

  { // Invalid header, too small
    auto buffer = getTestBuffer("MPKG0");

    EXPECT_TRUE(M::KGEN::isMojoPackage(*buffer));
    auto Err = M::KGEN::readBinaryPackageHeader(*buffer);
    EXPECT_TRUE(Err.isError());
    EXPECT_STREQ(Err.getError(), "invalid header size");

    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_TRUE(BuffAndHeaderOrErr.isError());
    EXPECT_STREQ(BuffAndHeaderOrErr.getError(),
                 "invalid Mojo package 'test.mojopkg': invalid header size");
  }

  { // Invalid header, too small to contain versioning information
    auto buffer = getTestBuffer("MPKG1000");
    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_TRUE(BuffAndHeaderOrErr.isError());
    EXPECT_STREQ(
        BuffAndHeaderOrErr.getError(),
        "invalid Mojo package 'test.mojopkg': read past end of buffer");
  }

  { // Invalid header, too small to contain versioning information
    auto buffer = getTestBuffer(StringRef("MPKG1000"
                                          "\x28\x10\x01\x00"
                                          "-dev\x00",
                                          16));
    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_TRUE(BuffAndHeaderOrErr.isError());
    EXPECT_STREQ(
        BuffAndHeaderOrErr.getError(),
        "invalid Mojo package 'test.mojopkg': invalid version encoding");
  }

  { // Invalid header, contains both versions but no checksum
    auto buffer = getTestBuffer(StringRef("MPKG1000"
                                          "\x28\x10\x01\x00"
                                          "-dev\x00"
                                          "\x29\x11\x00\x01"
                                          "-label\x00",
                                          28));
    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_TRUE(BuffAndHeaderOrErr.isError());
    EXPECT_STREQ(
        BuffAndHeaderOrErr.getError(),
        "invalid Mojo package 'test.mojopkg': invalid checksum encoding");
  }

  { // Invalid header, contains both versions and a checksum but is not aligned
    // to 8 bytes
    auto buffer = getTestBuffer(StringRef("MPKG1000"
                                          "\x28\x10\x01\x00"
                                          "-dev\x00"
                                          "\x29\x11\x00\x01"
                                          "-label\x00"
                                          "c\x00",
                                          30));
    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_TRUE(BuffAndHeaderOrErr.isError());
    EXPECT_STREQ(BuffAndHeaderOrErr.getError(),
                 "invalid Mojo package 'test.mojopkg': invalid header size");
  }

  { // Valid header, though no MLIR buffer
    auto buffer = getTestBuffer(StringRef("MPKG1000"
                                          "\x28\x10\x01\x00"
                                          "-dev\x00"
                                          "\x29\x11\x00\x01"
                                          "-label\x00"
                                          "c\x00"
                                          "x\x00",
                                          32));
    auto BuffAndHeaderOrErr =
        M::KGEN::getMLIRBufferAndHeaderFromPackage(*buffer);
    EXPECT_FALSE(BuffAndHeaderOrErr.isError());

    auto [header, mlirBuffer] = *BuffAndHeaderOrErr;

    EXPECT_EQ(header.mojoVersion.major, 40);
    EXPECT_EQ(header.mojoVersion.minor, 16);
    EXPECT_EQ(header.mojoVersion.patch, 1);
    EXPECT_STREQ(header.mojoVersion.label.c_str(), "-dev");

    EXPECT_EQ(header.modularVersion.major, 41);
    EXPECT_EQ(header.modularVersion.minor, 17);
    EXPECT_EQ(header.modularVersion.patch, 256);
    EXPECT_STREQ(header.modularVersion.label.c_str(), "-label");

    EXPECT_STREQ(header.mlirChecksum.c_str(), "c");

    // There's no MLIR buffer after the package header
    EXPECT_EQ(0, mlirBuffer.getBufferSize());
  }
}
