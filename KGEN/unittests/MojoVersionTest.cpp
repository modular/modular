//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/DialectChecksum/DialectChecksum.h"
#include "gtest/gtest.h"

TEST(MojoVersionTest, testNotEmpty) {
  llvm::StringRef checksum = M::getMojoMlirDialectChecksum();
  ASSERT_FALSE(checksum.empty());
}
