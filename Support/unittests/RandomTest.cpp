//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Random.h"
#include "llvm/ADT/ArrayRef.h"

#include "Support/ErrorOr.h"
#include "gtest/gtest.h"

using namespace M;

TEST(Random, Works) {
  SecureRandomBytesGenerator rng;

  SmallVector<uint8_t> buf;
  buf.resize(32);
  MutableArrayRef<uint8_t> randView(buf.begin(), buf.size());
  auto err = rng.getRandomBytes(randView);
  EXPECT_FALSE(err.isError()) << err.getError();
}
