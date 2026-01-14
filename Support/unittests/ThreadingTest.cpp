//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Threading/ThreadAffinity.h"

#include "Support/ErrorOr.h"
#include "gtest/gtest.h"

using namespace M;

TEST(Threading, haveThreadAffinity) {
#if defined(__linux__)
  // On linux, we expect haveThreadAffinity() to return True.
  // This may not be the case on every linux system in the world (presumably,
  // it's possible to build the kernel without affinity support), but all linux
  // systems of interest at this time should support this. If this test fails,
  // it could be a recurrence of the bug documented in GEX-3070.
  EXPECT_TRUE(haveThreadAffinity()) << "linux system without thread affinity?";
#endif

#if defined(__APPLE__)
  // On macos, we don't have/use thread affinity.
  EXPECT_FALSE(haveThreadAffinity());
#endif
}
