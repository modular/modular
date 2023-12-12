//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// These are bare-minimum 'smoke' tests of the time profiler. Not tested:
//  - multi-threading
//  - 'Total' entries
//  - elision of short or ill-formed entries
//  - detail callback
//  - no calls to now() if profiling is disabled
//  - suppression of contributions to total entries for nested entries
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

#include "Support/Profiling/TimeProfiler.h"

#include "gtest/gtest.h"

using namespace M;

static std::string teardownTrace(TimeTraceProfiler &profiler) {
  SmallVector<char, 1024> smallVector;
  llvm::raw_svector_ostream os(smallVector);
  profiler.writeJSONForTesting(os);
  return os.str().str();
}

namespace {

TEST(TimeProfiler, Scope_Smoke) {
  TimeTraceProfiler profiler(/*timeTraceGranularity=*/0, "test");

  {
    TimeTraceScope</*Enabled=*/true> scope(
        ProfilerEntry<true>::create("event", StringLiteral("detail")));
  }

  std::string json = teardownTrace(profiler);
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Begin_End_Smoke) {
  TimeTraceProfiler profiler(/*timeTraceGranularity=*/0, "test");

  ProfilerEntry<true>::createAndPush("event", StringLiteral("detail"));
  ProfilerEntry<true>::endAndPop();

  std::string json = teardownTrace(profiler);
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Begin_End_Disabled) {
  // Nothing should be observable here. The test is really just making sure
  // we've not got a stray nullptr deref.
  ProfilerEntry<true>::createAndPush("event", StringLiteral("detail"));
  ProfilerEntry<true>::endAndPop();
}

TEST(TimeProfiler, Entry_Smoke) {
  TimeTraceProfiler profiler(/*timeTraceGranularity=*/0, "test");

  auto entry = ProfilerEntry<true>::create("event", StringLiteral("detail"));
  std::move(entry).record();

  std::string json = teardownTrace(profiler);
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Entry_Disabled) {
  // Only get the default entry if tracing is not setup.
  auto entry = ProfilerEntry<true>::create("event", StringLiteral("detail"));
  ASSERT_TRUE(entry.empty());
  std::move(entry).record();
}

} // namespace
