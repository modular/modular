//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// File originates from:
//   Repo:   https://github.com/llvm/llvm-project.git
//   Commit: 271f3b91bbf80e9cf22d9e6bee738abb496fecf9
//   Path:   llvm/unittests/Support/TimeProfilerTest.cpp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

#include "Support/TimeProfiler.h"

#include "gtest/gtest.h"

using namespace M;

static std::string teardownTrace() {
  SmallVector<char, 1024> smallVector;
  llvm::raw_svector_ostream os(smallVector);
  Detail::timeTraceProfilerWriteTrace(os);
  return os.str().str();
}

namespace {

TEST(TimeProfiler, Scope_Smoke) {
  TimeTraceProfiler profiler(/*timeTraceGranularity=*/0, "test");

  { TimeTraceScope</*Enabled=*/true> scope("event", "detail"); }

  std::string json = teardownTrace();
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Begin_End_Smoke) {
  TimeTraceProfiler profiler(/*timeTraceGranularity=*/0, "test");

  timeTraceProfilerBegin("event", "detail");
  timeTraceProfilerEnd();

  std::string json = teardownTrace();
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Begin_End_Disabled) {
  // Nothing should be observable here. The test is really just making sure
  // we've not got a stray nullptr deref.
  timeTraceProfilerBegin("event", "detail");
  timeTraceProfilerEnd();
}

TEST(TimeProfiler, Entry_Smoke) {
  TimeTraceProfiler profiler(/*timeTraceGranularity=*/0, "test");

  auto entry = timeTraceProfilerBeginEntry("event", "detail");
  timeTraceProfilerStartEntry(entry);
  timeTraceProfilerEndEntry(std::move(entry));

  std::string json = teardownTrace();
  ASSERT_TRUE(json.find(R"("name":"event")") != std::string::npos);
  ASSERT_TRUE(json.find(R"("detail":"detail")") != std::string::npos);
}

TEST(TimeProfiler, Entry_Disabled) {
  // Only get the default entry if tracing is not setup.
  auto entry = timeTraceProfilerBeginEntry("event", "detail");
  timeTraceProfilerStartEntry(entry);
  ASSERT_TRUE(entry.name.empty());
  ASSERT_TRUE(entry.detail.empty());
  ASSERT_EQ(entry.start, TimeTraceProfilerEntry<true>::TimePointType());
  ASSERT_EQ(entry.end, TimeTraceProfilerEntry<true>::TimePointType());
  timeTraceProfilerEndEntry(std::move(entry));
}

} // namespace
