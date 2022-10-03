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
//   Path:   llvm/lib/Support/TimeProfiler.cpp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hierarchical time profiler.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "Support/TimeProfiler.h"

using namespace M;

namespace {

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::system_clock;
using std::chrono::time_point;
using std::chrono::time_point_cast;

using ClockType = TimeTraceProfilerEntry<true>::ClockType;
using TimePointType = TimeTraceProfilerEntry<true>::TimePointType;
using DurationType = duration<ClockType::rep, ClockType::period>;
using CountAndDurationType = std::pair<size_t, DurationType>;
using NameAndCountAndDurationType =
    std::pair<std::string, CountAndDurationType>;

struct TimeTraceProfiler;

struct TimeTraceProfilerInstances {
  std::mutex Lock;
  std::vector<TimeTraceProfiler *> List;
};

TimeTraceProfilerInstances &getTimeTraceProfilerInstances() {
  static TimeTraceProfilerInstances Instances;
  return Instances;
}

struct TimeTraceProfiler {
  explicit TimeTraceProfiler(unsigned TimeTraceGranularity = 0,
                             StringRef ProcName = "")
      : BeginningOfTime(system_clock::now()), StartTime(ClockType::now()),
        ProcName(ProcName), Pid(llvm::sys::Process::getProcessId()),
        Tid(llvm::get_threadid()), TimeTraceGranularity(TimeTraceGranularity) {
    llvm::get_thread_name(ThreadName);
  }

  void begin(std::string Name, llvm::function_ref<std::string()> Detail) {
    Stack.emplace_back(ClockType::now(), TimePointType(), std::move(Name),
                       Detail());
  }

  void end() {
    assert(!Stack.empty() && "Must call begin() first");
    end(std::move(Stack.back()));
    Stack.pop_back();
  }

  void end(TimeTraceProfilerEntry<true> &&E) {
    E.End = ClockType::now();

    // Check that end times monotonically increase.
    assert((Entries.empty() ||
            (E.getFlameGraphStartUs(StartTime) + E.getFlameGraphDurUs() >=
             Entries.back().getFlameGraphStartUs(StartTime) +
                 Entries.back().getFlameGraphDurUs())) &&
           "TimeProfiler scope ended earlier than previous scope");

    // Calculate duration at full precision for overall counts.
    DurationType Duration = E.End - E.Start;

    // Only include sections longer or equal to TimeTraceGranularity msec.
    if (duration_cast<microseconds>(Duration).count() >= TimeTraceGranularity)
      Entries.emplace_back(E);

    // Track total time taken by each "name", but only the topmost levels of
    // them; e.g. if there's a template instantiation that instantiates other
    // templates from within, we only want to add the topmost one. "topmost"
    // happens to be the ones that don't have any currently open entries above
    // itself.
    if (llvm::none_of(llvm::drop_begin(llvm::reverse(Stack)),
                      [&](const TimeTraceProfilerEntry<true> &Val) {
                        return Val.Name == E.Name;
                      })) {
      auto &CountAndTotal = CountAndTotalPerName[E.Name];
      CountAndTotal.first++;
      CountAndTotal.second += Duration;
    }
  }

  // Write events from this TimeTraceProfilerInstance and
  // ThreadTimeTraceProfilerInstances.
  void write(llvm::raw_pwrite_stream &OS) {
    // Acquire Mutex as reading ThreadTimeTraceProfilerInstances.
    auto &Instances = getTimeTraceProfilerInstances();
    std::lock_guard<std::mutex> Lock(Instances.Lock);
    assert(Stack.empty() &&
           "All profiler sections should be ended when calling write");
    assert(llvm::all_of(Instances.List,
                        [](const auto &TTP) { return TTP->Stack.empty(); }) &&
           "All profiler sections should be ended when calling write");

    llvm::json::OStream J(OS);
    J.objectBegin();
    J.attributeBegin("traceEvents");
    J.arrayBegin();

    // Emit all events for the main flame graph.
    auto writeEvent = [&](const auto &E, uint64_t Tid) {
      auto StartUs = E.getFlameGraphStartUs(StartTime);
      auto DurUs = E.getFlameGraphDurUs();

      J.object([&] {
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(Tid));
        J.attribute("ph", "X");
        J.attribute("ts", StartUs);
        J.attribute("dur", DurUs);
        J.attribute("name", E.Name);
        if (!E.Detail.empty()) {
          J.attributeObject("args", [&] { J.attribute("detail", E.Detail); });
        }
      });
    };
    for (const TimeTraceProfilerEntry<true> &E : Entries)
      writeEvent(E, this->Tid);
    for (const TimeTraceProfiler *TTP : Instances.List)
      for (const TimeTraceProfilerEntry<true> &E : TTP->Entries)
        writeEvent(E, TTP->Tid);

    // Emit totals by section name as additional "thread" events, sorted from
    // longest one.
    // Find highest used thread id.
    uint64_t MaxTid = this->Tid;
    for (const TimeTraceProfiler *TTP : Instances.List)
      MaxTid = std::max(MaxTid, TTP->Tid);

    // Combine all CountAndTotalPerName from threads into one.
    llvm::StringMap<CountAndDurationType> AllCountAndTotalPerName;
    auto combineStat = [&](const auto &Stat) {
      StringRef Key = Stat.getKey();
      auto Value = Stat.getValue();
      auto &CountAndTotal = AllCountAndTotalPerName[Key];
      CountAndTotal.first += Value.first;
      CountAndTotal.second += Value.second;
    };
    for (const auto &Stat : CountAndTotalPerName)
      combineStat(Stat);
    for (const TimeTraceProfiler *TTP : Instances.List)
      for (const auto &Stat : TTP->CountAndTotalPerName)
        combineStat(Stat);

    std::vector<NameAndCountAndDurationType> SortedTotals;
    SortedTotals.reserve(AllCountAndTotalPerName.size());
    for (const auto &Total : AllCountAndTotalPerName)
      SortedTotals.emplace_back(std::string(Total.getKey()), Total.getValue());

    llvm::sort(SortedTotals, [](const NameAndCountAndDurationType &A,
                                const NameAndCountAndDurationType &B) {
      return A.second.second > B.second.second;
    });

    // Report totals on separate threads of tracing file.
    uint64_t TotalTid = MaxTid + 1;
    for (const NameAndCountAndDurationType &Total : SortedTotals) {
      auto DurUs = duration_cast<microseconds>(Total.second.second).count();
      auto Count = AllCountAndTotalPerName[Total.first].first;

      J.object([&] {
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(TotalTid));
        J.attribute("ph", "X");
        J.attribute("ts", 0);
        J.attribute("dur", DurUs);
        J.attribute("name", "Total " + Total.first);
        J.attributeObject("args", [&] {
          J.attribute("count", int64_t(Count));
          J.attribute("avg ms", int64_t(DurUs / Count / 1000));
        });
      });

      ++TotalTid;
    }

    auto writeMetadataEvent = [&](const char *Name, uint64_t Tid,
                                  StringRef arg) {
      J.object([&] {
        J.attribute("cat", "");
        J.attribute("pid", Pid);
        J.attribute("tid", int64_t(Tid));
        J.attribute("ts", 0);
        J.attribute("ph", "M");
        J.attribute("name", Name);
        J.attributeObject("args", [&] { J.attribute("name", arg); });
      });
    };

    writeMetadataEvent("process_name", Tid, ProcName);
    writeMetadataEvent("thread_name", Tid, ThreadName);
    for (const TimeTraceProfiler *TTP : Instances.List)
      writeMetadataEvent("thread_name", TTP->Tid, TTP->ThreadName);

    J.arrayEnd();
    J.attributeEnd();

    // Emit the absolute time when this TimeProfiler started.
    // This can be used to combine the profiling data from
    // multiple processes and preserve actual time intervals.
    J.attribute("beginningOfTime",
                time_point_cast<microseconds>(BeginningOfTime)
                    .time_since_epoch()
                    .count());

    J.objectEnd();
  }

  SmallVector<TimeTraceProfilerEntry<true>, 16> Stack;
  SmallVector<TimeTraceProfilerEntry<true>, 128> Entries;
  llvm::StringMap<CountAndDurationType> CountAndTotalPerName;
  // System clock time when the session was begun.
  const time_point<system_clock> BeginningOfTime;
  // Profiling clock time when the session was begun.
  const TimePointType StartTime;
  const std::string ProcName;
  const llvm::sys::Process::Pid Pid;
  SmallString<0> ThreadName;
  const uint64_t Tid;

  // Minimum time granularity (in microseconds)
  const unsigned TimeTraceGranularity;
};

} // anonymous namespace

// Per Thread instance
static LLVM_THREAD_LOCAL TimeTraceProfiler *TimeTraceProfilerInstance = nullptr;

void M::timeTraceProfilerInitialize(unsigned TimeTraceGranularity,
                                    StringRef ProcName) {
  assert(TimeTraceProfilerInstance == nullptr &&
         "Profiler should not be initialized");
  TimeTraceProfilerInstance = new TimeTraceProfiler(
      TimeTraceGranularity, llvm::sys::path::filename(ProcName));
}

// Removes all TimeTraceProfilerInstances.
// Called from main thread.
void M::timeTraceProfilerCleanup() {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler should be initialized");
  delete TimeTraceProfilerInstance;
  TimeTraceProfilerInstance = nullptr;

  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  for (auto *TTP : Instances.List)
    delete TTP;
  Instances.List.clear();
}

// Finish TimeTraceProfilerInstance on a worker thread.
// This doesn't remove the instance, just moves the pointer to global vector.
void M::timeTraceProfilerFinishThread() {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler should be initialized");
  auto &Instances = getTimeTraceProfilerInstances();
  std::lock_guard<std::mutex> Lock(Instances.Lock);
  Instances.List.push_back(TimeTraceProfilerInstance);
  TimeTraceProfilerInstance = nullptr;
}

void M::timeTraceProfilerWrite(llvm::raw_pwrite_stream &OS) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler should be initialized");
  TimeTraceProfilerInstance->write(OS);
}

llvm::Error M::timeTraceProfilerWrite(StringRef PreferredFileName,
                                      StringRef FallbackFileName) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler should be initialized");

  std::string Path = PreferredFileName.str();
  if (Path.empty()) {
    Path = FallbackFileName == "-" ? "out" : FallbackFileName.str();
    Path += ".time-trace";
  }

  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_TextWithCRLF);
  if (EC)
    return llvm::createStringError(EC, "Could not open " + Path);

  timeTraceProfilerWrite(OS);
  return llvm::Error::success();
}

void M::Detail::timeTraceProfilerBeginImpl(
    std::string &&Name, llvm::function_ref<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->begin(std::move(Name), Detail);
}

void M::Detail::timeTraceProfilerEndImpl() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end();
}

M::TimeTraceProfilerEntry<true> M::Detail::timeTraceProfilerBeginEntryImpl(
    std::string &&Name, llvm::function_ref<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    return {TimeTraceProfilerEntry<true>::ClockType::now(),
            TimeTraceProfilerEntry<true>::TimePointType(), std::string(Name),
            Detail()};
  else
    return {};
}

void M::Detail::timeTraceProfilerEndEntryImpl(
    TimeTraceProfilerEntry<true> &&Entry) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end(std::move(Entry));
}

void M::Detail::timeTraceProfilerStartEntryImpl(
    TimeTraceProfilerEntry<true> &Entry) {
  if (TimeTraceProfilerInstance != nullptr)
    Entry.Start = TimeTraceProfilerEntry<true>::ClockType::now();
}
