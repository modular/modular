//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/TimeProfiler.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h"
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

using namespace M;

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

namespace {
struct TimeTraceThreadProfiler {
  explicit TimeTraceThreadProfiler(unsigned timeTraceGranularity)
      : tid(llvm::get_threadid()), timeTraceGranularity(timeTraceGranularity) {
    llvm::get_thread_name(threadName);
  }

  /// Start a new entry with the given name and detail.
  void begin(StringRef name, function_ref<std::string()> detailFn) {
    stack.emplace_back(name, detailFn);
  }

  void begin(StringRef name, StringRef detail) {
    stack.emplace_back(name, detail);
  }

  /// End the current running entry.
  void end() {
    assert(!stack.empty() && "must call begin() first");
    record(std::move(stack.back()));
    stack.pop_back();
  }

  /// Record the given entry.
  void record(TimeTraceProfilerEntry<true> &&entry) {
    if (entry.name.empty())
      return;

    entry.end = ClockType::now();

    // Calculate duration at full precision for overall counts.
    DurationType duration = entry.end - entry.start;

    // Only include sections longer or equal to timeTraceGranularity msec.
    if (duration_cast<microseconds>(duration).count() >= timeTraceGranularity)
      entries.emplace_back(entry);

    // Track total time taken by each "name", but only the topmost levels of
    // them; e.g. if there's a template instantiation that instantiates other
    // templates from within, we only want to add the topmost one. "topmost"
    // happens to be the ones that don't have any currently open entries above
    // itself.
    if (stack.empty() ||
        llvm::none_of(llvm::drop_begin(llvm::reverse(stack)),
                      [&](const TimeTraceProfilerEntry<true> &val) {
                        return val.name == entry.name;
                      })) {
      auto &countAndTotal = countAndTotalPerName[entry.name];
      countAndTotal.first++;
      countAndTotal.second += duration;
    }
  }

  /// The stack of currently running timers.
  SmallVector<TimeTraceProfilerEntry<true>, 16> stack;

  /// The set of completed timer entries.
  SmallVector<TimeTraceProfilerEntry<true>, 128> entries;

  /// The total time taken by each "name".
  llvm::StringMap<CountAndDurationType> countAndTotalPerName;

  /// The name of the thread this profiler is running on.
  SmallString<0> threadName;

  /// The id of the thread this profiler is running on.
  const uint64_t tid;

  // Minimum time granularity (in microseconds)
  const unsigned timeTraceGranularity;
};

/// This class represents the profiler context for a specific thread.
struct ThreadProfilerContext {
  ~ThreadProfilerContext();

  /// Return the profiler instance for this thread, or nullptr if one isn't
  /// active.
  static TimeTraceThreadProfiler *get();

  /// The profiler attached to this thread.
  TimeTraceThreadProfiler *profiler = nullptr;
};

/// This class represents the main context used for profiling.
struct GlobalProfilerContext {
  GlobalProfilerContext(unsigned granularity, StringRef name)
      : timeTraceGranularity(granularity), procName(name),
        pid(llvm::sys::Process::getProcessId()),
        beginningOfTime(system_clock::now()), startTime(ClockType::now()) {}

  /// The main profiler context instance.
  static GlobalProfilerContext *instance;

  /// The minimum time granularity (in microseconds) for time trace profiler.
  unsigned timeTraceGranularity = 0;

  /// The name of the process this profiler is running on.
  StringRef procName;

  /// The id of the process this profiler is running on.
  const llvm::sys::Process::Pid pid;

  /// System clock time when the session was begun.
  time_point<system_clock> beginningOfTime;

  /// Profiling clock time when the session was begun.
  const TimePointType startTime;

  /// Lock used to guard access to the running profilers.
  std::mutex lock;

  /// The set of running profilers for each thread.
  std::vector<std::unique_ptr<TimeTraceThreadProfiler>> profilers;

  /// A set of active thread profiler contexts.
  DenseSet<ThreadProfilerContext *> threadProfilerContexts;
};
} // anonymous namespace

GlobalProfilerContext *GlobalProfilerContext::instance = nullptr;

TimeTraceThreadProfiler *ThreadProfilerContext::get() {
  static thread_local ThreadProfilerContext instance;
  if (!instance.profiler && GlobalProfilerContext::instance) {
    auto &ctx = *GlobalProfilerContext::instance;
    std::lock_guard<std::mutex> lock(ctx.lock);

    // Add this profiler to the main context.
    ctx.profilers.emplace_back(
        std::make_unique<TimeTraceThreadProfiler>(ctx.timeTraceGranularity));
    ctx.threadProfilerContexts.insert(&instance);
    instance.profiler = ctx.profilers.back().get();
  }
  return instance.profiler;
}

ThreadProfilerContext::~ThreadProfilerContext() {
  // The current thread is dying, so try to pass over ownership of the
  // profiler to the main context.
  if (auto *ctx = GlobalProfilerContext::instance) {
    std::lock_guard<std::mutex> lock(ctx->lock);
    ctx->threadProfilerContexts.erase(this);
  }
}

//===----------------------------------------------------------------------===//
// TimeTraceProfiler
//===----------------------------------------------------------------------===//

void M::Detail::timeTraceProfilerInitialize(unsigned timeTraceGranularity,
                                            StringRef procName) {
  assert(!GlobalProfilerContext::instance &&
         "profiler should not be initialized");
  GlobalProfilerContext::instance = new GlobalProfilerContext(
      timeTraceGranularity, llvm::sys::path::filename(procName));

  // Prep the profiler for the main thread.
  ThreadProfilerContext::get();
}

void M::Detail::timeTraceProfilerDestroy() {
  assert(GlobalProfilerContext::instance && "profiler should be initialized");

  { // Clear out any dangling pointers in thread profiler contexts.
    std::lock_guard<std::mutex> guard(GlobalProfilerContext::instance->lock);
    for (auto *tpc : GlobalProfilerContext::instance->threadProfilerContexts)
      tpc->profiler = nullptr;
  }

  delete GlobalProfilerContext::instance;
  GlobalProfilerContext::instance = nullptr;
}

//===----------------------------------------------------------------------===//
// Trace Output
//===----------------------------------------------------------------------===//

void M::Detail::timeTraceProfilerWriteTrace(llvm::raw_pwrite_stream &os) {
  assert(GlobalProfilerContext::instance && "profiler should be initialized");
  auto &ctx = *GlobalProfilerContext::instance;
  std::lock_guard<std::mutex> lock(ctx.lock);
  auto profilers = llvm::make_pointee_range(ctx.profilers);
  assert(llvm::all_of(profilers,
                      [](const auto &ttp) { return ttp.stack.empty(); }) &&
         "all profiler sections should be ended when calling write");

  // For visualization purposes only.
  // Sometimes callers can have the same start time (in ns) as their callees.
  // Since events are push to Entries when the trace event ends, callees
  // appear before callers in Entries. When Perfetto sees 2 events with the
  // same start time, it displays the first one (callee) above the second one
  // (caller) which is not what we want. After reversing Entries, callers
  // appear before callees, and therefore callers appear above callees in the
  // profiler.
  for (TimeTraceThreadProfiler &ttp : profilers)
    std::reverse(ttp.entries.begin(), ttp.entries.end());

  llvm::json::OStream jsonOS(os);
  jsonOS.objectBegin();
  jsonOS.attributeBegin("traceEvents");
  jsonOS.arrayBegin();

  // Emit all events for the main flame graph.
  auto writeEvent = [&](const auto &event, uint64_t tid) {
    auto startUs = event.getFlameGraphStartUs(ctx.startTime);
    auto durUs = event.getFlameGraphDurUs();
    jsonOS.object([&] {
      jsonOS.attribute("pid", ctx.pid);
      jsonOS.attribute("tid", int64_t(tid));
      jsonOS.attribute("ph", "X");
      jsonOS.attribute("ts", startUs);
      jsonOS.attribute("dur", durUs);
      jsonOS.attribute("name", event.name);
      if (!event.detail.empty()) {
        jsonOS.attributeObject(
            "args", [&] { jsonOS.attribute("detail", event.detail); });
      }
    });
  };
  for (const TimeTraceThreadProfiler &ttp : profilers)
    for (const TimeTraceProfilerEntry<true> &entry : ttp.entries)
      writeEvent(entry, ttp.tid);

  auto writeMetadataEvent = [&](const char *name, uint64_t tid, StringRef arg) {
    jsonOS.object([&] {
      jsonOS.attribute("cat", "");
      jsonOS.attribute("pid", ctx.pid);
      jsonOS.attribute("tid", int64_t(tid));
      jsonOS.attribute("ts", 0);
      jsonOS.attribute("ph", "M");
      jsonOS.attribute("name", name);
      jsonOS.attributeObject("args", [&] { jsonOS.attribute("name", arg); });
    });
  };

  writeMetadataEvent("process_name", ctx.pid, ctx.procName);
  for (const TimeTraceThreadProfiler &ttp : profilers)
    writeMetadataEvent("thread_name", ttp.tid, ttp.threadName);

  jsonOS.arrayEnd();
  jsonOS.attributeEnd();

  // Emit the absolute time when time profiling started. This can be used to
  // combine the profiling data from multiple processes and preserve actual time
  // intervals.
  jsonOS.attribute("beginningOfTime",
                   time_point_cast<microseconds>(ctx.beginningOfTime)
                       .time_since_epoch()
                       .count());

  jsonOS.objectEnd();
}

//===----------------------------------------------------------------------===//
// Event Stream Output
//===----------------------------------------------------------------------===//

namespace {

/// A more convenient representation for the event stream output.
struct Event {
  int64_t startUs;
  uint64_t tid;
  int64_t durUs;
  std::string name;
  std::string detail;
  int64_t endUs;

  Event() = default;

  /// Event representing time trace profiling entry
  Event(const TimeTraceProfilerEntry<true> &e, uint64_t tid,
        TimePointType startTime)
      : startUs(e.getFlameGraphStartUs(startTime)), tid(tid),
        durUs(e.getFlameGraphDurUs()), name(e.name), detail(e.detail),
        endUs(e.getFlameGraphStartUs(startTime) + e.getFlameGraphDurUs()) {}

  /// Event representing the 'end' of that event.
  Event toEnd() {
    Event result;
    result.startUs = endUs;
    result.tid = tid;
    result.durUs = -1;
    result.name = name;
    result.detail = detail;
    result.endUs = endUs;
    return result;
  }

  bool operator<(const Event &rhs) const {
    return std::tie(startUs, tid, durUs, name, detail) <
           std::tie(rhs.startUs, rhs.tid, rhs.durUs, rhs.name, rhs.detail);
  }

  void write(llvm::raw_pwrite_stream &os) const {
    os << llvm::format("%6d  %10d  ", tid, startUs);
    if (durUs >= 0)
      os << llvm::format("%10d  ", durUs);
    else
      os << "       END  ";
    os << name;
    if (!detail.empty())
      os << "/" << detail;
    os << "\n";
  }
};
} // namespace

void M::Detail::timeTraceProfilerWriteEventStream(llvm::raw_pwrite_stream &os) {
  assert(GlobalProfilerContext::instance && "profiler should be initialized");
  auto &ctx = *GlobalProfilerContext::instance;
  std::lock_guard<std::mutex> lock(ctx.lock);

  std::vector<Event> events;
  for (const TimeTraceThreadProfiler &ttp :
       llvm::make_pointee_range(ctx.profilers)) {
    for (const TimeTraceProfilerEntry<true> &e : ttp.entries) {
      events.emplace_back(e, ttp.tid, ctx.startTime);
      events.emplace_back(events.back().toEnd());
    }
  }
  std::sort(events.begin(), events.end());

  os << "   Tid     StartUs       DurUs  Name/Detail\n";
  os << "------  ----------  ----------  ------------------------------\n";
  for (const auto &event : events)
    event.write(os);
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

ErrorOrSuccess M::Detail::timeTraceProfilerWrite(StringRef preferredFileName,
                                                 StringRef fallbackFileName) {
  assert(GlobalProfilerContext::instance && "profiler should be initialized");

  // Set up filename base.
  std::string path = preferredFileName.str();
  if (path.empty())
    path = fallbackFileName == "-" ? "out" : fallbackFileName.str();

  std::error_code ec;

  {
    // Write time trace.
    std::string tracePath = path == "-" ? path : path + ".time-trace";
    llvm::raw_fd_ostream os(tracePath, ec, llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return Error(Twine("could not open ") + tracePath + "(" +
                   Twine(ec.message()) + ")");
    timeTraceProfilerWriteTrace(os);
  }

  {
    // Write the raw event stream.
    std::string eventStreamPath =
        path == "-" ? path : path + ".time-events.txt";
    llvm::raw_fd_ostream os(eventStreamPath, ec,
                            llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return Error(Twine("could not open ") + eventStreamPath + "(" +
                   Twine(ec.message()) + ")");
    timeTraceProfilerWriteEventStream(os);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TimeTraceProfilerEntry
//===----------------------------------------------------------------------===//

void M::Detail::timeTraceProfilerBeginImpl(
    StringRef name, llvm::function_ref<std::string()> detailFn) {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->begin(name, detailFn);
}

void M::Detail::timeTraceProfilerBeginImpl(StringRef name, StringRef detail) {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->begin(name, detail);
}

void M::Detail::timeTraceProfilerEndImpl() {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->end();
}

M::TimeTraceProfilerEntry<true>
M::Detail::timeTraceProfilerBeginEntryImpl(StringRef name, StringRef detail) {
  if (ThreadProfilerContext::get())
    return TimeTraceProfilerEntry<true>(name, detail);
  else
    return {};
}

M::TimeTraceProfilerEntry<true> M::Detail::timeTraceProfilerBeginEntryImpl(
    StringRef name, llvm::function_ref<std::string()> detailFn) {
  if (ThreadProfilerContext::get())
    return TimeTraceProfilerEntry<true>(name, detailFn());
  else
    return {};
}

void M::Detail::timeTraceProfilerEndEntryImpl(
    TimeTraceProfilerEntry<true> &&entry) {
  if (auto *profiler = ThreadProfilerContext::get())
    profiler->record(std::move(entry));
}

void M::Detail::timeTraceProfilerStartEntryImpl(
    TimeTraceProfilerEntry<true> &entry) {
  if (ThreadProfilerContext::get())
    entry.start = TimeTraceProfilerEntry<true>::ClockType::now();
}
