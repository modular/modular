//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This provides lightweight and dependency-free machinery to trace execution
// time around arbitrary code. Three API flavors are available.
//
// The primary API uses a RAII object to trigger tracing:
//
// \code
//   {
//     TimeTraceScope<> scope("my_event_name");
//     ...my code...
//   }
// \endcode
//
// If the code to be profiled does not have a natural lexical scope then
// it is also possible to start and end events with respect to an implicit
// per-thread stack of profiling entries:
//
// \code
//   timeTraceProfilerBegin("my_event_name");
//   ...my code...
//   timeTraceProfilerEnd();  // must be called on all control flow paths
// \endcode
//
// Finally, it is also possible to manually create, begin and complete time
// profiling entries. This API allows an entry to be created in one
// context, stored, then completed in another. The completing context need not
// be on the same thread as the creating context:
//
// \code
//   auto entry = ProfilerEntry<true>::create("my_event_name");
//   ...
//   // Possibly on a different thread
//   entry.restart(); // optional, if wish to decouple
//                    // setup from start time
//   ...my code...
//   std::move(entry).record();
// \endcode
//
// Time profiling entries can be given an arbitrary name and, optionally,
// an arbitrary 'detail' string. The resulting trace will include 'Total'
// entries summing the time spent for each name. Thus, it's best to choose
// names to be fairly generic, and rely on the detail field to capture
// everything else of interest.
//
// To avoid lifetime issues name and detail strings are copied into the event
// entries at their time of creation. Care should be taken to make string
// construction cheap to prevent 'Heisenperf' effects. In particular, the
// 'detail' argument may be a string-returning closure:
//
// \code
//   int n;
//   {
//     TimeTraceScope scope("my_event_name",
//                          [n]() { return (Twine("x=") + Twine(n)).str(); });
//     ...my code...
//   }
// \endcode
// The closure will not be called if tracing is disabled. Otherwise, the
// resulting string will be directly moved into the entry.
//
// If string construction is a significant cost it is possible to construct
// the entry outside the critical section:
//
// \code
//   auto entry = ProfilerEntry<true>::create("my_event_name",
//                                            [=]() { ... expensive ... });
//   ...non critical code...
//   entry.restart();
//   ...my critical code...
//   std::move(entry).record();
// \endcode
//
// The main process should first construct a TimeTraceProfiler, which is used
// to anchor the various timing functionality, and exposes support for writing
// to various formats. Note that only one such profiler may be active at any
// given time.
//
// Timestamps come from std::chrono::high_resolution_clock, so all threads
// see the same time at the highest available resolution.
//
// Currently, there are a number of compatible viewers:
//  - chrome://tracing is the original chromium trace viewer.
//  - http://ui.perfetto.dev is the replacement for the above, under active
//    development by Google as part of the 'Perfetto' project.
//  - https://www.speedscope.app/ has also been reported as an option.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TIMEPROFILER_H
#define SUPPORT_PROFILING_TIMEPROFILER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <chrono>
#include <sstream>
#include <variant>

namespace llvm {
class raw_pwrite_stream;
}

/// If you are desperate to see a trace of begin and end events as they
/// happen define the following. Obviously this is expensive and likely
/// to change the concurrent behaviour of your program.
/// TODO: Why does LLVM_DEBUG not work here?

// #define TRACE_IN_REAL_TIME

namespace M {
//===----------------------------------------------------------------------===//
// TimeTraceProfiler
//===----------------------------------------------------------------------===//

namespace Detail {
/// Initialize the time trace profiler. This should be called on the main
/// thread, and must not be called again until after timeTraceProfilerDestroy
/// has been called.
void timeTraceProfilerInitialize(unsigned timeTraceGranularity,
                                 StringRef procName);
/// Destroy the time trace profiler. This should be called on the main thread.
void timeTraceProfilerDestroy();

/// Write profiling data to output stream.
/// Data produced is JSON, in Chrome "Trace Event" format, see
/// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
void timeTraceProfilerWriteTrace(llvm::raw_pwrite_stream &os);

/// Write raw event stream to output stream.
/// Data is textual, one line per event.
void timeTraceProfilerWriteEventStream(llvm::raw_pwrite_stream &OS);

/// Append given input shape to internal list.
/// These will be included in metadata written to output stream.
void timeTraceProfilerAddInputShape(const std::string &shape);

/// Write profiling data to a file.
/// The function will write to preferredFileName if provided, if not then will
/// write to fallbackFileName appending .time-trace. Returns a StringError
/// indicating a failure if the function is unable to open the file for writing.
ErrorOrSuccess timeTraceProfilerWrite(StringRef preferredFileName,
                                      StringRef fallbackFileName);
} // namespace Detail

/// This class represents the main time trace profiler, of which only one should
/// ever be active at a given time.
struct TimeTraceProfiler {
  TimeTraceProfiler(unsigned timeTraceGranularity, StringRef procName) {
    Detail::timeTraceProfilerInitialize(timeTraceGranularity, procName);
  }
  ~TimeTraceProfiler() { Detail::timeTraceProfilerDestroy(); }

  void addInputShape(const std::string &shape) {
    Detail::timeTraceProfilerAddInputShape(shape);
  }

  //===--------------------------------------------------------------------===//
  // Output

  /// Write profiling data to a file.
  /// The function will write to preferredFileName if provided, if not then will
  /// write to fallbackFileName appending .time-trace. Returns a StringError
  /// indicating a failure if the function is unable to open the file for
  /// writing.
  ErrorOrSuccess write(StringRef preferredFileName,
                       StringRef fallbackFileName) {
    return Detail::timeTraceProfilerWrite(preferredFileName, fallbackFileName);
  }
};

//===----------------------------------------------------------------------===//
// ProfilerEntry
//===----------------------------------------------------------------------===//

///
/// Represents an open or completed timing/sampling tracing entry. However
/// if Enabled is false, will be the trivial empty struct. Timing entries
/// capture the beginning and end timestamps for a named event. Sampling
/// entries capture a single size_t value sampling a named value of interest.
/// The duration of sampling entries is ignored by viewer, however sampling
/// entries must still be recorded by invoking 'record()'.
///
/// Here's the interface supported for entries:
///   -- Empty entry, never recorded.
///   ProfilerEntry()
///
///   -- Start recording a timing entry with name and result of detailFn.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry
///   create(StringRef name, llvm::function_ref<std::string()> detailFn);
///
///   -- Ditto, but detail is literal string.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry create(StringRef name, StringRef detail);
///
///   -- Start recording a sampling entry with name and result of valueFn.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry
///   create(StringRef name, llvm::function_ref<size_t()> valueFn);
///
///   -- Ditto, but value is already computed.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static ProfilerEntry create(StringRef name, size_t value);
///
///   -- Return true if entry is empty.
///   bool empty() const;
///
///   -- Restart the entry's clock.
///   void restart();
///
///   -- Stop the entry's clock, and move the entry into the profiling
///   -- database.
///   void record() &&
///
///   -- Return copy of this entry, with clock restarted, and with suffix
///   -- appended to name.
///   -- CAUTION: Suffix must be a literal to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   ProfilerEntry withNameSuffix(StringRef suffix) const;
///
///   -- Return copy of this entry, with clock restarted, and with result
///   -- of detailFn appended to detail.
///   -- CAUTION: Function must be a literal to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   ProfilerEntry withDetailSuffix(
///                  llvm::function_ref<std::string()> detailFn) const;
///
///   -- Return copy of this entry, with clock restarted, but with possibly
///   -- distinct 'Enabled' template parameter.
///   template <typename Result> Result copy() const;
///
template <bool Enabled>
struct ProfilerEntry {};

namespace Detail {
// For internal use only. Returns true if profiling is active.
bool timeTraceProfilerIsActive();

// For internal use only. Records entry on the currently active profiler.
void timeTraceProfilerRecord(ProfilerEntry<true> &&entry);

// For internal use only. Begins a time section with name and detail on the
// currently active profiler.
void timeTraceProfilerBegin(ProfilerEntry<true> &&entry);

// For internal use only. Ends the last timing section began on the currently
// active profiler.
void timeTraceProfilerEnd();
} // namespace Detail

/// Disabled profiling entry. Everything will be a no-op provided arguments
/// are literals.
template <>
struct ProfilerEntry<false> {
  // No copy, only move.
  ProfilerEntry(const ProfilerEntry &) = delete;
  ProfilerEntry &operator=(const ProfilerEntry &) = delete;
  ProfilerEntry(ProfilerEntry &&) = default;
  ProfilerEntry &operator=(ProfilerEntry &&) = default;

  ProfilerEntry() = default;

  static ProfilerEntry create(StringRef name,
                              llvm::function_ref<std::string()> detailFn) {
    return {};
  }
  static ProfilerEntry create(StringRef name, StringRef detail = "") {
    return {};
  }
  static ProfilerEntry create(StringRef name,
                              llvm::function_ref<size_t()> valueFn) {
    return {};
  }
  static ProfilerEntry create(StringRef name, size_t value) { return {}; }

  bool empty() { return true; }
  void restart() {}
  void record() && {}
  ProfilerEntry withNameSuffix(StringRef suffix) const { return {}; }
  ProfilerEntry
  withDetailSuffix(llvm::function_ref<std::string()> detailFn) const {
    return {};
  }
  ProfilerEntry
  withNameDetailSuffix(StringRef suffix,
                       llvm::function_ref<std::string()> detailFn) const {
    return {};
  }
  template <typename Result>
  Result copy() const {
    return {};
  }
};

/// Enabled profiling entry. Strings and times are constructed only if an
/// active profiler is available. Otherwise all instances will be the default
/// profiling instance.
///
/// Operations on an already recorded or moved entry are no-ops.
template <>
struct ProfilerEntry<true> {
  // No copy, only move.
  ProfilerEntry(const ProfilerEntry &) = delete;
  ProfilerEntry &operator=(const ProfilerEntry &) = delete;
  ProfilerEntry(ProfilerEntry &&) = default;
  ProfilerEntry &operator=(ProfilerEntry &&) = default;

  // We use the high_resolution_clock for maximum precision.
  // It may not be steady (ClockType::is_steady may be false), which means
  // it is possible for profiles to yield invalid durations during leap
  // second transitions or other system clock adjustments. This rare glitch
  // seems worthwhile in exchange for the precision.
  // Under linux glibc++ the high_resolution_clock is consistent across threads
  // which is necessary for building cross-thread entries.
  // It is unknown whether that's the case under Windows, and the C++ standard
  // does not appear to impose any thread consistency on any of the clocks.
  using ClockType = std::chrono::high_resolution_clock;
  using TimePointType = std::chrono::time_point<ClockType>;
  using FloatUsType = std::chrono::duration<double, std::micro>;

  using PayloadType = std::variant<
      // For timing entries, extra details of the event (may be empty).
      std::string,
      // For sampling entries, current value of sampled value.
      size_t>;

  TimePointType start;
  TimePointType end;
  std::string name;
  PayloadType payload;

  ProfilerEntry() = default;

#if defined(TRACE_IN_REAL_TIME)
#define TRACE(X) X
#else
#define TRACE(X)
#endif

  static ProfilerEntry create(StringRef name,
                              llvm::function_ref<std::string()> detailFn) {
    if (!Detail::timeTraceProfilerIsActive())
      return {};
    return ProfilerEntry(name, detailFn());
  }

  static ProfilerEntry create(StringRef name, StringRef detail = "") {
    if (!Detail::timeTraceProfilerIsActive())
      return {};
    return ProfilerEntry(name, detail);
  }

  static ProfilerEntry create(StringRef name,
                              llvm::function_ref<size_t()> valueFn) {
    if (!Detail::timeTraceProfilerIsActive())
      return {};
    return ProfilerEntry(name, valueFn());
  }

  static ProfilerEntry create(StringRef name, size_t value) {
    if (!Detail::timeTraceProfilerIsActive())
      return {};
    return ProfilerEntry(name, value);
  }

  bool empty() { return name.empty(); }

  void restart() {
    if (name.empty())
      return;
    start = ClockType::now();
    TRACE(llvm::dbgs() << toImmediateDebugString());
  }

  void record() && {
    if (name.empty())
      return;
    end = ClockType::now();
    TRACE(llvm::dbgs() << toImmediateDebugString());
    Detail::timeTraceProfilerRecord(std::move(*this));
  }

  ProfilerEntry withNameSuffix(StringRef suffix) const {
    if (name.empty())
      return {};
    return ProfilerEntry(Twine(name).concat(Twine(suffix)).str(), getDetail());
  }

  ProfilerEntry
  withDetailSuffix(llvm::function_ref<std::string()> detailFn) const {
    if (name.empty())
      return {};
    return ProfilerEntry(name, Twine(getDetail()).concat(detailFn()).str());
  }

  ProfilerEntry
  withNameDetailSuffix(StringRef suffix,
                       llvm::function_ref<std::string()> detailFn) const {
    if (name.empty())
      return {};
    return ProfilerEntry(Twine(name).concat(name).concat(Twine(suffix)).str(),
                         Twine(getDetail()).concat(detailFn()).str());
  }

  template <typename Result>
  Result copy() const {
    if (name.empty())
      return {};
    if (isTiming())
      return Result::create(name, getDetail());
    else
      return Result::create(name, getValue());
  }

  // Calculate timings for FlameGraph. Convert to floating point
  // microsecond representation so that caller and callee aren't
  // truncated to have the same start time
  FloatUsType::rep getFlameGraphStartUs(TimePointType startTime) const {
    return FloatUsType(start - startTime).count();
  }

  FloatUsType::rep getFlameGraphDurUs() const {
    return FloatUsType(end - start).count();
  }

  bool isTiming() const { return std::holds_alternative<std::string>(payload); }
  bool isSampling() const { return std::holds_alternative<size_t>(payload); }
  const std::string &getDetail() const {
    return std::get<std::string>(payload);
  }
  size_t getValue() const { return std::get<size_t>(payload); }

private:
  ProfilerEntry(StringRef name, StringRef detail)
      : start(ClockType::now()), end(TimePointType()), name(name),
        payload(std::string(detail)) {
    TRACE(llvm::dbgs() << toImmediateDebugString());
  }

  ProfilerEntry(StringRef name, size_t value)
      : start(ClockType::now()), end(TimePointType()), name(name),
        payload(value) {}

  /// Returns brief description of entry as constructed so far. We return
  /// a string rather than streaming to the final output for atomicity of
  /// output. The same entry may appear to have multiple beginnings if it
  /// is restarted, however the entry will have at most one end.
  std::string toImmediateDebugString();

#undef TRACE
};

//===----------------------------------------------------------------------===//
// TimeTraceScope
//===----------------------------------------------------------------------===//

/// RAII class to automatically record the constructed or given profile entry
/// when the object goes out of scope.
template <bool Enabled = true>
struct TimeTraceScope {
  TimeTraceScope() = delete;
  TimeTraceScope(const TimeTraceScope &) = delete;
  TimeTraceScope &operator=(const TimeTraceScope &) = delete;
  TimeTraceScope(TimeTraceScope &&) = delete;
  TimeTraceScope &operator=(TimeTraceScope &&) = delete;

  explicit TimeTraceScope(ProfilerEntry<Enabled> &&entry)
      : entry(std::move(entry)) {}

  TimeTraceScope(StringRef name, llvm::function_ref<std::string()> detailFn) {
    entry = ProfilerEntry<Enabled>::create(name, detailFn);
  }

  explicit TimeTraceScope(StringRef name, StringRef detail = "") {
    entry = ProfilerEntry<Enabled>::create(name, detail);
  }

  ~TimeTraceScope() { std::move(entry).record(); }

  ProfilerEntry<Enabled> entry;
};

// The trivial deduction guide.
template <bool Enabled>
TimeTraceScope(ProfilerEntry<Enabled> &&) -> TimeTraceScope<Enabled>;

//===----------------------------------------------------------------------===//
// Procedural begin/end interface
//===----------------------------------------------------------------------===//

/// Manually begin a time section, with the given name and detail.
/// Profiler copies the string data, so the pointers can be given into
/// temporaries. Time sections can be hierarchical; every Begin must have a
/// matching End pair but they can nest. However if Enabled is false then
/// methods are a no-op.
template <bool Enabled = true>
void timeTraceProfilerBegin(StringRef name,
                            llvm::function_ref<std::string()> detailFn) {
  if constexpr (Enabled)
    Detail::timeTraceProfilerBegin(ProfilerEntry<true>::create(name, detailFn));
}

template <bool Enabled = true>
void timeTraceProfilerBegin(StringRef name, StringRef detail) {
  if constexpr (Enabled)
    Detail::timeTraceProfilerBegin(ProfilerEntry<true>::create(name, detail));
}

/// Manually end the last time section. However if Enabled is false then
/// the method is a no-op.
template <bool Enabled = true>
void timeTraceProfilerEnd() {
  if constexpr (Enabled)
    Detail::timeTraceProfilerEnd();
}

} // namespace M

#endif // SUPPORT_PROFILING_TIMEPROFILER_H
