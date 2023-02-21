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
//   auto entry = TimeTraceProfilerEntry<true>::create("my_event_name");
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
//   auto entry = TimeTraceProfilerEntry<true>::create("my_event_name",
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

#ifndef SUPPORT_TIMEPROFILER_H
#define SUPPORT_TIMEPROFILER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <chrono>
#include <sstream>

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
/// Intialize the time trace profiler. This should be called on the main thread,
/// and must not be called again until after timeTraceProfilerDestroy has been
/// called.
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
// TimeTraceProfilerEntry
//===----------------------------------------------------------------------===//

///
/// Represents an open or completed time tracing entry to be captured.
/// However if Enabled is false, will be the trivial empty struct.
///
/// Here's the interface supported for entries:
///   -- Empty entry, never recorded.
///   TimeTraceProfilerEntry()
///
///   -- Start recording with name and result of detailFn.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static TimeTraceProfilerEntry
///   create(StringRef name, llvm::function_ref<std::string()> detailFn);
///
///   -- Ditto, but detail is literal string.
///   -- CAUTION: Both must be literals to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   static TimeTraceProfilerEntry create(StringRef name, StringRef detail);
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
///   TimeTraceProfilerEntry withNameSuffix(StringRef suffix) const;
///
///   -- Return copy of this entry, with clock restarted, and with result
///   -- of detailFn appended to detail.
///   -- CAUTION: Function must be a literal to guarantee zero-cost when
///   -- profiling disabled at compile time.
///   TimeTraceProfilerEntry withDetailSuffix(
///       llvm::function_ref<std::string()>  detailFn) const;
///
///   -- Return copy of this entry, with clock restarted, but with possibly
///   -- distinct 'Enabled' template parameter.
///   template <typename Result> Result copy() const;
///
template <bool Enabled>
struct TimeTraceProfilerEntry {};

namespace Detail {
// For internal use only. Returns true if profiling is active.
bool timeTraceProfilerIsActive();

// For internal use only. Records entry on the currently active profiler.
void timeTraceProfilerRecord(TimeTraceProfilerEntry<true> &&entry);

// For internal use only. Begins a time section with name and detail on the
// currently active profiler.
void timeTraceProfilerBegin(TimeTraceProfilerEntry<true> &&entry);

// For internal use only. Ends the last timing section began on the currently
// active profiler.
void timeTraceProfilerEnd();
} // namespace Detail

/// Disabled profiling entry. Everything will be a no-op provided arguments
/// are literals.
template <>
struct TimeTraceProfilerEntry<false> {
  // No copy, only move.
  TimeTraceProfilerEntry(const TimeTraceProfilerEntry &) = delete;
  TimeTraceProfilerEntry &operator=(const TimeTraceProfilerEntry &) = delete;
  TimeTraceProfilerEntry(TimeTraceProfilerEntry &&) = default;
  TimeTraceProfilerEntry &operator=(TimeTraceProfilerEntry &&) = default;

  TimeTraceProfilerEntry() = default;

  static TimeTraceProfilerEntry
  create(StringRef name, llvm::function_ref<std::string()> detailFn) {
    return {};
  }
  static TimeTraceProfilerEntry create(StringRef name, StringRef detail = "") {
    return {};
  }

  void restart() {}
  void record() && {}
  TimeTraceProfilerEntry withNameSuffix(StringRef suffix) const { return {}; }
  TimeTraceProfilerEntry
  withDetailSuffix(llvm::function_ref<std::string()> detailFn) const {
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
struct TimeTraceProfilerEntry<true> {
  // No copy, only move.
  TimeTraceProfilerEntry(const TimeTraceProfilerEntry &) = delete;
  TimeTraceProfilerEntry &operator=(const TimeTraceProfilerEntry &) = delete;
  TimeTraceProfilerEntry(TimeTraceProfilerEntry &&) = default;
  TimeTraceProfilerEntry &operator=(TimeTraceProfilerEntry &&) = default;

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

  TimePointType start;
  TimePointType end;
  std::string name;
  std::string detail;

  TimeTraceProfilerEntry() = default;

#if defined(TRACE_IN_REAL_TIME)
#define TRACE(X) X
#else
#define TRACE(X)
#endif

  static TimeTraceProfilerEntry
  create(StringRef name, llvm::function_ref<std::string()> detailFn) {
    if (!Detail::timeTraceProfilerIsActive())
      return {};
    return TimeTraceProfilerEntry(name, detailFn());
  }

  static TimeTraceProfilerEntry create(StringRef name, StringRef detail = "") {
    if (!Detail::timeTraceProfilerIsActive())
      return {};
    return TimeTraceProfilerEntry(name, detail);
  }

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

  TimeTraceProfilerEntry withNameSuffix(StringRef suffix) const {
    if (name.empty())
      return {};
    return TimeTraceProfilerEntry(Twine(name).concat(Twine(suffix)).str(),
                                  std::string(detail));
  }

  TimeTraceProfilerEntry
  withDetailSuffix(llvm::function_ref<std::string()> detailFn) const {
    if (name.empty())
      return {};
    return TimeTraceProfilerEntry(name, Twine(detail).concat(detailFn()).str());
  }

  template <typename Result>
  Result copy() const {
    if (name.empty())
      return {};
    return Result::create(name, detail);
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

private:
  TimeTraceProfilerEntry(StringRef name, StringRef detail)
      : start(ClockType::now()), end(TimePointType()), name(name),
        detail(detail) {
    TRACE(llvm::dbgs() << toImmediateDebugString());
  }

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

  explicit TimeTraceScope(TimeTraceProfilerEntry<Enabled> &&entry)
      : entry(std::move(entry)) {}

  TimeTraceScope(StringRef name, llvm::function_ref<std::string()> detailFn) {
    entry = TimeTraceProfilerEntry<Enabled>::create(name, detailFn);
  }

  explicit TimeTraceScope(StringRef name, StringRef detail = "") {
    entry = TimeTraceProfilerEntry<Enabled>::create(name, detail);
  }

  ~TimeTraceScope() { std::move(entry).record(); }

  TimeTraceProfilerEntry<Enabled> entry;
};

// The trivial deduction guide.
template <bool Enabled>
TimeTraceScope(TimeTraceProfilerEntry<Enabled> &&) -> TimeTraceScope<Enabled>;

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
    Detail::timeTraceProfilerBegin(
        TimeTraceProfilerEntry<true>::create(name, detailFn));
}

template <bool Enabled = true>
void timeTraceProfilerBegin(StringRef name, StringRef detail) {
  if constexpr (Enabled)
    Detail::timeTraceProfilerBegin(
        TimeTraceProfilerEntry<true>::create(name, detail));
}

/// Manually end the last time section. However if Enabled is false then
/// the method is a no-op.
template <bool Enabled = true>
void timeTraceProfilerEnd() {
  if constexpr (Enabled)
    Detail::timeTraceProfilerEnd();
}

} // namespace M

#endif
