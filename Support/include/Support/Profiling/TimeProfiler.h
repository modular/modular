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
// Finally, it is also possible to manually create and complete time profiling
// entries. This API allows an entry to be created in one context, stored,
// then completed in another. The completing context need not be on the same
// thread as the creating context:
//
// \code
//   auto entry = ProfilerEntry<true>::create(StringLiteral("my_event_name"));
//   ...
//   // Possibly on a different thread
//   ...my code...
//   std::move(entry).record();
// \endcode
//
// Time profiling entries can be given an arbitrary name and, optionally,
// an arbitrary 'detail' string.
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
//  - http://ui.perfetto.dev is the Chrome profile viewer, under active
//    development by Google as part of the 'Perfetto' project.
//  - https://www.speedscope.app/ has also been reported as an option.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_TIMEPROFILER_H
#define SUPPORT_PROFILING_TIMEPROFILER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <sstream>

namespace llvm {
class raw_pwrite_stream;
namespace json {
class OStream;
}
} // namespace llvm

/// Define to 1 to have all enabled profiling events dumped to stderr as they
/// are created. The profiling entry type must be enabled, and profiling must
/// be active.
#define TRACE_IN_REAL_TIME 0

namespace M {

/// Function to call to return profiling entry name or description string.
using ProfilerPrintFn = llvm::function_ref<std::string()>;

/// Globally unique id for every CreateEvent and SampleEvent. Zero denotes
/// no-event.
using ProfilerEventId = size_t;

/// An InternableString is a StringRef with underlying lifetime guaranteed
/// at least up until the next call to TimeTraceProfiler::intern(). A typical
/// example would be a string literal from a MEFFile.
struct InternableString : StringRef {
  using StringRef::StringRef;
};

namespace ProfilingDetail {

constexpr bool kProfilingEnabled = MODULAR_LLCL_MAX_PROFILING_LEVEL > 0;

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
using DurationType = std::chrono::duration<ClockType::rep, ClockType::period>;

//===----------------------------------------------------------------------===//
// ProfilingDetail::Label
//===----------------------------------------------------------------------===//

/// A way to derive the name or detail string for a profiling entry. When safe,
/// allows string copies to be avoided until either the final profile is being
/// written, or a call is made to TimeTraceProfiler::intern().
///
/// Since it is so common, also allows the combination of a string and a
/// single uint64_t value, rendered as "name:42", without requiring the caller
/// to do any string manipulation.
class Label {
public:
  static constexpr int kIntPayloadBits = 62;
  static constexpr uint64_t kNoIntPayload = (1ul << kIntPayloadBits) - 1;

  /*implicit*/
  constexpr Label(uint64_t value = kNoIntPayload)
      : tag(kLiteral), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // No string copy, no interning.
  /*implicit*/
  constexpr Label(StringLiteral stringLiteral, uint64_t value = kNoIntPayload)
      : stringPayload(stringLiteral), tag(kLiteral), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // CAUTION: No string copy, but may need to be interned.
  /*implicit*/
  constexpr Label(InternableString internableString,
                  uint64_t value = kNoIntPayload)
      : stringPayload(internableString), tag(kLiteral), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // Moves string.
  /*implicit*/
  Label(std::string string, uint64_t value = kNoIntPayload)
      : stringPayload(std::move(string)), tag(kOwned), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // CAUTION: Copies string.
  /*implicit*/
  Label(StringRef stringRef, uint64_t value = kNoIntPayload)
      : stringPayload(stringRef.str()), tag(kOwned), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  // CAUTION: Calls function and moves result string.
  /*implicit*/
  Label(ProfilerPrintFn printFn, uint64_t value = kNoIntPayload)
      : stringPayload(printFn()), tag(kOwned), intPayload(value) {
    assert(intPayload == value && "overflow for int payload");
  }

  ~Label() { reset(); }

  /// If the label possibly contains a borrow, evaluate it to its owning
  /// std::string form.
  void intern();

  /// Returns the label in string form.
  std::string toString() const;

  bool empty() const;
  std::optional<uint64_t> getInt() const {
    return intPayload < kNoIntPayload ? intPayload : std::optional<uint64_t>();
  }

  // No copy, only move.
  Label(const Label &) = delete;
  Label &operator=(const Label &) = delete;
  Label(Label &&that) { moveFrom(that); }
  Label &operator=(Label &&that) {
    reset();
    moveFrom(that);
    return *this;
  }

private:
  void reset();
  void moveFrom(Label &that);

  /// String-like payload.
  union StringPayload {
    /// String literal. Never needs to be interned.
    StringLiteral stringLiteral;
    /// Like a string literal, but may need to be interned before the
    /// underlying object from which the string is borrowed is freed.
    InternableString internableString;
    /// An owned string, safe from all lifetime issues.
    std::string ownedString;

    constexpr StringPayload() : stringLiteral("") {}
    constexpr StringPayload(StringLiteral stringLiteral)
        : stringLiteral(stringLiteral) {}
    constexpr StringPayload(InternableString internableString)
        : internableString(internableString) {}
    StringPayload(std::string &&ownedString)
        : ownedString(std::move(ownedString)) {}
    ~StringPayload() {}
  } stringPayload;

  /// Which of the above is the true payload.
  enum Tag { kLiteral = 0, kInternable = 1, kOwned = 2 };
  uint64_t tag : 2;

  /// Additional integer value, or kNoIntPayload if none.
  uint64_t intPayload : kIntPayloadBits;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::BeginEvent
//===----------------------------------------------------------------------===//

/// Represents the creation of a timing profiling entry.
struct BeginEvent {
  uint64_t seqNum;
  ProfilerEventId id;
  ProfilerEventId parentId = 0;
  TimePointType start = ClockType::now();
  Label name;
  Label detail;

  template <typename NameStr>
  BeginEvent(uint64_t seqNum, ProfilerEventId id, ProfilerEventId parentId,
             NameStr &&name, uint64_t nameValue = Label::kNoIntPayload)
      : seqNum(seqNum), id(id), parentId(parentId),
        name(std::forward<NameStr>(name), nameValue) {}

  template <typename NameStr, typename DetailStr>
  BeginEvent(uint64_t seqNum, ProfilerEventId id, ProfilerEventId parentId,
             NameStr &&name, DetailStr &&detail,
             uint64_t detailValue = Label::kNoIntPayload)
      : seqNum(seqNum), id(id), parentId(parentId),
        name(std::forward<NameStr>(name)),
        detail(std::forward<DetailStr>(detail), detailValue) {}

  /// Intern the name and detail labels.
  void intern() {
    name.intern();
    detail.intern();
  }

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::EndEvent
//===----------------------------------------------------------------------===//

/// Represents the end of a timing profiling entry. It is valid for
/// a profiling entry to be begin on one thread and ended on another.
struct EndEvent {
  uint64_t seqNum;
  ProfilerEventId id;
  TimePointType end = ClockType::now();

  EndEvent(uint64_t seqNum, ProfilerEventId id) : seqNum(seqNum), id(id) {}

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::SampleEvent
//===----------------------------------------------------------------------===//

/// Represents the sampling of an integer value.
struct SampleEvent {
  uint64_t seqNum;
  TimePointType stamp = ClockType::now();
  uint64_t value;
  Label name;

  template <typename NameStr>
  SampleEvent(uint64_t seqNum, uint64_t value, NameStr &&name,
              uint64_t nameValue = Label::kNoIntPayload)
      : seqNum(seqNum), value(value),
        name(std::forward<NameStr>(name), nameValue) {}

  /// Intern the name label.
  void intern() { name.intern(); }

  void dump() const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::EventList
//===----------------------------------------------------------------------===//

constexpr size_t kEventListBlockSize = 1024;

template <typename T>
struct EventListEntry {
  std::unique_ptr<EventListEntry> tail;
  SmallVector<T, kEventListBlockSize> events;
};

template <typename T>
struct EventList {
  std::unique_ptr<EventListEntry<T>> head;
  EventListEntry<T> *last = nullptr;

  template <typename... Args>
  const T &emplace_back(Args &&...args) {
    if (head == nullptr) {
      assert(last == nullptr);
      head = std::make_unique<EventListEntry<T>>();
      last = head.get();
    }
    if (last->events.size() >= kEventListBlockSize) {
      assert(last->tail == nullptr);
      last->tail = std::make_unique<EventListEntry<T>>();
      last = last->tail.get();
    }
    return last->events.emplace_back(std::forward<Args>(args)...);
  }

  void enumerate(llvm::function_ref<void(const T &)> func) const {
    EventListEntry<T> *curr = head.get();
    while (curr) {
      llvm::for_each(curr->events, func);
      curr = curr->tail.get();
    }
  }

  void enumerate(llvm::function_ref<void(T &)> func) {
    EventListEntry<T> *curr = head.get();
    while (curr) {
      llvm::for_each(curr->events, func);
      curr = curr->tail.get();
    }
  }
};

using BeginEventList = EventList<BeginEvent>;
using EndEventList = EventList<EndEvent>;
using SampleEventList = EventList<SampleEvent>;

//===----------------------------------------------------------------------===//
// ProfilingDetail::CompletedEntry
//===----------------------------------------------------------------------===//

/// A completed profiling entry, built from the combination of begin, end,
/// sampling and parent events.
struct CompletedEntry {
  enum Flavor { kBegin = 0, kEnd = 1, kSample = 2 };
  Flavor flavor = kBegin;
  uint64_t seqNum = 0;
  ProfilerEventId id = 0;
  ProfilerEventId parentId = 0;
  uint64_t tid = 0;
  TimePointType start;
  TimePointType end;
  DurationType dur;
  std::string name;
  std::string detail;
  uint64_t value = 0;

  CompletedEntry() = default;

  explicit CompletedEntry(const BeginEvent &beginEvent)
      : flavor(kBegin), seqNum(beginEvent.seqNum), id(beginEvent.id),
        parentId(beginEvent.parentId), start(beginEvent.start),
        name(beginEvent.name.toString()), detail(beginEvent.detail.toString()) {
  }

  CompletedEntry(uint64_t tid, const EndEvent &endEvent)
      : flavor(kEnd), seqNum(endEvent.seqNum), id(endEvent.id), tid(tid),
        start(endEvent.end), end(endEvent.end) {}

  CompletedEntry(uint64_t tid, const SampleEvent &sampleEvent)
      : flavor(kSample), seqNum(sampleEvent.seqNum), tid(tid),
        start(sampleEvent.stamp), end(sampleEvent.stamp),
        name(sampleEvent.name.toString()), value(sampleEvent.value) {}

  /// Update this begin entry with details from end event.
  void mergeEndIntoBegin(uint64_t endTid, const EndEvent &endEvent);

  /// Update this end entry with details from begin entry.
  void mergeBeginIntoEnd(const CompletedEntry &beginEntry);

  /// Update this entry to include the name and details from all of parents.
  void prependParents(ArrayRef<const CompletedEntry *> parents);

  /// Global temporal ordering, with ties broken by per-thread sequence numbers.
  bool operator<(const CompletedEntry &that) const {
    return std::tie(start, tid, seqNum) <
           std::tie(that.start, that.tid, seqNum);
  }

  /// Prints entry in JSON form to os.
  void print(llvm::json::OStream &os, TimePointType startTime,
             llvm::sys::Process::Pid pid, DurationType granularity) const;

  /// Prints entry in compact form to os.
  void print(llvm::raw_pwrite_stream &os, TimePointType startTime) const;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::TimeTraceThreadProfiler
//===----------------------------------------------------------------------===//

struct TimeTraceThreadProfiler {
  explicit TimeTraceThreadProfiler(uint16_t threadIndex);

  /// Begin a new timing entry, and return its globally unique id.
  template <typename... Args>
  ProfilerEventId begin(Args &&...args) {
    ProfilerEventId id = nextId++;
    [[maybe_unused]] const BeginEvent &event = beginEvents.emplace_back(
        nextSeqNum++, id, /*parentId=*/(ProfilerEventId)0,
        std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
    return id;
  }

  /// Begin a new timing entry with the given parent, and return its globally
  /// unique id.
  template <typename... Args>
  ProfilerEventId beginWithParent(ProfilerEventId parentId, Args &&...args) {
    ProfilerEventId id = nextId++;
    [[maybe_unused]] const BeginEvent &event = beginEvents.emplace_back(
        nextSeqNum++, id, parentId, std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
    return id;
  }

  /// End the timing entry with the given id. The event need not have been
  /// begun on this thread.
  void end(ProfilerEventId id) {
    [[maybe_unused]] const EndEvent &event =
        endEvents.emplace_back(nextSeqNum++, id);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
  }

  /// Begin a new timing entry, and push it onto the stack of currently
  /// running entries. A corresponding call to endAndPop() must be made
  /// from the same thread.
  template <typename... Args>
  void beginAndPush(Args &&...args) {
    ProfilerEventId id = begin(std::forward<Args>(args)...);
    stack.push_back(id);
  }

  /// End the most recently pushed timing event.
  void endAndPop() {
    assert(!stack.empty() && "unbalanced push/pop");
    end(stack.pop_back_val());
  }

  /// Record a sampling entry.
  template <typename... Args>
  void sample(uint64_t value, Args &&...args) {
    [[maybe_unused]] SampleEvent &event = sampleEvents.emplace_back(
        nextSeqNum++, value, std::forward<Args>(args)...);
#if TRACE_IN_REAL_TIME
    event.dump();
#endif
  }

  // Intern all event labels.
  void intern();

  /// The id of the thread this profiler is running on.
  const uint64_t tid;

  /// The name of the thread this profiler is running on.
  SmallString<0> threadName;

  /// Next available begin event id.
  ProfilerEventId nextId;

  /// The next sequence number to use for all events. This helps us preserve
  /// the per-thread temporal ordering of events even when events have the
  /// same start time.
  uint64_t nextSeqNum = 0;

  /// The stack of begun but not yet ended timing events.
  SmallVector<ProfilerEventId> stack;

  /// Recorded events.
  BeginEventList beginEvents;
  EndEventList endEvents;
  SampleEventList sampleEvents;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::ThreadProfilerContext
//===----------------------------------------------------------------------===//

/// This class represents the profiler context for a specific thread.
struct ThreadProfilerContext {
  ~ThreadProfilerContext();

  /// Return the profiler instance for this thread, or nullptr if profiling
  /// is not active.
  static TimeTraceThreadProfiler *get();

  /// The profiler attached to this thread.
  TimeTraceThreadProfiler *profiler = nullptr;
};

//===----------------------------------------------------------------------===//
// ProfilingDetail::GlobalProfilerContext
//===----------------------------------------------------------------------===//

/// This class represents the main context used for profiling.
struct GlobalProfilerContext {
  GlobalProfilerContext(DurationType granularity, StringRef name);

  /// Collect all the begin, end, and sample events over all threads, reconcile
  /// them, and return them as timing entries sorted by time then thread id.
  std::vector<CompletedEntry> getCompletedEntries();

  /// Write all the completed entries in JSON form to os, using format in:
  /// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
  void writeJsonTrace(llvm::raw_pwrite_stream &os,
                      ArrayRef<CompletedEntry> entries);

  /// Write all the completed entries in plaint text form to os.
  void writeTextTrace(llvm::raw_pwrite_stream &os,
                      ArrayRef<CompletedEntry> entries);

  /// The minimum time granularity for time trace profiler.
  DurationType granularity;

  /// The name of the process this profiler is running on.
  StringRef procName;

  /// The id of the process this profiler is running on.
  const llvm::sys::Process::Pid pid;

  /// System clock time when the session was begun.
  std::chrono::time_point<std::chrono::system_clock> beginningOfTime;

  /// Profiling clock time when the session was begun.
  const TimePointType startTime;

  /// Lock used to guard access to the running profilers.
  std::mutex lock;

  /// The set of running profilers for each thread.
  std::vector<std::unique_ptr<TimeTraceThreadProfiler>> profilers;

  /// The next available thread index, to ensure all ProfilerEventIds are
  /// globally unique across all thread profilers.
  uint16_t nextThreadIndex = 0;

  /// A set of active thread profiler contexts.
  DenseSet<ThreadProfilerContext *> threadProfilerContexts;

  SmallVector<std::string> inputShapes;
};

/// For internal use only. Returns true if profiling is active.
bool timeTraceProfilerIsActive();

} // namespace ProfilingDetail

//===----------------------------------------------------------------------===//
// TimeTraceProfiler
//===----------------------------------------------------------------------===//

/// This class represents the main time trace profiler, of which only one should
/// ever be active at a given time.
struct TimeTraceProfiler {
  /// Initialize the time trace profiler. This should be constructed from the
  /// main thread.
  TimeTraceProfiler(unsigned timeTraceGranularity, StringRef procName);

  /// Destroy the time trace profiler. This should be destroyed from the
  /// main thread.
  ~TimeTraceProfiler();

  /// Append given input shape to internal list.
  /// These will be included in metadata written to output stream.
  void addInputShape(const std::string &shape);

  /// Write profiling data to a file.
  /// The function will write to preferredFileName if provided, if not then will
  /// write to fallbackFileName appending .time-trace. Returns a StringError
  /// indicating a failure if the function is unable to open the file for
  /// writing.
  ErrorOrSuccess write(StringRef preferredFileName, StringRef fallbackFileName);

  /// Writes the profiling data in JSON form to os. Visible for testing.
  void writeJSONForTesting(llvm::raw_pwrite_stream &os);

  /// Make sure all internable strings are captured in all profiling entries.
  void intern();
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
///   -- Stop the entry's clock, and move the entry into the profiling
///   -- database.
///   void record() &&
///
template <bool Enabled>
struct ProfilerEntry {};

/// Disabled profiling entry. Everything is a no-op.
template <>
struct ProfilerEntry<false> {
  // No copy, only move.
  ProfilerEntry(const ProfilerEntry &) = delete;
  ProfilerEntry &operator=(const ProfilerEntry &) = delete;
  ProfilerEntry(ProfilerEntry &&) = default;
  ProfilerEntry &operator=(ProfilerEntry &&) = default;

  ProfilerEntry() = default;

  static constexpr bool isEnabled() { return false; }

  template <typename... Args>
  static ProfilerEntry create(Args &&...args) {
    return {};
  }

  template <typename... Args>
  static ProfilerEntry createWithParent(ProfilerEventId parentId,
                                        Args &&...args) {
    return {};
  }

  template <typename... Args>
  static void sample(uint64_t value, Args &&...args) {}

  bool empty() const { return true; }
  ProfilerEventId getId() const { return 0; }

  void record() && {}
};

/// Enabled profiling entry. Entries are created only if the profiler is active.
template <>
struct ProfilerEntry<true> {
  // No copy, only move.
  ProfilerEntry(const ProfilerEntry &) = delete;
  ProfilerEntry &operator=(const ProfilerEntry &) = delete;
  ProfilerEntry(ProfilerEntry &&) = default;
  ProfilerEntry &operator=(ProfilerEntry &&) = default;

  ProfilerEntry() = default;

  static constexpr bool isEnabled() { return true; }

  template <typename... Args>
  static ProfilerEntry create(Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      return ProfilerEntry(ctx->begin(std::forward<Args>(args)...));
    return {};
  }

  template <typename... Args>
  static ProfilerEntry createWithParent(ProfilerEventId parentId,
                                        Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      return ProfilerEntry(
          ctx->beginWithParent(parentId, std::forward<Args>(args)...));
    return {};
  }

  template <typename... Args>
  static void sample(uint64_t value, Args &&...args) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      ctx->sample(value, std::forward<Args>(args)...);
  }

  bool empty() const { return id == 0; }
  ProfilerEventId getId() const { return id; }

  void record() && {
    if (id == 0)
      return;
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      return ctx->end(id);
  }

private:
  ProfilerEntry(ProfilerEventId id) : id(id) {}

  ProfilerEventId id = 0;
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

  explicit TimeTraceScope(ProfilerEntry<Enabled> entry)
      : entry(std::move(entry)) {}
  explicit TimeTraceScope(StringRef name, StringRef detail = {})
      : entry(ProfilerEntry<Enabled>::create(name, detail)) {}
  TimeTraceScope(StringRef name, ProfilerPrintFn printFn)
      : entry(ProfilerEntry<Enabled>::create(name, printFn)) {}

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
void timeTraceProfilerBegin(StringRef name, StringRef detail = {}) {
  if constexpr (Enabled) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      ctx->beginAndPush(name, detail);
  }
}

/// Manually end the last time section. However if Enabled is false then
/// the method is a no-op.
template <bool Enabled = true>
void timeTraceProfilerEnd() {
  if constexpr (Enabled) {
    if (auto *ctx = ProfilingDetail::ThreadProfilerContext::get())
      ctx->endAndPop();
  }
}

} // namespace M

#endif // SUPPORT_PROFILING_TIMEPROFILER_H
