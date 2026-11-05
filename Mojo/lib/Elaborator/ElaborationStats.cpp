//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/ElaborationStats.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <ctime>
#include <mutex>

using namespace M;
using namespace M::KGEN;
using namespace mlir;

namespace {
/// The parts of the report the driver is collecting. Global because the
/// offload elaborations run deep inside the host's, which has no handle to
/// pass them.
struct CollectorState {
  std::mutex mutex;
  bool enabled = false;
  llvm::SmallVector<ElaborationStatsPart> parts;
};

CollectorState &getCollectorState() {
  static CollectorState state;
  return state;
}

/// One loop of the report: its records merged.
struct LoopRow {
  FileLineColLoc fileLoc;
  Location loc;
  llvm::StringRef generator;
  int64_t instances = 0;
  int64_t minIterations = std::numeric_limits<int64_t>::max();
  int64_t maxIterations = 0;
  int64_t bodyOps = 0;
  int64_t resultOps = 0;
};

struct LoopSummary {
  llvm::SmallVector<LoopRow> rows;
  int64_t totalOps = 0;
};

/// The interpreter calls one operation made to one function, merged.
struct CallRow {
  FileLineColLoc fileLoc;
  Location loc;
  llvm::StringRef callee;
  int64_t cacheHits = 0;
  int64_t interpreted = 0;
  int64_t cpuNanos = 0;
  int64_t maxCPUNanos = 0;
};

/// The interpreter calls to one function, merged over the operations that made
/// them.
struct CalleeRow {
  llvm::StringRef callee;
  int64_t callSites = 0;
  int64_t cacheHits = 0;
  int64_t interpreted = 0;
  int64_t cpuNanos = 0;
  int64_t maxCPUNanos = 0;
};

struct CallSummary {
  llvm::SmallVector<CalleeRow> callees;
  llvm::SmallVector<CallRow> rows;
  int64_t cacheHits = 0;
  int64_t interpreted = 0;
  int64_t cpuNanos = 0;
};
} // namespace

/// The rows each interpreter table of the text report lists. A large compile
/// makes calls from thousands of places; the JSON report lists them all.
static constexpr size_t kMaxTextCallRows = 20;

int64_t M::KGEN::getThreadCPUNanos() {
  timespec time;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time);
  return int64_t(time.tv_sec) * 1'000'000'000 + time.tv_nsec;
}

void M::KGEN::enableElaborationStats() {
  CollectorState &state = getCollectorState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.enabled = true;
  state.parts.clear();
}

llvm::SmallVector<ElaborationStatsPart> M::KGEN::takeElaborationStats() {
  CollectorState &state = getCollectorState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.enabled = false;
  return std::move(state.parts);
}

static double nanosToSeconds(int64_t nanos) { return double(nanos) / 1e9; }

/// The symbol name without its parameter and argument lists. The report keys
/// rows by location, so those lists only add noise.
static llvm::StringRef getBaseName(StringAttr symbol) {
  return symbol.getValue().take_until(
      [](char c) { return c == '[' || c == '('; });
}

static void printLocation(llvm::raw_ostream &os, FileLineColLoc fileLoc,
                          Location loc) {
  if (fileLoc) {
    os << fileLoc.getFilename().getValue() << ':' << fileLoc.getLine() << ':'
       << fileLoc.getColumn();
    return;
  }
  loc.print(os);
}

static void writeLocationJSON(llvm::json::OStream &json, FileLineColLoc fileLoc,
                              Location loc) {
  if (fileLoc) {
    json.attribute("file", fileLoc.getFilename().getValue());
    json.attribute("line", int64_t(fileLoc.getLine()));
    json.attribute("column", int64_t(fileLoc.getColumn()));
    return;
  }
  std::string text;
  llvm::raw_string_ostream textOS(text);
  loc.print(textOS);
  json.attribute("location", text);
}

/// Groups `records` by loop and generator, sorted by the operations produced.
static LoopSummary summarizeLoops(llvm::ArrayRef<ComptimeForRecord> records) {
  llvm::MapVector<std::pair<Location, StringAttr>, LoopRow> groups;
  LoopSummary summary;
  for (const ComptimeForRecord &record : records) {
    auto [it, inserted] =
        groups.try_emplace(std::make_pair(record.loc, record.generator),
                           LoopRow{record.loc->findInstanceOf<FileLineColLoc>(),
                                   record.loc, getBaseName(record.generator)});
    LoopRow &row = it->second;
    ++row.instances;
    row.minIterations = std::min(row.minIterations, record.iterations);
    row.maxIterations = std::max(row.maxIterations, record.iterations);
    row.bodyOps = std::max(row.bodyOps, record.bodyOps);
    row.resultOps += record.resultOps;
    summary.totalOps += record.resultOps;
  }
  for (auto &[key, row] : groups)
    summary.rows.push_back(std::move(row));
  llvm::stable_sort(summary.rows, [](const LoopRow &lhs, const LoopRow &rhs) {
    return lhs.resultOps > rhs.resultOps;
  });
  return summary;
}

template <typename RowT>
static void sortByCPUTime(llvm::SmallVectorImpl<RowT> &rows) {
  llvm::stable_sort(rows, [](const RowT &lhs, const RowT &rhs) {
    return std::make_pair(lhs.cpuNanos, lhs.cacheHits) >
           std::make_pair(rhs.cpuNanos, rhs.cacheHits);
  });
}

/// Groups `calls` by callee, and by location and callee, each sorted by CPU
/// time. A callee's instantiations share a row, as one operation in a generic
/// function calls a different one for each instantiation of its own.
static CallSummary summarizeCalls(llvm::ArrayRef<InterpreterCallRecord> calls) {
  llvm::MapVector<std::pair<Location, llvm::StringRef>, CallRow> groups;
  CallSummary summary;
  for (const InterpreterCallRecord &call : calls) {
    llvm::StringRef callee = getBaseName(call.callee);
    auto [it, inserted] = groups.try_emplace(
        std::make_pair(call.loc, callee),
        CallRow{call.loc->findInstanceOf<FileLineColLoc>(), call.loc, callee});
    CallRow &row = it->second;
    if (!call.cpuNanos) {
      ++row.cacheHits;
      ++summary.cacheHits;
      continue;
    }
    ++row.interpreted;
    row.cpuNanos += *call.cpuNanos;
    row.maxCPUNanos = std::max(row.maxCPUNanos, *call.cpuNanos);
    ++summary.interpreted;
    summary.cpuNanos += *call.cpuNanos;
  }
  llvm::MapVector<llvm::StringRef, CalleeRow> callees;
  for (auto &[key, row] : groups) {
    CalleeRow &callee =
        callees.try_emplace(row.callee, CalleeRow{row.callee}).first->second;
    ++callee.callSites;
    callee.cacheHits += row.cacheHits;
    callee.interpreted += row.interpreted;
    callee.cpuNanos += row.cpuNanos;
    callee.maxCPUNanos = std::max(callee.maxCPUNanos, row.maxCPUNanos);
    summary.rows.push_back(std::move(row));
  }
  for (auto &[name, callee] : callees)
    summary.callees.push_back(callee);
  sortByCPUTime(summary.callees);
  sortByCPUTime(summary.rows);
  return summary;
}

void M::KGEN::reportElaborationStats(
    bool isOffload, const CompilationOptions &options,
    std::vector<ComptimeForRecord> comptimeForRecords,
    std::optional<std::vector<InterpreterCallRecord>> interpreterCalls) {
  std::string label = (isOffload ? "offload " : "host ") + options.targetTriple;
  if (!options.targetCpu.empty())
    label += " " + options.targetCpu;
  ElaborationStatsPart part{std::move(label), std::move(comptimeForRecords),
                            std::move(interpreterCalls)};

  CollectorState &state = getCollectorState();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.enabled) {
      state.parts.push_back(std::move(part));
      return;
    }
  }

  llvm::errs() << "===--- Elaboration statistics (--elaboration-stats): "
               << part.label << " ===\n";
  printElaborationStats(llvm::errs(), part);
}

static void printComptimeFors(llvm::raw_ostream &os,
                              llvm::ArrayRef<ComptimeForRecord> records) {
  LoopSummary summary = summarizeLoops(records);
  os << llvm::formatv(
      "comptime for: {0:N} loops, {1:N} instantiations, {2:N} ops produced "
      "(after elaboration, before inlining)\n",
      summary.rows.size(), records.size(), summary.totalOps);
  if (summary.rows.empty())
    return;

  os << "\n   ops produced  instances   iterations  body ops  loop\n";
  for (const LoopRow &row : summary.rows) {
    std::string iterations = std::to_string(row.maxIterations);
    if (row.minIterations != row.maxIterations)
      iterations = std::to_string(row.minIterations) + "-" + iterations;
    os << llvm::formatv("{0,15:N}  {1,9:N}  {2,11}  {3,8:N}  ", row.resultOps,
                        row.instances, iterations, row.bodyOps);
    printLocation(os, row.fileLoc, row.loc);
    os << "  @" << row.generator << '\n';
  }
}

static void printInterpreterCalls(llvm::raw_ostream &os,
                                  llvm::ArrayRef<InterpreterCallRecord> calls) {
  CallSummary summary = summarizeCalls(calls);
  os << llvm::formatv("interpreter: {0:N} calls, {1:N} cache hits, {2:N} "
                      "interpreted in {3:F4} s CPU\n",
                      calls.size(), summary.cacheHits, summary.interpreted,
                      nanosToSeconds(summary.cpuNanos));
  if (summary.rows.empty())
    return;

  auto printMoreRows = [&](size_t rows) {
    if (rows > kMaxTextCallRows) {
      os << llvm::formatv("... {0:N} more rows; --timing-json lists them all\n",
                          rows - kMaxTextCallRows);
    }
  };

  os << "\nBy callee, each function's calls summed over all its call sites:\n"
        "          CPU s  interpreted  cache hits     max s  call sites  "
        "callee\n";
  for (const CalleeRow &row :
       llvm::ArrayRef(summary.callees).take_front(kMaxTextCallRows)) {
    os << llvm::formatv(
        "{0,15:F4}  {1,11:N}  {2,10:N}  {3,8:F4}  {4,10:N}  @{5}\n",
        nanosToSeconds(row.cpuNanos), row.interpreted, row.cacheHits,
        nanosToSeconds(row.maxCPUNanos), row.callSites, row.callee);
  }
  printMoreRows(summary.callees.size());

  os << "\nBy call site, one operation's calls to one function:\n"
        "          CPU s  interpreted  cache hits     max s  call site  "
        "callee\n";
  for (const CallRow &row :
       llvm::ArrayRef(summary.rows).take_front(kMaxTextCallRows)) {
    os << llvm::formatv("{0,15:F4}  {1,11:N}  {2,10:N}  {3,8:F4}  ",
                        nanosToSeconds(row.cpuNanos), row.interpreted,
                        row.cacheHits, nanosToSeconds(row.maxCPUNanos));
    printLocation(os, row.fileLoc, row.loc);
    os << "  @" << row.callee << '\n';
  }
  printMoreRows(summary.rows.size());
}

void M::KGEN::printElaborationStats(llvm::raw_ostream &os,
                                    const ElaborationStatsPart &part) {
  printComptimeFors(os, part.comptimeForRecords);
  if (part.interpreterCalls) {
    os << '\n';
    printInterpreterCalls(os, *part.interpreterCalls);
  }
}

void M::KGEN::writeElaborationStatsJSON(
    llvm::raw_ostream &os, llvm::ArrayRef<ElaborationStatsPart> parts) {
  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.array([&] {
    for (const ElaborationStatsPart &part : parts) {
      LoopSummary loops = summarizeLoops(part.comptimeForRecords);
      json.object([&] {
        json.attribute("pipeline", part.label);
        json.attributeObject("comptime_for", [&] {
          json.attribute("loop_count", int64_t(loops.rows.size()));
          json.attribute("instantiation_count",
                         int64_t(part.comptimeForRecords.size()));
          json.attribute("ops_produced", loops.totalOps);
          json.attributeArray("loops", [&] {
            for (const LoopRow &row : loops.rows) {
              json.object([&] {
                writeLocationJSON(json, row.fileLoc, row.loc);
                json.attribute("generator", row.generator);
                json.attribute("instances", row.instances);
                json.attributeObject("iterations", [&] {
                  json.attribute("min", row.minIterations);
                  json.attribute("max", row.maxIterations);
                });
                json.attribute("body_ops", row.bodyOps);
                json.attribute("ops_produced", row.resultOps);
              });
            }
          });
        });
        if (!part.interpreterCalls)
          return;
        CallSummary calls = summarizeCalls(*part.interpreterCalls);
        json.attributeObject("interpreter", [&] {
          json.attribute("call_count", int64_t(part.interpreterCalls->size()));
          json.attribute("cache_hit_count", calls.cacheHits);
          json.attribute("interpreted_count", calls.interpreted);
          json.attribute("cpu_seconds", nanosToSeconds(calls.cpuNanos));
          json.attributeArray("callees", [&] {
            for (const CalleeRow &row : calls.callees) {
              json.object([&] {
                json.attribute("callee", row.callee);
                json.attribute("call_sites", row.callSites);
                json.attribute("cache_hits", row.cacheHits);
                json.attribute("interpreted", row.interpreted);
                json.attribute("cpu_seconds", nanosToSeconds(row.cpuNanos));
                json.attribute("max_cpu_seconds",
                               nanosToSeconds(row.maxCPUNanos));
              });
            }
          });
          json.attributeArray("calls", [&] {
            for (const CallRow &row : calls.rows) {
              json.object([&] {
                writeLocationJSON(json, row.fileLoc, row.loc);
                json.attribute("callee", row.callee);
                json.attribute("cache_hits", row.cacheHits);
                json.attribute("interpreted", row.interpreted);
                json.attribute("cpu_seconds", nanosToSeconds(row.cpuNanos));
                json.attribute("max_cpu_seconds",
                               nanosToSeconds(row.maxCPUNanos));
              });
            }
          });
        });
      });
    }
  });
}
