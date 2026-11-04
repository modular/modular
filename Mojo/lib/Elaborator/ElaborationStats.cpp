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

struct Summary {
  llvm::SmallVector<LoopRow> rows;
  int64_t totalOps = 0;
};
} // namespace

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

/// Groups `records` by loop and generator, sorted by the operations produced.
static Summary summarize(llvm::ArrayRef<ComptimeForRecord> records) {
  llvm::MapVector<std::pair<Location, StringAttr>, LoopRow> groups;
  Summary summary;
  for (const ComptimeForRecord &record : records) {
    auto [it, inserted] = groups.try_emplace(
        std::make_pair(record.loc, record.generator),
        LoopRow{record.loc->findInstanceOf<FileLineColLoc>(), record.loc,
                // The location identifies the loop, so the parameter and
                // argument lists in the symbol name only add noise.
                record.generator.getValue().take_until(
                    [](char c) { return c == '[' || c == '('; })});
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

void M::KGEN::reportElaborationStats(bool isOffload,
                                     const CompilationOptions &options,
                                     std::vector<ComptimeForRecord> records) {
  std::string label = (isOffload ? "offload " : "host ") + options.targetTriple;
  if (!options.targetCpu.empty())
    label += " " + options.targetCpu;
  ElaborationStatsPart part{std::move(label), std::move(records)};

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

void M::KGEN::printElaborationStats(llvm::raw_ostream &os,
                                    const ElaborationStatsPart &part) {
  Summary summary = summarize(part.records);
  os << llvm::formatv(
      "comptime for: {0:N} loops, {1:N} instantiations, {2:N} ops produced "
      "(after elaboration, before inlining)\n",
      summary.rows.size(), part.records.size(), summary.totalOps);
  if (summary.rows.empty())
    return;

  os << "\n   ops produced  instances   iterations  body ops  loop\n";
  for (const LoopRow &row : summary.rows) {
    std::string iterations = std::to_string(row.maxIterations);
    if (row.minIterations != row.maxIterations)
      iterations = std::to_string(row.minIterations) + "-" + iterations;
    os << llvm::formatv("{0,15:N}  {1,9:N}  {2,11}  {3,8:N}  ", row.resultOps,
                        row.instances, iterations, row.bodyOps);
    if (row.fileLoc) {
      os << row.fileLoc.getFilename().getValue() << ':' << row.fileLoc.getLine()
         << ':' << row.fileLoc.getColumn();
    } else {
      row.loc.print(os);
    }
    os << "  @" << row.generator << '\n';
  }
}

void M::KGEN::writeElaborationStatsJSON(
    llvm::raw_ostream &os, llvm::ArrayRef<ElaborationStatsPart> parts) {
  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.array([&] {
    for (const ElaborationStatsPart &part : parts) {
      Summary summary = summarize(part.records);
      json.object([&] {
        json.attribute("pipeline", part.label);
        json.attributeObject("comptime_for", [&] {
          json.attribute("loop_count", int64_t(summary.rows.size()));
          json.attribute("instantiation_count", int64_t(part.records.size()));
          json.attribute("ops_produced", summary.totalOps);
          json.attributeArray("loops", [&] {
            for (const LoopRow &row : summary.rows) {
              json.object([&] {
                if (row.fileLoc) {
                  json.attribute("file", row.fileLoc.getFilename().getValue());
                  json.attribute("line", int64_t(row.fileLoc.getLine()));
                  json.attribute("column", int64_t(row.fileLoc.getColumn()));
                } else {
                  std::string text;
                  llvm::raw_string_ostream textOS(text);
                  row.loc.print(textOS);
                  json.attribute("location", text);
                }
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
      });
    }
  });
}
