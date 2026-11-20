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

#include "Mojo/MojoParser/ParserStats.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/SharedState.h"
#include "ParserProfiler.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <mutex>

using namespace M;
using namespace M::KGEN::LIT;
using namespace mlir;
using llvm::SmallVector;
using llvm::StringRef;

namespace {
/// The parts of the report the driver is collecting. Global because the parse
/// entry points have no handle to the driver.
struct CollectorState {
  std::mutex mutex;
  bool enabled = false;
  llvm::SmallVector<ParserStatsPart> parts;
};

CollectorState &getCollectorState() {
  static CollectorState state;
  return state;
}
} // namespace

/// The declarations the text report lists. The JSON report lists them all.
static constexpr size_t kMaxTextRows = 30;

static constexpr std::array<ParserActivity, kNumParserActivities>
    kAllActivities = {ParserActivity::Signature, ParserActivity::Body,
                      ParserActivity::SyntheticSignature,
                      ParserActivity::SyntheticBody, ParserActivity::Bytecode};

StringRef M::KGEN::LIT::getParserActivityName(ParserActivity activity) {
  switch (activity) {
  case ParserActivity::Signature:
    return "signature";
  case ParserActivity::Body:
    return "body";
  case ParserActivity::SyntheticSignature:
    return "synthetic_signature";
  case ParserActivity::SyntheticBody:
    return "synthetic_body";
  case ParserActivity::Bytecode:
    return "bytecode";
  }
  llvm_unreachable("unknown parser activity");
}

int64_t ParserStatsRow::getTotalNanos() const {
  int64_t total = 0;
  for (int64_t nanos : this->nanos)
    total += nanos;
  return total;
}

void M::KGEN::LIT::enableParserStats() {
  CollectorState &state = getCollectorState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.enabled = true;
  state.parts.clear();
}

bool M::KGEN::LIT::isParserStatsEnabled() {
  CollectorState &state = getCollectorState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.enabled;
}

llvm::SmallVector<ParserStatsPart> M::KGEN::LIT::takeParserStats() {
  CollectorState &state = getCollectorState();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.enabled = false;
  return std::move(state.parts);
}

void M::KGEN::LIT::reportParserStats(ParserStatsPart part) {
  CollectorState &state = getCollectorState();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.enabled)
    state.parts.push_back(std::move(part));
}

//===----------------------------------------------------------------------===//
// ParserProfiler
//===----------------------------------------------------------------------===//

ParserProfiler::ParserProfiler(SharedState &shared)
    : shared(shared), start(Clock::now()), last(start) {}

static int64_t toNanos(std::chrono::steady_clock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

void ParserProfiler::charge(Clock::time_point now) {
  int64_t nanos = toNanos(now - last);
  if (stack.empty())
    unattributedNanos += nanos;
  else
    rows[stack.back().row].nanos[size_t(stack.back().activity)] += nanos;
  last = now;
}

static StringRef getDeclKindName(ASTDecl &decl) {
  switch (decl.getKind()) {
  case ASTDecl::Kind::Base:
    return "base";
  case ASTDecl::Kind::IR:
    return "ir";
  case ASTDecl::Kind::OtherOp:
    return "other";
  case ASTDecl::Kind::Var:
    return "var";
  case ASTDecl::Kind::Alias:
    return "alias";
  case ASTDecl::Kind::Conformance:
    return "conformance";
  case ASTDecl::Kind::Extension:
    return "extension";
  case ASTDecl::Kind::Def:
    return "def";
  case ASTDecl::Kind::FnSignature:
    return "fn_signature";
  case ASTDecl::Kind::EnumCase:
    return "enum_case";
  case ASTDecl::Kind::Import:
    return "import";
  case ASTDecl::Kind::TopLevelModule:
    return "top_level_module";
  case ASTDecl::Kind::FileModule:
    return "file_module";
  case ASTDecl::Kind::Package:
    return "package";
  case ASTDecl::Kind::Struct:
    return "struct";
  case ASTDecl::Kind::StructField:
    return "struct_field";
  case ASTDecl::Kind::Trait:
    return "trait";
  case ASTDecl::Kind::Witness:
    return "witness";
  case ASTDecl::Kind::ConformanceMember:
    return "conformance_member";
  }
  llvm_unreachable("unknown decl kind");
}

/// Returns the names of `decl` and its parents, outermost first, joined with
/// `.`. A declaration without a name shows as `<kind>`.
static std::string getQualifiedName(ASTDecl &decl) {
  SmallVector<std::string> names;
  for (ASTDecl *cur = &decl; cur; cur = cur->getParentDecl()) {
    if (isa<TopLevelModuleASTDecl>(cur))
      break;
    if (std::optional<StringRef> name = cur->getName(); name && !name->empty())
      names.push_back(name->str());
    else
      names.push_back(("<" + getDeclKindName(*cur) + ">").str());
  }
  std::reverse(names.begin(), names.end());
  return llvm::join(names, ".");
}

static std::string getLocationString(SharedState &shared, ASTDecl &decl) {
  mlir::Location loc = mlir::UnknownLoc::get(shared.getContext());
  if (decl.getLoc().isValid())
    loc = shared.translateLocation(decl.getLoc());
  else if (Operation *op = decl.getIfOperation())
    loc = op->getLoc();

  std::string result;
  llvm::raw_string_ostream os(result);
  // Debug info wraps the source location in a fused location.
  if (auto fileLoc = loc->findInstanceOf<mlir::FileLineColLoc>())
    os << fileLoc.getFilename().getValue() << ':' << fileLoc.getLine() << ':'
       << fileLoc.getColumn();
  else
    loc.print(os);
  return result;
}

unsigned ParserProfiler::getOrCreateRow(ASTDecl &decl) {
  auto [it, inserted] = rowIndex.try_emplace(&decl, rows.size());
  if (inserted) {
    ParserStatsRow &row = rows.emplace_back();
    row.name = getQualifiedName(decl);
    row.kind = getDeclKindName(decl).str();
    row.location = getLocationString(shared, decl);
  }
  return it->second;
}

void ParserProfiler::push(ASTDecl &decl, ParserActivity activity) {
  Clock::time_point now = Clock::now();
  charge(now);
  bool isNewRow = !rowIndex.contains(&decl);
  unsigned row = getOrCreateRow(decl);
  ++rows[row].frames;
  stack.push_back({row, activity});
  // Naming a new row is the profiler's work, not the parser's.
  if (isNewRow) {
    last = Clock::now();
    overheadNanos += toNanos(last - now);
  }
}

void ParserProfiler::pop() {
  assert(!stack.empty() && "unbalanced parser profiler frames");
  charge(Clock::now());
  stack.pop_back();
}

ParserStatsPart ParserProfiler::finish(StringRef label) {
  Clock::time_point now = Clock::now();
  charge(now);

  ParserStatsPart part;
  part.label = label.str();
  part.unattributedNanos = unattributedNanos;
  part.overheadNanos = overheadNanos;
  part.totalNanos = toNanos(now - start);

  // A declaration that the resolver replaces keeps its name and location, so
  // merge the rows of the old and the new one.
  llvm::StringMap<size_t> merged;
  for (ParserStatsRow &row : rows) {
    std::string key = row.kind + '\0' + row.name + '\0' + row.location;
    auto [it, inserted] = merged.try_emplace(key, part.rows.size());
    if (inserted) {
      part.rows.push_back(std::move(row));
      continue;
    }
    ParserStatsRow &existing = part.rows[it->second];
    for (size_t i = 0; i < kNumParserActivities; ++i)
      existing.nanos[i] += row.nanos[i];
    existing.frames += row.frames;
  }
  llvm::stable_sort(part.rows,
                    [](const ParserStatsRow &a, const ParserStatsRow &b) {
                      return a.getTotalNanos() > b.getTotalNanos();
                    });
  rows.clear();
  rowIndex.clear();
  return part;
}

//===----------------------------------------------------------------------===//
// Report
//===----------------------------------------------------------------------===//

static double nanosToSeconds(int64_t nanos) { return double(nanos) / 1e9; }

static int64_t sumActivity(const ParserStatsPart &part,
                           ParserActivity activity) {
  int64_t total = 0;
  for (const ParserStatsRow &row : part.rows)
    total += row.nanos[size_t(activity)];
  return total;
}

void M::KGEN::LIT::printParserStats(llvm::raw_ostream &os,
                                    const ParserStatsPart &part) {
  os << llvm::formatv("parse: {0:F4} s wall, {1:N} declarations\n",
                      nanosToSeconds(part.totalNanos), part.rows.size());

  auto printLine = [&](StringRef name, int64_t nanos) {
    double percent =
        part.totalNanos ? 100.0 * double(nanos) / double(part.totalNanos) : 0.0;
    os << llvm::formatv("{0,12:F4}  {1,6:F1}%  {2}\n", nanosToSeconds(nanos),
                        percent, name);
  };
  os << "\nSelf time by activity:\n";
  os << llvm::formatv("{0,12}  {1,7}  {2}\n", "self s", "share", "activity");
  for (ParserActivity activity : kAllActivities)
    printLine(getParserActivityName(activity), sumActivity(part, activity));
  printLine("unattributed", part.unattributedNanos);
  printLine("profiler_overhead", part.overheadNanos);

  os << "\nDeclarations by self time:\n";
  os << llvm::formatv("{0,12}  {1,12}  {2,12}  {3,12}  {4,8}  {5}\n", "self s",
                      "signature s", "body s", "other s", "frames",
                      "declaration");
  for (const ParserStatsRow &row :
       llvm::ArrayRef(part.rows).take_front(kMaxTextRows)) {
    int64_t total = row.getTotalNanos();
    int64_t signature = row.nanos[size_t(ParserActivity::Signature)];
    int64_t body = row.nanos[size_t(ParserActivity::Body)];
    os << llvm::formatv("{0,12:F4}  {1,12:F4}  {2,12:F4}  {3,12:F4}  {4,8:N}  "
                        "{5} {6}  {7}\n",
                        nanosToSeconds(total), nanosToSeconds(signature),
                        nanosToSeconds(body),
                        nanosToSeconds(total - signature - body), row.frames,
                        row.kind, row.name, row.location);
  }
  if (part.rows.size() > kMaxTextRows)
    os << llvm::formatv("... {0:N} more; the JSON report lists them all\n",
                        part.rows.size() - kMaxTextRows);
}

void M::KGEN::LIT::writeParserStatsJSON(llvm::raw_ostream &os,
                                        llvm::ArrayRef<ParserStatsPart> parts) {
  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.array([&] {
    for (const ParserStatsPart &part : parts) {
      json.object([&] {
        json.attribute("module", part.label);
        json.attribute("total_seconds", nanosToSeconds(part.totalNanos));
        json.attribute("unattributed_seconds",
                       nanosToSeconds(part.unattributedNanos));
        json.attribute("profiler_overhead_seconds",
                       nanosToSeconds(part.overheadNanos));
        json.attributeObject("activities", [&] {
          for (ParserActivity activity : kAllActivities)
            json.attribute(getParserActivityName(activity),
                           nanosToSeconds(sumActivity(part, activity)));
        });
        json.attributeArray("declarations", [&] {
          for (const ParserStatsRow &row : part.rows) {
            json.object([&] {
              json.attribute("name", row.name);
              json.attribute("kind", row.kind);
              json.attribute("location", row.location);
              json.attribute("self_seconds",
                             nanosToSeconds(row.getTotalNanos()));
              json.attribute("frames", row.frames);
              json.attributeObject("activities", [&] {
                for (ParserActivity activity : kAllActivities)
                  if (int64_t nanos = row.nanos[size_t(activity)])
                    json.attribute(getParserActivityName(activity),
                                   nanosToSeconds(nanos));
              });
            });
          }
        });
      });
    }
  });
}
