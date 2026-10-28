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

#include "Mojo/ToolCommon/OptionNormalizers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace KGEN;
using llvm::SmallVector;
using llvm::StringLiteral;
using llvm::StringRef;

ErrorOrSuccess
M::validateCompilationOptions(const CompilationOptions &options) {
  bool asan = options.sanitizers.has(Sanitizers::kAddress);
  if (options.sharedLibasan && !asan)
    return Error(
        "cannot enable --shared-libasan without enabling --sanitize=address");
  if (!options.externalLibasan.empty()) {
    if (options.sharedLibasan)
      return Error("--external-libasan cannot be used with --shared-libasan");
    if (!asan)
      return Error("cannot use --external-libasan without enabling "
                   "--sanitize=address");
  }
  return success();
}

std::optional<DiagnosticFormat>
M::normalizeDiagnosticFormat(llvm::opt::OptSpecifier opt, int,
                             const llvm::opt::ArgList &args,
                             MarshallingErrors &errors) {
  const llvm::opt::Arg *arg = args.getLastArg(opt);
  if (!arg)
    return std::nullopt;
  StringLiteral kDiagnosticFormatText = "text";
  StringLiteral kDiagnosticFormatJSON = "json";
  StringRef format = arg->getValue();
  if (format == kDiagnosticFormatText)
    return DiagnosticFormat::Text;
  if (format == kDiagnosticFormatJSON)
    return DiagnosticFormat::JSON;
  errors.push_back(
      llvm::formatv("invalid diagnostic format '{0}', expected one of: `{1}` "
                    "(the default), or `{2}`",
                    format, kDiagnosticFormatText, kDiagnosticFormatJSON));
  return std::nullopt;
}

/// Returns the last value of `opt`, or nothing if `opt` is absent or empty.
static std::optional<StringRef>
getLastNonEmptyValue(llvm::opt::OptSpecifier opt,
                     const llvm::opt::ArgList &args) {
  StringRef value = args.getLastArgValue(opt);
  if (value.empty())
    return std::nullopt;
  return value;
}

std::optional<unsigned>
M::normalizeOptimizationLevel(llvm::opt::OptSpecifier opt, int,
                              const llvm::opt::ArgList &args,
                              MarshallingErrors &errors) {
  const llvm::opt::Arg *arg = args.getLastArg(opt);
  if (!arg)
    return std::nullopt;
  StringRef level = arg->getValue();
  if (level.size() == 1 && '0' <= level[0] && level[0] <= '3')
    return level[0] - '0';
  errors.push_back(llvm::formatv("invalid optimization level '{0}', expected "
                                 "number from 0-3 inclusive",
                                 level));
  return std::nullopt;
}

std::optional<FpMode> M::normalizeFpMode(llvm::opt::OptSpecifier opt, int,
                                         const llvm::opt::ArgList &args,
                                         MarshallingErrors &errors) {
  std::vector<std::string> values = args.getAllArgValues(opt);
  if (values.empty())
    return std::nullopt;
  // Parse the list of `feature=value` items.
  FpMode mode;
  for (StringRef value : values) {
    SmallVector<StringRef> items;
    value.split(items, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef item : items) {
      switch (parseFpModeItem(item, mode)) {
      case FpModeItemStatus::Ok:
        break;
      case FpModeItemStatus::UnknownFeature:
        errors.push_back(llvm::formatv("invalid fp-mode '{0}', the only "
                                       "supported feature is 'contract'",
                                       item));
        return std::nullopt;
      case FpModeItemStatus::InvalidValue:
        errors.push_back(llvm::formatv("invalid fp-mode '{0}', expected "
                                       "'contract=fast' or 'contract=off'",
                                       item));
        return std::nullopt;
      }
    }
  }
  return mode;
}

std::optional<CompilationOptions::DebugInfoLevel>
M::normalizeDebugLevel(llvm::opt::OptSpecifier opt, int,
                       const llvm::opt::ArgList &args,
                       MarshallingErrors &errors) {
  const llvm::opt::Arg *arg = args.getLastArg(opt);
  if (!arg)
    return std::nullopt;
  StringLiteral kDebugLevelNone = "none";
  StringLiteral kDebugLevelLineTables = "line-tables";
  StringLiteral kDebugLevelFull = "full";
  StringRef level = arg->getValue();
  if (!llvm::is_contained(
          {kDebugLevelNone, kDebugLevelLineTables, kDebugLevelFull}, level)) {
    errors.push_back(llvm::formatv("invalid debug level '{0}', expected one "
                                   "of: `{1}` (the default value), `{2}`, or "
                                   "`{3}`",
                                   level, kDebugLevelNone,
                                   kDebugLevelLineTables, kDebugLevelFull));
    return std::nullopt;
  }
  return llvm::StringSwitch<CompilationOptions::DebugInfoLevel>(level)
      .Case(kDebugLevelNone, CompilationOptions::kNoDebug)
      .Case(kDebugLevelLineTables, CompilationOptions::kLineTablesOnly)
      .Case(kDebugLevelFull, CompilationOptions::kFullDebugInfo);
}

std::optional<size_t> M::normalizeNumThreads(llvm::opt::OptSpecifier opt, int,
                                             const llvm::opt::ArgList &args,
                                             MarshallingErrors &errors) {
  std::optional<StringRef> value = getLastNonEmptyValue(opt, args);
  if (!value)
    return std::nullopt;
  size_t numThreads;
  if (llvm::to_integer(*value, numThreads))
    return numThreads;
  errors.push_back("Invalid number of threads: " + value->str());
  return std::nullopt;
}

std::optional<int64_t>
M::normalizeLoopUnrollingWarnThreshold(llvm::opt::OptSpecifier opt, int,
                                       const llvm::opt::ArgList &args,
                                       MarshallingErrors &errors) {
  std::optional<StringRef> value = getLastNonEmptyValue(opt, args);
  if (!value)
    return std::nullopt;
  int64_t threshold;
  if (llvm::to_integer(*value, threshold))
    return threshold;
  errors.push_back(("invalid loop-unrolling-warn-threshold'" + *value +
                    "', expected an integer number")
                       .str());
  return std::nullopt;
}

std::optional<int>
M::normalizeElaborationErrorLimit(llvm::opt::OptSpecifier opt, int,
                                  const llvm::opt::ArgList &args,
                                  MarshallingErrors &errors) {
  std::optional<StringRef> value = getLastNonEmptyValue(opt, args);
  if (!value)
    return std::nullopt;
  int limit;
  if (llvm::to_integer(*value, limit))
    return limit;
  errors.push_back(("invalid elaboration-error-limit'" + *value +
                    "', expected an integer number")
                       .str());
  return std::nullopt;
}

std::optional<CompilationOptions::ErrorVerboseLevel>
M::normalizeElaborationErrorVerbose(llvm::opt::OptSpecifier opt, int,
                                    const llvm::opt::ArgList &args,
                                    MarshallingErrors &errors) {
  const llvm::opt::Arg *arg = args.getLastArg(opt);
  if (!arg)
    return std::nullopt;
  StringLiteral kNoParams = "no-params";
  StringLiteral kSimpleParams = "simple-params";
  StringLiteral kAllParams = "all-params";
  StringRef verbose = arg->getValue();
  if (!llvm::is_contained({kNoParams, kSimpleParams, kAllParams}, verbose)) {
    errors.push_back(llvm::formatv(
        "invalid elaboration-error-verbose '{0}', expected one of: "
        "`{1}` (the default value), `{2}`, or `{3}`",
        verbose, kNoParams, kSimpleParams, kAllParams));
    return std::nullopt;
  }
  return llvm::StringSwitch<CompilationOptions::ErrorVerboseLevel>(verbose)
      .Case(kNoParams, CompilationOptions::kNoParams)
      .Case(kSimpleParams, CompilationOptions::kSimpleParams)
      .Case(kAllParams, CompilationOptions::kAllParams);
}

std::optional<unsigned>
M::normalizeElaborationMaxDepth(llvm::opt::OptSpecifier opt, int,
                                const llvm::opt::ArgList &args,
                                MarshallingErrors &errors) {
  std::optional<StringRef> value = getLastNonEmptyValue(opt, args);
  if (!value)
    return std::nullopt;
  unsigned depth;
  if (llvm::to_integer(*value, depth))
    return depth;
  errors.push_back(("invalid elaboration-max-depth'" + *value +
                    "', expected an unsigned number")
                       .str());
  return std::nullopt;
}

std::optional<Sanitizers> M::normalizeSanitizer(llvm::opt::OptSpecifier opt,
                                                int,
                                                const llvm::opt::ArgList &args,
                                                MarshallingErrors &errors) {
  std::optional<StringRef> value = getLastNonEmptyValue(opt, args);
  if (!value)
    return std::nullopt;
  Sanitizers sanitizers;
  if (*value == "address") {
    sanitizers.enable(Sanitizers::kAddress);
  } else if (*value == "thread") {
    sanitizers.enable(Sanitizers::kThread);
  } else {
    errors.push_back(("invalid sanitizer '" + *value +
                      "', expected one of: `address` or `thread`")
                         .str());
    return std::nullopt;
  }
  return sanitizers;
}

std::optional<CompilationOptions::DebugInfoLanguage>
M::normalizeDebugInfoLanguage(llvm::opt::OptSpecifier opt, int,
                              const llvm::opt::ArgList &args,
                              MarshallingErrors &errors) {
  std::optional<StringRef> value = getLastNonEmptyValue(opt, args);
  if (!value)
    return std::nullopt;
  if (*value == "C")
    return CompilationOptions::kLangC;
  if (*value == "Mojo")
    return CompilationOptions::kLangMojo;
  errors.push_back(("invalid debug info language '" + *value +
                    "', expected one of: `C` or `Mojo`")
                       .str());
  return std::nullopt;
}
