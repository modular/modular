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
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace KGEN;
using llvm::SmallVector;
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
