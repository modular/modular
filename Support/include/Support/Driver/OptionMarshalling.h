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
//
// This file parses options that carry `OptParser.td` marshalling information.
// `llvm-tblgen -gen-opt-parser-defs` emits one
// `<PREFIX>OPTION_WITH_MARSHALLING` row per such option; expanding the rows
// with `PARSE_OPTION_WITH_MARSHALLING` stores each option's value in the field
// its key path names.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DRIVER_OPTIONMARSHALLING_H
#define SUPPORT_DRIVER_OPTIONMARSHALLING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"

#include <optional>
#include <string>
#include <vector>

namespace M {

/// The errors that normalizers report for invalid option values.
using MarshallingErrors = std::vector<std::string>;

inline std::optional<bool> normalizeSimpleFlag(llvm::opt::OptSpecifier opt, int,
                                               const llvm::opt::ArgList &args,
                                               MarshallingErrors &) {
  if (args.hasArg(opt))
    return true;
  return std::nullopt;
}

inline std::optional<bool>
normalizeSimpleNegativeFlag(llvm::opt::OptSpecifier opt, int,
                            const llvm::opt::ArgList &args,
                            MarshallingErrors &) {
  if (args.hasArg(opt))
    return false;
  return std::nullopt;
}

inline std::optional<std::string>
normalizeString(llvm::opt::OptSpecifier opt, int,
                const llvm::opt::ArgList &args, MarshallingErrors &) {
  if (const llvm::opt::Arg *arg = args.getLastArg(opt))
    return std::string(arg->getValue());
  return std::nullopt;
}

/// A value that does not parse as an integer leaves the option at its
/// default.
template <typename IntTy>
std::optional<IntTy> normalizeStringIntegral(llvm::opt::OptSpecifier opt, int,
                                             const llvm::opt::ArgList &args,
                                             MarshallingErrors &) {
  const llvm::opt::Arg *arg = args.getLastArg(opt);
  if (!arg)
    return std::nullopt;
  IntTy value;
  if (llvm::StringRef(arg->getValue()).getAsInteger(10, value))
    return std::nullopt;
  return value;
}

inline std::optional<std::vector<std::string>>
normalizeStringVector(llvm::opt::OptSpecifier opt, int,
                      const llvm::opt::ArgList &args, MarshallingErrors &) {
  if (!args.hasArg(opt))
    return std::nullopt;
  return args.getAllArgValues(opt);
}

inline std::optional<llvm::SmallVector<std::string>>
normalizeStringSmallVector(llvm::opt::OptSpecifier opt, int,
                           const llvm::opt::ArgList &args,
                           MarshallingErrors &) {
  if (!args.hasArg(opt))
    return std::nullopt;
  return llvm::to_vector_of<std::string>(args.getAllArgValues(opt));
}

} // namespace M

/// Parses one marshalled option from ARGS into its key path, adding any error
/// to ERRORS. Expand this where the subcommand's `options::OPT_*` enum and the
/// key paths' variables are visible. A field keeps its C++ initializer unless
/// the option is present, so the structs, not the `.td` records, hold the
/// defaults. Aliases share the key path of the option they alias, so their
/// rows are skipped.
#define PARSE_OPTION_WITH_MARSHALLING(                                         \
    ARGS, ERRORS, PREFIX_TYPE, SPELLING_OFFSET, ID, KIND, GROUP, ALIAS,        \
    ALIASARGS, FLAGS, VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS,       \
    METAVAR, VALUES, SUBCOMMANDIDS_OFFSET, SHOULD_PARSE, ALWAYS_EMIT, KEYPATH, \
    DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER, DENORMALIZER,     \
    TABLE_INDEX)                                                               \
  if (options::OPT_##ALIAS == options::OPT_INVALID && (SHOULD_PARSE))          \
    if (auto value = NORMALIZER(options::OPT_##ID, TABLE_INDEX, ARGS, ERRORS)) \
      KEYPATH = static_cast<decltype(KEYPATH)>(*value);

#endif // SUPPORT_DRIVER_OPTIONMARSHALLING_H
