//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOVARIANTTYPENAMEPARSER_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOVARIANTTYPENAMEPARSER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M::KGEN::Mojo {

/// One `Variant` type parameter's fully-qualified name (e.g.
/// `"std::collections::string::string::String"`) and its short display name
/// (e.g. `"String"`). The short form is what the summary shows to the user;
/// the fully-qualified form is what we match against to decide whether a
/// type-specific decoder (like the stdlib `String` inline/heap reader)
/// applies.
struct VariantTypeName {
  std::string fullName;
  std::string displayName;
};

/// Parse the mangled storage type-name of a stdlib `Variant` and return
/// one `VariantTypeName` per arm in declaration order.
///
/// Two wire formats are handled:
///
/// DWARF (non-REPL):
///   `!lit.struct<..._DefaultVariantStorage[... [@\22mod::Foo\22,
///   @\22Bar\22]]">` where `\22` is LLDB's octal encoding of `"`.
///
/// REPL:
///   `!lit.struct<..._DefaultVariantStorage<... [@mod::@Foo, @mod::@Bar]>>`
///   where type references are `@`-delimited components without quoting.
///
/// On some targets (observed on aarch64 Linux, not on Darwin arm64 or
/// Linux x86_64) the DWARF symbol carries a trailing `TypeList` parameter
/// that embeds a second `[@"...", @"..."]` listing the same types. The
/// parser stops at the FIRST closing `"]` after the opening `[@"` so that
/// trailing list can't merge with the real Ts pack.
///
/// Returns an empty vector for unrecognized / malformed input.
llvm::SmallVector<VariantTypeName>
extractVariantTypeNames(llvm::StringRef typeName);

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_FORMATTERS_MOJOVARIANTTYPENAMEPARSER_H
