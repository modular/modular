//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STRINGEXTRAS_H
#define SUPPORT_STRINGEXTRAS_H

#include "Support/LLVMForwardDecls.h"

namespace M {

/// Replace all occurrences of `oldStr` with `newStr` in `str`. Returns a new
/// string with the replacements. The replacement happens eagerly, from the
/// beginning of the string to the end (i.e. from lower indices to higher
/// indices).
void replaceAll(std::string &str, StringRef oldStr, StringRef newStr);

/// Pretty prints the memory in a human readable form (e.g. 1024 is
// printed as 1KB).
std::string humanMemorySize(size_t size);

/// Checks if the string represents something like True. For example, the
// strings "On" and "1" are considered true. The use case here is to have
// a utility function so that we are forgiving in the input that cannot
// be easily check (e.g. when specifying behavior via an environment variable)
bool isTrueLike(StringRef str);

/// Checks if the string represents something like False. For example, the
// strings "Off" and "0" are considered false. The use case here is to have
// a utility function so that we are forgiving in the input that cannot
// be easily check (e.g. when specifying behavior via an environment variable)
bool isFalseLike(StringRef str);

} // namespace M

#endif // SUPPORT_STRINGEXTRAS_H
